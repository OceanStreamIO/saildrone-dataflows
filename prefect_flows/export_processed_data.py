import logging
import os
import shutil
import sys
import traceback
import dask
import numpy as np

from datetime import datetime
from typing import List, Optional, Union
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from dask.distributed import Client, Lock

from prefect import flow, task
from prefect_dask import DaskTaskRunner, get_dask_client
from prefect.cache_policies import Inputs
from prefect.states import Completed
from prefect.artifacts import create_markdown_artifact
from prefect.futures import as_completed

from saildrone.store import FileSegmentService
from saildrone.process import apply_denoising, plot_and_upload_echograms
from saildrone.process.concat import merge_location_data, concatenate_and_rechunk
from saildrone.store import (PostgresDB, SurveyService, open_zarr_store, generate_container_name,
                             ensure_container_exists, save_zarr_store, save_dataset_to_netcdf)

from echopype.commongrid import compute_NASC, compute_MVBS

input_cache_policy = Inputs()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

load_dotenv()
DASK_CLUSTER_ADDRESS = os.getenv('DASK_CLUSTER_ADDRESS')
PROCESSED_CONTAINER_NAME = os.getenv('PROCESSED_CONTAINER_NAME')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 6))

CHUNKS = {"ping_time": 500, "range_sample": -1}
DEFAULT_TASK_TIMEOUT = 7_200  # 2 hours
MAX_RUNTIME_SECONDS = 3_300

AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
if not AZURE_STORAGE_CONNECTION_STRING:
    logging.error('AZURE_STORAGE_CONNECTION_STRING environment variable not set.')
    sys.exit(1)


class DenoiseOptions(BaseModel):
    def get(self, key, default_value=None):
        return getattr(self, key, default_value)


class MaskImpulseNoise(DenoiseOptions):
    depth_bin: int = Field(default=10,
                           description="Donwsampling bin size along vertical range variable (`range_var`) in meters.")
    num_side_pings: int = Field(default=2, description="Number of side pings to look at for the two-side comparison.")
    threshold: float = Field(default=10,
                             description="Impulse noise threshold value (in dB) for the two-side comparison.")
    range_var: str = Field(default='depth',
                           description="Vertical Axis Range Variable. Can be either \"depth\" or \"echo_range\".")


class MaskAttenuatedSignal(DenoiseOptions):
    upper_limit_sl: int = Field(default=180, description="Upper limit of deep scattering layer line (m).")
    lower_limit_sl: int = Field(default=300, description="Lower limit of deep scattering layer line (m).")
    num_side_pings: int = Field(default=15, description="Number of preceding & subsequent pings defining the block.")
    threshold: float = Field(default=10,
                             description="Attenuation signal threshold value (dB) for the ping-block comparison.")
    range_var: str = Field(default='depth',
                           description="Vertical Axis Range Variable. Can be either `depth` or `echo_range`.")


class TransientNoiseMask(DenoiseOptions):
    operation: str = Field(default='nanmedian',
                           description="Pooling function used in the pooled Sv aggregation, either 'nanmedian' or 'nanmean'.")
    depth_bin: int = Field(default=10, description="Bin size for depth calculation.")
    num_side_pings: int = Field(default=25, description="Number of side pings to include.")
    exclude_above: float = Field(default=250.0, description="Exclude data above this depth value.")
    threshold: float = Field(default=12.0,
                             description="Transient noise threshold value (in dB) for the pooling comparison.")
    range_var: str = Field(default='depth',
                           description="Vertical Range Variable. Can be either `depth` or `echo_range`.")


class RemoveBackgroundNoise(DenoiseOptions):
    ping_num: int = Field(default=5, description="Number of pings to obtain noise estimates")
    range_sample_num: int = Field(default=30, description="Number of range samples to consider.")
    background_noise_max: float = Field(default=-125,
                                        description="Maximum allowable background noise estimation (in dB).")
    SNR_threshold: float = Field(default=3.0,
                                 description="Signal-to-noise ratio threshold for background noise removal.")


@task(log_prints=True)
def get_worker_addresses(scheduler: str) -> list[str]:
    """
    A tiny prefect task that opens a short-lived Dask Client,
    asks the scheduler for the current workers, and returns *only*
    a list of their addresses (plain strings).
    """
    with Client(scheduler, name="discover-workers", timeout="5s") as c:
        return list(c.scheduler_info()["workers"])


@task(
    log_prints=True,
    timeout_seconds=DEFAULT_TASK_TIMEOUT,
    task_run_name="plot_echograms--{file_name}"
)
def task_plot_echograms(zarr_path, zarr_path_denoised, file_name, container_name, upload_path, chunks=None,
                        cmap='ocean_r'):
    try:
        ds = open_zarr_store(zarr_path, container_name=container_name, chunks=chunks, rechunk_after=True)

        plot_and_upload_echograms(ds,
                                  file_base_name=file_name,
                                  save_to_blobstorage=True,
                                  depth_var="depth",
                                  upload_path=upload_path,
                                  cmap=cmap,
                                  container_name=container_name)

        if zarr_path_denoised:
            ds_denoised = open_zarr_store(zarr_path_denoised, container_name=container_name, chunks=chunks,
                                          rechunk_after=True)

            plot_and_upload_echograms(ds_denoised,
                                      file_base_name=file_name,
                                      save_to_blobstorage=True,
                                      depth_var="depth",
                                      upload_path=upload_path,
                                      cmap=cmap,
                                      container_name=container_name)

        return Completed(message="plot_echograms completed successfully")
    except Exception as e:
        print(f"Error plotting echograms: {zarr_path}: ${str(e)}")
        traceback.print_exc()

        markdown_report = f"""# Error during plot_echograms
        Error occurred while plotting echograms: {zarr_path}

        {str(e)}
        
        ## Error details
        - **Error Message**: {str(e)}
        - **Traceback**: {traceback.format_exc()}
        """

        create_markdown_artifact(markdown_report)

        return Completed(message="plot_echograms completed with errors")


@task(
    log_prints=True,
    timeout_seconds=DEFAULT_TASK_TIMEOUT,
    task_run_name="save_to_netcdf--{file_name}"
)
def task_save_to_netcdf(zarr_path, zarr_path_denoised, nc_file_path, nc_file_path_denoised, container_name,
                        chunks=None):
    try:
        ds = open_zarr_store(zarr_path, container_name=container_name, chunks=chunks, rechunk_after=True)
        save_dataset_to_netcdf(ds, container_name=container_name, ds_path=nc_file_path)

        if zarr_path_denoised:
            ds_denoised = open_zarr_store(zarr_path_denoised, container_name=container_name, chunks=chunks,
                                          rechunk_after=True)
            save_dataset_to_netcdf(ds_denoised, container_name=container_name, ds_path=nc_file_path_denoised)

        return Completed(message="save_to_netcdf completed successfully")
    except Exception as e:
        print(f"Error saving to NetCDF: {zarr_path}: ${str(e)}")
        traceback.print_exc()

        markdown_report = f"""# Error during save_to_netcdf
        Error occurred while saving to NetCDF: {zarr_path}

        {str(e)}
        
        ## Error details
        - **Error Message**: {str(e)}
        - **Traceback**: {traceback.format_exc()}
        """

        create_markdown_artifact(markdown_report)

        return Completed(message="save_to_netcdf completed with errors")


@task(
    retries=3,
    retry_delay_seconds=60,
    cache_policy=input_cache_policy,
    retry_jitter_factor=0.1,
    refresh_cache=True,
    result_storage=None,
    timeout_seconds=DEFAULT_TASK_TIMEOUT,
    log_prints=True,
    task_run_name="process_file--{file_index}/{total}-{file_name}",
)
def process_single_file(file, file_name, source_container_name, cruise_id,
                        chunks,
                        export_container_name,
                        file_index,
                        total,
                        **kwargs):
    """
    Process a single file for Dask Futures: open it, merge location data, save the dataset, and return its path and frequency category.
    """
    zarr_path = f'{file_name}/{file_name}.zarr'
    location_data = file['location_data']
    file_freqs = file['file_freqs']
    file_start_time = file['file_start_time']
    file_end_time = file['file_end_time']
    file_id = file['id']
    file_name = file['file_name']

    compute_nasc_opt = kwargs.get("compute_nasc", False)
    save_to_netcdf = kwargs.get("save_to_netcdf", False)
    apply_seabed_mask = kwargs.get("apply_seabed_mask", False)
    plot_echograms = kwargs.get("plot_echograms", False)
    colormap = kwargs.get("colormap", "ocean_r")
    category = "short_pulse" if file_freqs == "38000.0,200000.0" else "long_pulse" if file_freqs == "38000.0" else cruise_id

    try:
        print(f"Processing file {zarr_path} with frequencies {file_freqs}")

        # Open the Zarr store lazily with Dask
        ds = open_zarr_store(zarr_path, cruise_id=cruise_id, container_name=source_container_name, chunks=chunks,
                             rechunk_after=True)

        # Merge location data
        ds = merge_location_data(ds, location_data)

        file_path = f"{category}/{file_name}/{file_name}.zarr"
        zarr_path = save_zarr_store(ds, container_name=export_container_name, zarr_path=file_path)

        # Apply denoising if specified
        denoising_applied = False
        sv_dataset_denoised = apply_denoising(ds, chunks_denoising=chunks, **kwargs)

        if sv_dataset_denoised is not None:
            denoising_applied = True
            file_path_denoised = f"{category}/{file_name}/{file_name}--denoised.zarr"
            save_zarr_store(sv_dataset_denoised, container_name=export_container_name, zarr_path=file_path_denoised)

        # if plot_echograms:
        #     upload_path = f"{category}/{file_name}"
        #     future = task_plot_echograms.submit(file_path,
        #                                         file_path_denoised,
        #                                         file_name,
        #                                         container_name=export_container_name,
        #                                         upload_path=upload_path,
        #                                         chunks=chunks,
        #                                         cmap=colormap)
        #     future.result()
        #
        # if save_to_netcdf:
        #     nc_file_path = f"{category}/{file_name}/{file_name}.nc"
        #     nc_file_path_denoised = None
        #
        #     if sv_dataset_denoised is not None:
        #         nc_file_path_denoised = f"{category}/{file_name}/{file_name}--denoised.nc"
        #
        #     task_save_to_netcdf.submit(file_path,
        #                                file_path_denoised,
        #                                nc_file_path,
        #                                nc_file_path_denoised,
        #                                container_name=export_container_name,
        #                                chunks=chunks)

        ##################################
        # compute NASC if specified
        zarr_path_nasc = None
        if compute_nasc_opt:
            ds_NASC = compute_NASC(
                sv_dataset_denoised,
                range_bin="10m",
                dist_bin="0.5nmi"
            )
            # Log-transform the NASC values for plotting
            ds_NASC["NASC_log"] = 10 * np.log10(ds_NASC["NASC"])
            ds_NASC["NASC_log"].attrs = {
                "long_name": "Log of NASC",
                "units": "m2 nmi-2"
            }
            file_path_nasc = f"{category}/{file_name}/{file_name}--NASC.zarr"
            zarr_path_nasc = save_zarr_store(ds_NASC,
                                             container_name=export_container_name,
                                             zarr_path=file_path_nasc)
            nc_file_path_nasc = f"{category}/{file_name}--NASC.nc"
            save_dataset_to_netcdf(ds_NASC, container_name=export_container_name, ds_path=nc_file_path_nasc)

        return denoising_applied, zarr_path_nasc, category
    except Exception as e:
        print(f"Error processing file: {zarr_path}: ${str(e)}")
        traceback.print_exc()

        markdown_report = f"""# Error report for {zarr_path}
        Error occurred while processing the file: {zarr_path}

        {str(e)}

        ## File details
        - **File Name**: {file_name}
        - **File ID**: {file_id}
        - **Cruise ID**: {cruise_id}
        - **Start Time**: {file_start_time}
        - **End Time**: {file_end_time}
        - **Location Data**: {location_data}

        ## Error details
        - **Error Message**: {str(e)}
        - **Traceback**: {traceback.format_exc()}
        """

        create_markdown_artifact(markdown_report)

        return Completed(message="Task completed with errors")


@task(
    retries=3,
    retry_delay_seconds=60,
    cache_policy=input_cache_policy,
    retry_jitter_factor=0.1,
    refresh_cache=True,
    result_storage=None,
    timeout_seconds=DEFAULT_TASK_TIMEOUT,
    log_prints=True,
    task_run_name="fan_out_side_tasks--{file_name}",
)
def fan_out_side_tasks(future, file_name, container_name, chunks, colormap, plot_echograms, save_to_netcdf):
    if future is None:
        return None

    denoising_applied, zarr_path_nasc, category = future
    upload_path = f"{category}/{file_name}"
    zarr_path = f"{category}/{file_name}/{file_name}.zarr"
    zarr_path_denoised = f"{category}/{file_name}/{file_name}--denoised.zarr" if denoising_applied else None

    futures = []
    if plot_echograms:
        future = task_plot_echograms.submit(
            zarr_path,
            zarr_path_denoised,
            file_name,
            container_name,
            upload_path,
            chunks,
            colormap
        )
        futures.append(future)

    if save_to_netcdf:
        nc_file_path = f"{category}/{file_name}/{file_name}.nc"
        nc_file_path_denoised = f"{category}/{file_name}/{file_name}--denoised.nc" if denoising_applied else None

        future_nc = task_save_to_netcdf.submit(
            zarr_path,
            zarr_path_denoised,
            nc_file_path,
            nc_file_path_denoised,
            container_name,
            chunks
        )
        futures.append(future_nc)

    _submit_and_collect(*futures)


@flow(
    task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS)
)
def export_processed_data(cruise_id: str,
                          source_container: str,
                          output_container: str,
                          start_datetime: Optional[datetime],
                          end_datetime: Optional[datetime],
                          plot_echograms: bool = False,
                          colormap: str = 'ocean_r',
                          compute_nasc: bool = False,
                          save_to_netcdf: bool = False,
                          mask_impulse_noise: Optional[MaskImpulseNoise] = None,
                          mask_attenuated_signal: Optional[MaskAttenuatedSignal] = None,
                          mask_transient_noise: Optional[TransientNoiseMask] = None,
                          remove_background_noise: Optional[RemoveBackgroundNoise] = None,
                          apply_seabed_mask: bool = False,
                          chunks_ping_time: int = CHUNKS['ping_time'],
                          chunks_depth: Optional[int] = CHUNKS['range_sample'],
                          batch_size: int = BATCH_SIZE
                          ):
    with PostgresDB() as db_connection:
        survey_service = SurveyService(db_connection)

        # Check if a survey with the given cruise_id exists
        survey_id = survey_service.get_survey_by_cruise_id(cruise_id)

        if not survey_id:
            survey_id = survey_service.insert_survey(cruise_id)
            logging.info(f"Inserted new survey with cruise_id: {cruise_id}")

        file_service = FileSegmentService(db_connection)
        condition = ""
        if start_datetime and end_datetime:
            condition += f" AND file_start_time > '{start_datetime}' AND file_end_time < '{end_datetime}'"

        files_list = file_service.get_files_by_survey_id(survey_id, condition=condition)

    total_files = len(files_list)
    logging.info(f"Total files to process: {total_files}")
    print(f"Total files to process: {total_files}")

    chunks = {
        'ping_time': chunks_ping_time,
        'depth': chunks_depth
    }

    export_container_name = output_container if output_container != '' else generate_container_name(cruise_id)
    if output_container == '':
        ensure_container_exists(export_container_name, public_access='container')

    in_flight = []
    workers = get_worker_addresses(scheduler=DASK_CLUSTER_ADDRESS)
    n_workers = len(workers)
    side_running_tasks = []

    for idx, file in enumerate(files_list):
        target_worker = workers[idx % n_workers]

        with dask.annotate(workers=[target_worker], allow_other_workers=False):
            future = process_single_file.submit(file,
                                                file_name=file['file_name'],
                                                source_container_name=source_container,
                                                cruise_id=cruise_id,
                                                chunks=chunks,
                                                export_container_name=export_container_name,
                                                file_index=idx,
                                                total=len(files_list),
                                                plot_echograms=plot_echograms,
                                                colormap=colormap,
                                                compute_nasc=compute_nasc,
                                                save_to_netcdf=save_to_netcdf,
                                                mask_impulse_noise=mask_impulse_noise,
                                                mask_attenuated_signal=mask_attenuated_signal,
                                                mask_transient_noise=mask_transient_noise,
                                                remove_background_noise=remove_background_noise,
                                                apply_seabed_mask=apply_seabed_mask
                                                )
        if plot_echograms or save_to_netcdf:
            side_tasks = fan_out_side_tasks.submit(future, file['file_name'], export_container_name, chunks, colormap,
                                                   plot_echograms, save_to_netcdf)
            side_running_tasks.append(side_tasks)

        in_flight.append(future)

        # Throttle when max concurrent tasks reached
        if len(in_flight) >= batch_size:
            finished = next(as_completed(in_flight))
            in_flight.remove(finished)

    # Wait for remaining tasks
    for remaining in in_flight + side_running_tasks:
        remaining.result()

    if os.path.exists('/tmp/oceanstream/netcdfdata'):
        shutil.rmtree('/tmp/oceanstream/netcdfdata', ignore_errors=True)

    print("All files have been processed.")


def _submit_and_collect(*calls) -> list:
    """Submit multiple tasks and synchronously wait for them to finish."""
    futures = [call for call in calls if call is not None]
    for future in as_completed(futures):
        future.wait()

    return futures


if __name__ == "__main__":
    client = Client(address=DASK_CLUSTER_ADDRESS)
    info = client.scheduler_info()
    workers_info = list(info["workers"])

    print(f"Running on dask cluster {info['address']} with {len(workers_info)} workers.")

    try:
        export_processed_data.serve(
            name='export-processed-data',
            parameters={
                'cruise_id': '',
                'source_container': PROCESSED_CONTAINER_NAME,
                'output_container': '',
                'start_datetime': None,
                'end_datetime': None,
                'plot_echograms': False,
                'colormap': 'ocean_r',
                'chunks_ping_time': 500,
                'chunks_depth': 500,
                'compute_nasc': False,
                'save_to_netcdf': False,
                'mask_impulse_noise': None,
                'mask_attenuated_signal': None,
                'mask_transient_noise': None,
                'remove_background_noise': None,
                'apply_seabed_mask': False,
                'batch_size': 4
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception occurred: {str(e)}', exc_info=True)
        sys.exit(1)
