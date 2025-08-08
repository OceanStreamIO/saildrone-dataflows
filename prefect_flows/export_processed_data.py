import logging
import os
import shutil
import sys
import traceback
from collections import defaultdict

from datetime import datetime, timedelta
from typing import List, Optional, Union
from dotenv import load_dotenv
from dask.distributed import Client, Lock

from prefect import flow, task
from prefect_dask import DaskTaskRunner, get_dask_client
from prefect.cache_policies import Inputs
from prefect.states import Completed, Failed
from prefect.artifacts import create_markdown_artifact
from prefect.futures import as_completed, PrefectFuture
from prefect.deployments import run_deployment

from prefect_flows.pydantic_models import NASC_Compute_Options, MVBS_Compute_Options, MaskImpulseNoise, \
    MaskAttenuatedSignal, TransientNoiseMask, RemoveBackgroundNoise, fill_missing_frequency_params

from saildrone.process import apply_denoising, plot_and_upload_echograms, get_files_list
from saildrone.process.workflow import compute_and_save_nasc, compute_and_save_mvbs
from saildrone.process.concat import merge_location_data, concatenate_and_rechunk
from saildrone.store import (PostgresDB, ExportService, open_zarr_store, generate_container_name,
                             ensure_container_exists, save_zarr_store, zip_and_save_netcdf_files,
                             save_dataset_to_netcdf)

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
CATEGORY_CONFIG = {
    "short_pulse": {
        "freq_key": "38000.0,200000.0",
        "zarr_name": "short_pulse{denoised}.zarr",
        "nc_name": "short_pulse{denoised}.nc",
        "file_base": "short_pulse{denoised}",
    },
    "long_pulse": {
        "freq_key": "38000.0",
        "zarr_name": "long_pulse{denoised}.zarr",
        "nc_name": "long_pulse{denoised}.nc",
        "file_base": "long_pulse{denoised}",
    },
    "exported_ds": {
        "freq_key": None,  # catch-all
        "zarr_name": "{batch_key}{denoised}.zarr",
        "nc_name": "{batch_key}{denoised}.nc",
        "file_base": "{batch_key}{denoised}"
    }
}

NETCDF_ROOT_DIR = os.getenv('NETCDF_ROOT_DIR', '/tmp/oceanstream/netcdfdata')
CHUNKS = {"ping_time": 1000, "depth": -1}
DEFAULT_TASK_TIMEOUT = 7_200  # 2 hours
MAX_RUNTIME_SECONDS = 3_300


def _nav_to_data_vars(ds):
    nav = ["latitude", "longitude", "speed_knots"]
    coords = [v for v in nav if v in ds.coords]

    if coords:
        ds = ds.reset_coords(coords)
    return ds


@task(log_prints=True)
def get_worker_addresses(scheduler: str) -> list[str]:
    """
    A tiny prefect task that opens a short-lived Dask Client,
    asks the scheduler for the current workers, and returns *only*
    a list of their addresses (plain strings).
    """
    with Client(scheduler, name="discover-workers", timeout="5s") as c:
        return list(c.scheduler_info()["workers"])



"""
################################################### NASC ###################################################
"""

@task(
    log_prints=True,
    retry_delay_seconds=60,
    cache_policy=input_cache_policy,
    refresh_cache=True,
    result_storage=None,
    task_run_name="compute_batch_nasc--{batch_key}"
)
def compute_batch_nasc(batch_results, batch_key, cruise_id, container_name, compute_nasc_options, plot_echograms=False,
                       save_to_netcdf=False, colormap='ocean_r', chunks=None):
    results = {}

    tag_for = {
        "short_pulse": "short_pulse",
        "long_pulse": "long_pulse",
        "exported_ds": batch_key,
    }

    def _run(pulse, tag):
        root = f"{batch_key}/{tag}"
        ds = open_zarr_store(f"{root}.zarr",
                             container_name=container_name,
                             rechunk_after=True,
                             chunks=chunks)
        ds = _nav_to_data_vars(ds)
        print('Computing NASC for pulse:', pulse, 'with tag:', tag, f'and root: {root}.zarr')
        print(ds.data_vars)

        nasc = compute_and_save_nasc(
            ds,
            zarr_path=f"{root}--nasc.zarr",
            compute_nasc_opts=compute_nasc_options,
            cruise_id=cruise_id,
            container_name=container_name,
        )
        results[pulse] = nasc

        if plot_echograms:
            plot_and_upload_echograms(
                nasc,
                file_base_name=f"{tag}--nasc",
                save_to_blobstorage=True,
                upload_path=f"{batch_key}",
                cmap=colormap,
                plot_var='NASC_log',
                container_name=container_name,
            )

        if save_to_netcdf:
            save_dataset_to_netcdf(
                nasc,
                container_name=container_name,
                ds_path=f"{root}--nasc.nc",
                base_local_temp_path=NETCDF_ROOT_DIR,
                is_temp_dir=False,
            )

    for pulse, tag in tag_for.items():
        if batch_results.get(pulse):
            _run(pulse, tag)

    return results


"""
################################################### MVBS ###################################################
"""


@task(
    log_prints=True,
    retry_delay_seconds=60,
    cache_policy=input_cache_policy,
    refresh_cache=True,
    result_storage=None,
    task_run_name="compute_batch_mvbs--{batch_key}"
)
def compute_batch_mvbs(batch_results, batch_key, cruise_id, container_name, compute_mvbs_options, plot_echograms=False,
                       save_to_netcdf=False, colormap='ocean_r', chunks=None):
    """
    Compute / plot / save MVBS for every pulse type present in *batch_results*.
    Returns {pulse_name: mvbs_dataset}.
    """
    results = {}

    tag_for = {
        "short_pulse": "short_pulse",
        "long_pulse": "long_pulse",
        "exported_ds": batch_key
    }

    def _run(pulse, tag):
        root = f"{batch_key}/{tag}"

        ds = open_zarr_store(f"{root}.zarr", container_name=container_name, chunks=chunks,
                             rechunk_after=True)
        ds = _nav_to_data_vars(ds)

        print('Computing MVBS for pulse:', pulse, 'with tag:', tag, f'and root: {root}.zarr')
        print(ds.data_vars)
        print(ds.dims)

        ds_mvbs = compute_and_save_mvbs(
            ds,
            cruise_id=cruise_id,
            zarr_path=f"{root}--mvbs.zarr",
            compute_mvbs_opts=compute_mvbs_options,
            container_name=container_name,
        )
        results[pulse] = ds_mvbs

        if plot_echograms:
            plot_and_upload_echograms(
                ds_mvbs,
                file_base_name=f"{tag}--mvbs",
                save_to_blobstorage=True,
                upload_path=f"{batch_key}",
                cmap=colormap,
                container_name=container_name,
            )

        if save_to_netcdf:
            save_dataset_to_netcdf(
                ds_mvbs,
                container_name=container_name,
                ds_path=f"{root}--mvbs.nc",
                base_local_temp_path=NETCDF_ROOT_DIR,
                is_temp_dir=False,
            )

    for pulse, tag in tag_for.items():
        if batch_results.get(pulse):
            _run(pulse, tag)

    return results


@task(
    retries=3,
    retry_delay_seconds=60,
    retry_jitter_factor=0.1,
    cache_policy=input_cache_policy,
    refresh_cache=True,
    result_storage=None,
    timeout_seconds=DEFAULT_TASK_TIMEOUT,
    log_prints=True,
    task_run_name="process_file--{file_index}/{total}-{file_name}",
)
def process_single_file(file, file_name, source_container_name, cruise_id,
                        chunks, export_container_name, file_index, total, **kwargs):
    """
    Process a single file for Dask Futures: open it, merge location data, save the dataset, and return its path and
    frequency category.
    """
    zarr_path = f'{file_name}/{file_name}.zarr'
    location_data = file['location_data']
    file_freqs = file['file_freqs']
    file_start_time = file['file_start_time']
    file_end_time = file['file_end_time']
    save_to_netcdf = kwargs.get('save_to_netcdf', False)
    plot_echograms = kwargs.get('plot_echograms', False)
    colormap = kwargs.get('colormap', 'ocean_r')
    file_id = file['id']
    category = "short_pulse" if file_freqs == "38000.0,200000.0" else "long_pulse" if file_freqs == "38000.0" else cruise_id

    try:
        print(f"1) Started processing file {zarr_path} with frequencies {file_freqs}")

        # Open the Zarr store lazily with Dask
        ds = open_zarr_store(zarr_path, cruise_id=cruise_id, container_name=source_container_name, chunks=chunks,
                             rechunk_after=True)

        # Merge location data
        ds = merge_location_data(ds, location_data)

        print('2) Merged location data')
        nc_file_output_path = None
        nc_file_size = 0
        file_path = f"{cruise_id}/{file_name}/{file_name}.zarr"
        zarr_path = save_zarr_store(ds, container_name=export_container_name, zarr_path=file_path, chunks=chunks)
        print('3) Saved to Zarr store:', file_path)
        echogram_files = None

        if plot_echograms:
            echogram_files = plot_and_upload_echograms(
                ds,
                file_base_name=file_name,
                save_to_blobstorage=True,
                upload_path=f"{cruise_id}/{file_name}",
                cmap=colormap,
                cruise_id=cruise_id,
                plot_var='Sv',
                container_name=export_container_name,
            )

        if save_to_netcdf:
            nc_file_path = f"{cruise_id}/{file_name}/{file_name}.nc"
            nc_file_output_path, nc_file_size = save_dataset_to_netcdf(ds, container_name=export_container_name,
                                                         ds_path=nc_file_path, base_local_temp_path=NETCDF_ROOT_DIR,
                                                         is_temp_dir=False)

            print('4) Saved to NetCDF:', nc_file_path)

        return category, zarr_path, nc_file_output_path, nc_file_size, echogram_files
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


@task
def trigger_concatenate_flow(
    files_list=None,
    days_to_combine=1,
    export_id=0,
    cruise_id='',
    container_name='',
    plot_echograms=False,
    save_to_netcdf=False,
    save_nasc_to_netcdf=True,
    save_mvbs_to_netcdf=True,
    colormap='ocean_r',
    compute_nasc_options=None,
    compute_mvbs_options=None,
    mask_impulse_noise=None,
    mask_attenuated_signal=None,
    mask_transient_noise=None,
    remove_background_noise=None,
    apply_seabed_mask=None,
    chunks=None,
    plot_channels_masked=None
):
    files = files_list.copy() if files_list else []
    for source_path, file_record in files:
        file_record.pop('location_data', None)

    state = run_deployment(
        name="concatenate-processed-files/concatenate_processed_files",
        parameters={
            "files_list": files,
            "days_to_combine": days_to_combine,
            'export_id': export_id,
            'cruise_id': cruise_id,
            'container_name': container_name,
            'plot_echograms': plot_echograms,
            'save_to_netcdf': save_to_netcdf,
            'save_nasc_to_netcdf': save_nasc_to_netcdf,
            'save_mvbs_to_netcdf': save_mvbs_to_netcdf,
            'colormap': colormap,
            'compute_nasc_options': compute_nasc_options,
            'compute_mvbs_options': compute_mvbs_options,
            'mask_impulse_noise': mask_impulse_noise,
            'mask_attenuated_signal': mask_attenuated_signal,
            'mask_transient_noise': mask_transient_noise,
            'remove_background_noise': remove_background_noise,
            'apply_seabed_mask': apply_seabed_mask,
            'chunks': chunks,
            'plot_channels_masked': plot_channels_masked
        },
        timeout=None
    )

    return state


@task(
    log_prints=True,
    task_run_name="concatenate_batches--{cruise_id}"
)
def concatenate_batches(cruise_id, **kwargs):
    print("Concatenating batches with files:", cruise_id)


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
                          days_to_combine: int = 1,
                          compute_nasc_options: Optional[NASC_Compute_Options] = None,
                          compute_mvbs_options: Optional[MVBS_Compute_Options] = None,
                          mask_impulse_noise: Optional[MaskImpulseNoise] = None,
                          mask_attenuated_signal: Optional[MaskAttenuatedSignal] = None,
                          mask_transient_noise: Optional[TransientNoiseMask] = None,
                          remove_background_noise: Optional[RemoveBackgroundNoise] = None,
                          apply_seabed_mask: bool = False,
                          chunks_ping_time: Optional[int] = CHUNKS['ping_time'],
                          chunks_depth: Optional[int] = CHUNKS['depth'],
                          save_to_netcdf: bool = False,
                          save_nasc_to_netcdf: bool = True,
                          save_mvbs_to_netcdf: bool = True,
                          base_url: str = '',
                          batch_size: int = BATCH_SIZE,
                          plot_channels_masked=None
                          ):
    denoise_params = {
        'impulse_noise': mask_impulse_noise,
        'attenuated_signal': mask_attenuated_signal,
        'transient_noise': mask_transient_noise,
        'background_noise': remove_background_noise,
    }

    agg_params = {
        'nasc': compute_nasc_options,
        'mvbs': compute_mvbs_options
    }

    if mask_impulse_noise not in (None, False):
        mask_impulse_noise = fill_missing_frequency_params(mask_impulse_noise)

    if remove_background_noise not in (None, False):
        remove_background_noise = fill_missing_frequency_params(remove_background_noise)

    if mask_attenuated_signal not in (None, False):
        mask_attenuated_signal = fill_missing_frequency_params(mask_attenuated_signal)

    if mask_transient_noise not in (None, False):
        mask_transient_noise = fill_missing_frequency_params(mask_transient_noise)

    files_list = get_files_list(
        source_container=source_container,
        cruise_id=cruise_id,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

    total_files = len(files_list)
    logging.info(f"Total files to process: {total_files}")
    print(f"Total files to process: {total_files}")

    chunks = {
        'ping_time': chunks_ping_time,
        'depth': chunks_depth
    }

    export_container_name = output_container if output_container != '' else generate_container_name(cruise_id)
    ensure_container_exists(export_container_name, public_access='container')

    if base_url and not base_url.endswith('/'):
        base_url += '/'

    in_flight = []
    with PostgresDB() as db_connection:
        export_service = ExportService(db_connection)

        export_id, export_key = export_service.create_export(
            container_name=export_container_name,
            cruise_id=cruise_id,
            start_date=start_datetime,
            end_date=end_datetime,
            num_files=total_files,
            denoise_params=denoise_params,
            agg_params=agg_params,
            base_url=f"{base_url}{export_container_name}/",
        )

        print(f"Export created with ID: {export_id}, Key: {export_key}")

    processed_files = defaultdict(list)

    for idx, (source_path, file) in enumerate(files_list):
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
                                            compute_nasc_options=compute_nasc_options,
                                            compute_mvbs_options=compute_mvbs_options,
                                            save_to_netcdf=save_to_netcdf
                                            )
        in_flight.append(future)

        processed_files[file['id']].append(future)

        # Throttle when max concurrent tasks reached
        if len(in_flight) >= batch_size:
            finished = next(as_completed(in_flight))
            in_flight.remove(finished)

    # Wait for remaining tasks
    for remaining in in_flight:
        remaining.result()

    with PostgresDB() as db_connection:
        export_service = ExportService(db_connection)

        for idx, (source_path, file) in enumerate(files_list):
            file_id = file['id']
            category, zarr_path, nc_file, nc_file_size, echogram_files = processed_files[file_id][0].result()

            zarr_path = str(zarr_path) if zarr_path else None
            nc_file = str(nc_file) if nc_file else None

            print('Processed data for file ID:', file_id, 'with futures:')
            export_service.add_file(export_id, file_id, echogram_files, zarr_path, None, nc_file, nc_file_size)

    future_zip = trigger_concatenate_flow.submit(
        files_list=files_list,
        days_to_combine=days_to_combine,
        export_id=export_id,
        cruise_id=cruise_id,
        container_name=export_container_name,
        plot_echograms=plot_echograms,
        save_to_netcdf=save_to_netcdf,
        save_nasc_to_netcdf=save_nasc_to_netcdf,
        save_mvbs_to_netcdf=save_mvbs_to_netcdf,
        colormap=colormap,
        compute_nasc_options=compute_nasc_options,
        compute_mvbs_options=compute_mvbs_options,
        mask_impulse_noise=mask_impulse_noise,
        mask_attenuated_signal=mask_attenuated_signal,
        mask_transient_noise=mask_transient_noise,
        remove_background_noise=remove_background_noise,
        apply_seabed_mask=apply_seabed_mask,
        chunks=chunks,
        plot_channels_masked=plot_channels_masked
    )
    future_zip.wait()


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
                'days_to_combine': 1,
                'compute_nasc_options': None,
                'compute_mvbs_options': None,
                'mask_impulse_noise': None,
                'mask_attenuated_signal': None,
                'mask_transient_noise': None,
                'remove_background_noise': None,
                'apply_seabed_mask': False,
                'chunks_ping_time': 1000,
                'chunks_depth': 1000,
                'save_to_netcdf': False,
                'save_nasc_to_netcdf': True,
                'save_mvbs_to_netcdf': True,
                'base_url': '',
                'batch_size': 4,
                'plot_channels_masked': []
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception occurred: {str(e)}', exc_info=True)
        sys.exit(1)
