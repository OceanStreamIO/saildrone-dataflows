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
    MaskAttenuatedSignal, TransientNoiseMask, RemoveBackgroundNoise

from saildrone.process import apply_denoising, plot_and_upload_echograms, get_files_list
from saildrone.process.workflow import compute_and_save_nasc, compute_and_save_mvbs
from saildrone.process.concat import merge_location_data, concatenate_and_rechunk
from saildrone.store import (FileSegmentService, PostgresDB, SurveyService, open_zarr_store, generate_container_name,
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


@task(log_prints=True)
def zip_netcdf_outputs(nc_file_paths, zip_name, container_name):
    flat_paths = [p for group in nc_file_paths for p in group if p]  # flatten and skip empty

    zip_and_save_netcdf_files(flat_paths, zip_name, container_name, tmp_dir=NETCDF_ROOT_DIR + '/tmp')
    logging.info(f"Uploaded archive {zip_name} to container {container_name}")


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
                depth_var="depth",
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
                depth_var="depth",
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
    log_prints=True,
    task_run_name="concatenate_batch_files--{batch_key}"
)
def concatenate_batch_files(batch_key, cruise_id, files, container_name, plot_echograms, save_to_netcdf,
                            colormap, **kwargs):
    """Run NASC, MVBS, … for one calendar batch."""
    # 1) bucket files by category
    batch_results = {cat: [] for cat in CATEGORY_CONFIG}
    chunks = kwargs.get('chunks', None)

    for file_info in files:
        freqs = file_info["file_freqs"]
        for category, cfg in CATEGORY_CONFIG.items():
            if cfg["freq_key"] is None or freqs == cfg["freq_key"]:
                path = f"{cruise_id}/{file_info['file_name']}/{file_info['file_name']}.zarr"
                batch_results[category].append(path)
                break

    def _process_category(cat: str):
        paths = batch_results[cat]
        if not paths:
            return

        section = CATEGORY_CONFIG[cat]
        print('Concatenating files for category:', cat, 'with paths:', paths)
        ds = concatenate_and_rechunk(paths, container_name=container_name, chunks=chunks)
        print(f"Finished concatenating {cat} dataset:", ds.data_vars)

        # save Zarr
        zarr_path = f"{batch_key}/{section['zarr_name']}".format(batch_key=batch_key, denoised='')
        save_zarr_store(ds, container_name=container_name, zarr_path=zarr_path)

        # optional echograms
        if plot_echograms:
            plot_and_upload_echograms(
                ds,
                file_base_name=section["file_base"].format(batch_key=batch_key, denoised=''),
                save_to_blobstorage=True,
                depth_var="depth",
                upload_path=batch_key,
                cmap=colormap,
                container_name=container_name,
            )

        ##########################################################
        print('5) Applying denoising')
        try:
            sv_dataset_denoised = apply_denoising(ds, chunks_denoising=chunks, **kwargs)
        except Exception as e:
            print(f"Error applying denoising to {zarr_path}: {str(e)}")
            traceback.print_exc()
            sv_dataset_denoised = None

        print('5) Denoising applied', sv_dataset_denoised)

        if sv_dataset_denoised is not None:
            zarr_path_denoised = f"{batch_key}/{section['zarr_name']}".format(batch_key=batch_key, denoised='--denoised')
            save_zarr_store(sv_dataset_denoised, container_name=container_name, zarr_path=zarr_path_denoised)
            print('6) Saved denoised dataset to Zarr store:', zarr_path)

            if plot_echograms:
                plot_and_upload_echograms(
                    ds,
                    file_base_name=section["file_base"].format(batch_key=batch_key, denoised='--denoised'),
                    save_to_blobstorage=True,
                    depth_var="depth",
                    upload_path=batch_key,
                    cmap=colormap,
                    container_name=container_name,
                )

            if save_to_netcdf:
                # FIXME: move the NetCDF convertion to a new flow
                nc_file_path_denoised = zarr_path
                # save_dataset_to_netcdf(
                #     sv_dataset_denoised,
                #     container_name=export_container_name,
                #     ds_path=nc_file_path_denoised,
                #     base_local_temp_path=NETCDF_ROOT_DIR,
                #     is_temp_dir=False,
                # )
                #
                # print('7) Saved denoised dataset to NetCDF:', nc_file_path_denoised)
        ##########################################################

        # optional NetCDF
        if save_to_netcdf:
            nc_path = f"{batch_key}/{section['nc_name']}".format(batch_key=batch_key, denoised='')
            # save_dataset_to_netcdf(
            #     ds,
            #     container_name=container_name,
            #     ds_path=nc_path,
            #     base_local_temp_path=NETCDF_ROOT_DIR,
            #     is_temp_dir=False,
            # )

    # 2) run through each category
    for category in CATEGORY_CONFIG:
        _process_category(category)

    logging.info(
        f"Running batch aggregation for key: {batch_key}, with {len(files)} files."
    )
    return batch_results


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
        nc_file_output_path_denoised = None
        file_path = f"{cruise_id}/{file_name}/{file_name}.zarr"
        zarr_path = save_zarr_store(ds, container_name=export_container_name, zarr_path=file_path, chunks=chunks)
        print('3) Saved to Zarr store:', file_path)

        if plot_echograms:
            plot_and_upload_echograms(
                ds,
                file_base_name=file_name,
                save_to_blobstorage=True,
                depth_var="depth",
                upload_path=f"{cruise_id}/{file_name}",
                cmap=colormap,
                plot_var='Sv',
                container_name=export_container_name,
            )

        if save_to_netcdf:
            nc_file_path = f"{cruise_id}/{file_name}/{file_name}.nc"
            nc_file_output_path = save_dataset_to_netcdf(ds, container_name=export_container_name,
                                                         ds_path=nc_file_path, base_local_temp_path=NETCDF_ROOT_DIR,
                                                         is_temp_dir=False)

            print('4) Saved to NetCDF:', nc_file_path)

        return category, nc_file_output_path, nc_file_output_path_denoised
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
def trigger_netcdf_flow(container, file_list):
    flat_paths = [p for group in file_list for p in group if p]  # flatten and skip empty

    print('Triggering NetCDF flow with container:', flat_paths)

    state = run_deployment(
        name="generate-netcdf-zip-export/generate-netcdf-zip",
        parameters={
            "output_container": container,
            "file_list": flat_paths
        },
        timeout=0
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
                          batch_size: int = BATCH_SIZE
                          ):
    denoised = (mask_impulse_noise is not None or
                mask_transient_noise is not None or
                mask_attenuated_signal is not None or
                remove_background_noise is not None)

    files_list = get_files_list(
        source_directory=source_container,
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
    if output_container == '':
        ensure_container_exists(export_container_name, public_access='container')

    in_flight = []
    workers = get_worker_addresses(scheduler=DASK_CLUSTER_ADDRESS)
    n_workers = len(workers)
    netcdf_outputs = []

    for idx, (source_path, file) in enumerate(files_list):
        # target_worker = workers[idx % n_workers]
        # with dask.annotate(workers=[target_worker], allow_other_workers=False):
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

        # Throttle when max concurrent tasks reached
        if len(in_flight) >= batch_size:
            finished = next(as_completed(in_flight))
            in_flight.remove(finished)

    # Wait for remaining tasks
    for remaining in in_flight:
        remaining.result()

    future_con = concatenate_batches.submit(cruise_id, denoised=denoised,
                                            container_name=export_container_name,
                                            compute_nasc_options=compute_nasc_options,
                                            plot_echograms=compute_nasc_options,
                                            save_to_netcdf=save_to_netcdf,
                                            colormap=colormap,
                                            chunks=chunks)
    future_con.wait()

    # Aggregate results by batch
    by_batch = defaultdict(list)
    agg_in_flight = []
    agg_side_tasks = []
    files_to_convert = []

    for source_path, file_record in files_list:
        ts = file_record["file_start_time"]
        key = _batch_key(ts, days_to_combine)
        by_batch[key].append(file_record)

    batches = by_batch.items()
    print(f"Total batches to process: {len(batches)}")

    for key, files in batches:
        print('Processing batch:', key, 'with', len(files), 'files.')
        future = concatenate_batch_files.submit(key, cruise_id, files, export_container_name, plot_echograms,
                                                save_to_netcdf,
                                                colormap,
                                                mask_impulse_noise=mask_impulse_noise,
                                                mask_attenuated_signal=mask_attenuated_signal,
                                                mask_transient_noise=mask_transient_noise,
                                                remove_background_noise=remove_background_noise,
                                                apply_seabed_mask=apply_seabed_mask,
                                                chunks=chunks)

        agg_in_flight.append(future)

        if compute_nasc_options:
            future_nasc_task = compute_batch_nasc.submit(future, key, cruise_id, export_container_name,
                                                         compute_nasc_options=compute_nasc_options,
                                                         plot_echograms=plot_echograms,
                                                         save_to_netcdf=save_to_netcdf,
                                                         colormap=colormap,
                                                         chunks=chunks)
            agg_side_tasks.append(future_nasc_task)

        if compute_mvbs_options:
            future_mvbs_task = compute_batch_mvbs.submit(future, key, cruise_id, export_container_name,
                                                         compute_mvbs_options=compute_mvbs_options,
                                                         plot_echograms=plot_echograms,
                                                         save_to_netcdf=save_to_netcdf,
                                                         colormap=colormap,
                                                         chunks=chunks)
            agg_side_tasks.append(future_mvbs_task)

        if len(agg_in_flight) >= batch_size:
            finished = next(as_completed(agg_in_flight))
            agg_in_flight.remove(finished)

    for remaining in agg_in_flight + agg_side_tasks:
        remaining.result()

    # if save_to_netcdf:
    #     future_zip = trigger_netcdf_flow.submit(
    #         file_list=files_to_convert,
    #         container=export_container_name
    #     )
    #     future_zip.wait()

    if os.path.exists('/tmp/oceanstream/netcdfdata'):
        shutil.rmtree('/tmp/oceanstream/netcdfdata', ignore_errors=True)



def _batch_key(ts: datetime, width_days: int) -> str:
    """
    Anchor `ts` to the start of its `width`-day window and return a
    filename-safe key.
      width == 1  →  '2023-08-08'
      width >  1  →  '2023-08-08_to_2023-08-10'   (inclusive range)
    """
    anchor = datetime(ts.year, ts.month, ts.day)  # midnight of that day

    if width_days == 1:
        return f"{anchor:%Y-%m-%d}"

    anchor -= timedelta(days=(anchor - datetime.min).days % width_days)
    end = anchor + timedelta(days=width_days - 1)

    return f"{anchor:%Y-%m-%d}_to_{end:%Y-%m-%d}"


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
                'batch_size': 4
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception occurred: {str(e)}', exc_info=True)
        sys.exit(1)
