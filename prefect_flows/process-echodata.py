import logging
import os
import shutil
import sys
import traceback
from datetime import datetime
from dask.distributed import get_worker

from pathlib import Path
from typing import List, Optional, Union
from dotenv import load_dotenv
from dask.distributed import Client
from prefect import flow, task
from prefect.futures import as_completed
from prefect_dask import DaskTaskRunner, get_dask_client
from prefect.cache_policies import Inputs
from prefect.states import Completed
from prefect.artifacts import create_markdown_artifact
from prefect.deployments import run_deployment

from prefect_flows.pydantic_models import ReprocessingOptions, MaskImpulseNoise, \
    MaskAttenuatedSignal, TransientNoiseMask, RemoveBackgroundNoise
from saildrone.process import process_converted_file, plot_and_upload_echograms
from saildrone.utils import load_local_files, get_metadata_for_files
from saildrone.store import (FileSegmentService, PostgresDB, SurveyService, open_zarr_store,
                             save_dataset_to_netcdf, ensure_container_exists, save_zarr_store, list_zarr_files)

input_cache_policy = Inputs()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

load_dotenv()

RAW_DATA_MOUNT = os.getenv('RAW_DATA_MOUNT')
RAW_DATA_LOCAL = os.getenv('RAW_DATA_LOCAL')
ECHODATA_OUTPUT_PATH = os.getenv('ECHODATA_OUTPUT_PATH')
DASK_CLUSTER_ADDRESS = os.getenv('DASK_CLUSTER_ADDRESS')
CONVERTED_CONTAINER_NAME = os.getenv('CONVERTED_CONTAINER_NAME')
PROCESSED_CONTAINER_NAME = os.getenv('PROCESSED_CONTAINER_NAME')
WEBAPP_CONTAINER_NAME = os.getenv('WEBAPP_CONTAINER_NAME')
GPSDATA_CONTAINER_NAME = os.getenv('GPSDATA_CONTAINER_NAME')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 6))

NETCDF_ROOT_DIR = '/mnt/saildronedata'
DEFAULT_TASK_TIMEOUT = 7_200  # 2 hours
MAX_RUNTIME_SECONDS = 3_300


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
    timeout_seconds=DEFAULT_TASK_TIMEOUT,
    task_run_name="save_to_netcdf--{file_name}"
)
def task_save_to_netcdf(payload, file_name, container_name, chunks=None):
    if payload is None:
        return []

    try:
        cruise_id = payload['cruise_id']
        denoising_applied = payload.get('denoised', False)
        file_path = f"{cruise_id}/{file_name}/{file_name}"
        nc_file_path = f"{file_path}.nc"
        zarr_path = f"{file_path}.zarr"
        saved_paths = []

        zarr_path_denoised = f"{file_path}_denoised.zarr" if denoising_applied else None
        nc_file_path_denoised = f"{file_path}--denoised.nc" if denoising_applied else None

        ds = open_zarr_store(zarr_path, container_name=container_name, chunks=chunks, rechunk_after=True)
        nc_file_output_path = save_dataset_to_netcdf(ds, container_name=container_name, ds_path=nc_file_path,
                                                     base_local_temp_path=NETCDF_ROOT_DIR, is_temp_dir=False)
        saved_paths.append(nc_file_output_path)

        if zarr_path_denoised:
            ds_denoised = open_zarr_store(zarr_path_denoised, container_name=container_name, chunks=chunks,
                                          rechunk_after=True)
            nc_file_denoised_output_path = save_dataset_to_netcdf(ds_denoised, container_name=container_name,
                                                                  ds_path=nc_file_path_denoised,
                                                                  base_local_temp_path=NETCDF_ROOT_DIR,
                                                                  is_temp_dir=False)
            saved_paths.append(nc_file_denoised_output_path)

        return saved_paths
    except Exception as e:
        traceback.print_exc()

        markdown_report = f"""# Error during save_to_netcdf
        Error occurred while saving to NetCDF: {file_name}

        {str(e)}

        ## Error details
        - **Error Message**: {str(e)}
        - **Traceback**: {traceback.format_exc()}
        """

        create_markdown_artifact(markdown_report)

        return []


@task(
    log_prints=True,
    retries=1,
    timeout_seconds=DEFAULT_TASK_TIMEOUT,
    task_run_name="plot_echograms--{file_name}"
)
def task_plot_echograms_normal(payload, file_name, container_name, echograms_container, chunks=None, cmap='ocean_r'):
    if payload is None:
        return None

    if not isinstance(payload, dict) or 'file_name' not in payload:
        raise ValueError("Invalid payload passed to task_plot_echograms_normal")

    file_name = payload['file_name']

    try:
        cruise_id = payload['cruise_id']
        file_path = f"{cruise_id}/{file_name}/{file_name}"
        upload_path = f"{cruise_id}/{file_name}"
        zarr_path = f"{file_path}.zarr"
        ds = open_zarr_store(zarr_path, container_name=container_name, chunks=chunks, rechunk_after=True)

        plot_and_upload_echograms(ds,
                                  file_base_name=file_name,
                                  save_to_blobstorage=True,
                                  depth_var="depth",
                                  upload_path=upload_path,
                                  cmap=cmap,
                                  container_name=echograms_container)

        return Completed(message="plot_echograms completed successfully")
    except Exception as e:
        traceback.print_exc()

        markdown_report = f"""# Error during plot_echograms
        Error occurred while plotting echograms: {file_name}

        {str(e)}

        ## Error details
        - **Error Message**: {str(e)}
        - **Traceback**: {traceback.format_exc()}
        """

        create_markdown_artifact(markdown_report)

        return Completed(message="plot_echograms completed with errors")


@task(
    log_prints=True,
    retries=1,
    timeout_seconds=DEFAULT_TASK_TIMEOUT,
    task_run_name="plot_echograms_denoised--{file_name}"
)
def task_plot_echograms_denoised(payload, file_name, container_name, echograms_container, chunks=None, cmap='ocean_r'):
    if payload is None:
        return None

    if not isinstance(payload, dict) or 'file_name' not in payload:
        raise ValueError("Invalid payload passed to task_plot_echograms_normal")

    file_name = payload['file_name']

    try:
        denoising_applied = payload.get('denoised', False)
        cruise_id = payload['cruise_id']
        file_path = f"{cruise_id}/{file_name}/{file_name}"
        upload_path = f"{cruise_id}/{file_name}"
        zarr_path_denoised = f"{file_path}_denoised.zarr" if denoising_applied else None

        if zarr_path_denoised:
            ds_denoised = open_zarr_store(zarr_path_denoised, container_name=container_name, chunks=chunks,
                                          rechunk_after=True)

            plot_and_upload_echograms(ds_denoised,
                                      file_base_name=f"{file_name}--denoised",
                                      save_to_blobstorage=True,
                                      depth_var="depth",
                                      upload_path=upload_path,
                                      cmap=cmap,
                                      container_name=echograms_container)

        return Completed(message="plot_echograms_denoised completed successfully")
    except Exception as e:
        traceback.print_exc()

        markdown_report = f"""# Error during plot_echograms_denoised
        Error occurred while plotting echograms: {file_name}

        {str(e)}

        ## Error details
        - **Error Message**: {str(e)}
        - **Traceback**: {traceback.format_exc()}
        """

        create_markdown_artifact(markdown_report)

        return Completed(message="plot_echograms_denoised completed with errors")


@task(
    log_prints=True,
    retries=1,
    timeout_seconds=DEFAULT_TASK_TIMEOUT,
    task_run_name="plot_echograms_seabed--{file_name}"
)
def task_plot_echograms_seabed(payload, file_name, container_name, echograms_container, chunks=None, cmap='ocean_r'):
    if payload is None:
        return None

    file_name = payload['file_name']

    try:
        seabed_mask = payload['seabed_mask']
        cruise_id = payload['cruise_id']

        file_path = f"{cruise_id}/{file_name}/{file_name}"
        upload_path = f"{cruise_id}/{file_name}"
        zarr_path_seabed = f"{file_path}_seabed.zarr" if seabed_mask else None

        if zarr_path_seabed:
            ds = open_zarr_store(zarr_path_seabed, container_name=container_name, chunks=chunks, rechunk_after=True)

            plot_and_upload_echograms(ds,
                                      file_base_name=f"{file_name}--seabed",
                                      save_to_blobstorage=True,
                                      depth_var="depth",
                                      upload_path=upload_path,
                                      cmap=cmap,
                                      container_name=echograms_container)

        return Completed(message="plot_echograms_denoised completed successfully")
    except Exception as e:
        traceback.print_exc()

        markdown_report = f"""# Error during plot_echograms_seabed
        Error occurred while plotting echograms: {file_name}

        {str(e)}

        ## Error details
        - **Error Message**: {str(e)}
        - **Traceback**: {traceback.format_exc()}
        """

        create_markdown_artifact(markdown_report)

        return Completed(message="plot_echograms_seabed completed with errors")


@task(
    retries=3,
    retry_delay_seconds=60,
    cache_policy=input_cache_policy,
    retry_jitter_factor=0.1,
    refresh_cache=True,
    result_storage=None,
    timeout_seconds=DEFAULT_TASK_TIMEOUT,
    log_prints=True,
    task_run_name="process-{file_name}",
)
def process_single_file(source_path: Path, file_name, denoised, location_data, **kwargs):
    try:
        worker = get_worker()
        print(f"Running on Dask worker: {worker.address}")

        cruise_id = kwargs.get('cruise_id')
        load_from_blobstorage = kwargs.get('load_from_blobstorage')
        source_container = kwargs.get('source_container')
        save_to_blobstorage = kwargs.get('save_to_blobstorage')
        output_container = kwargs.get('output_container')
        save_to_directory = kwargs.get('save_to_directory')
        output_directory = kwargs.get('output_directory')
        chunks_echodata = kwargs.get('chunks_echodata', None)
        apply_seabed_mask = kwargs.get('apply_seabed_mask', False)
        skip_processed = kwargs.get('skip_processed', False)

        if skip_processed:
            payload = {
                'file_name': source_path.stem,
                'cruise_id': cruise_id,
                'denoised': denoised,
                'seabed_mask': apply_seabed_mask
            }

            return payload

        output_path = output_directory
        converted_container_name = None
        processed_container_name = None

        if save_to_directory is not True:
            output_path = None

        if load_from_blobstorage is True:
            converted_container_name = source_container

        if save_to_blobstorage is True:
            processed_container_name = output_container

        return process_converted_file(source_path,
                                      output_path=output_path,
                                      chunks=chunks_echodata,
                                      file_name=file_name,
                                      location_data=location_data,
                                      converted_container_name=converted_container_name,
                                      processed_container_name=processed_container_name,
                                      gps_container_name=GPSDATA_CONTAINER_NAME,
                                      **kwargs)
    except Exception as e:
        print(f"Error processing file: {source_path.name}: ${str(e)}")

        markdown_report = f"""# Error report for {source_path.name}
        Error occurred while processing the file: {source_path}
        {str(e)}
        
        ## Error details
        - **Error Message**: {str(e)}
        - **Traceback**: {traceback.format_exc()}
        """
        create_markdown_artifact(markdown_report)

        return Completed(message="Task completed with errors")


def _process_files_list(files_list_with_data, save_to_netcdf, denoised=None, **kwargs):
    in_flight = []
    side_running_tasks = []
    netcdf_outputs = []

    plot_echograms = kwargs.get('plot_echograms', False)
    chunks_sv_data = kwargs.get('chunks_sv_data', None)
    output_container = kwargs.get('output_container', None)
    echograms_container = kwargs.get('echograms_container', None)
    colormap = kwargs.get('colormap', 'ocean_r')
    apply_seabed_mask = kwargs.get('apply_seabed_mask', False)
    batch_size = kwargs.get('batch_size', BATCH_SIZE)

    for source_path, file_record in files_list_with_data:
        location_data = file_record["location_data"] if 'location_data' in file_record else None
        file_name = file_record["file_name"]
        future = process_single_file.submit(source_path,
                                            file_name=file_name,
                                            denoised=denoised,
                                            location_data=location_data,
                                            **kwargs)

        if save_to_netcdf:
            future_nc_task = task_save_to_netcdf.submit(future, file_name, output_container, chunks_sv_data)
            side_running_tasks.append(future_nc_task)
            netcdf_outputs.append(future_nc_task)

        if plot_echograms:
            future_plot_task = task_plot_echograms_normal.submit(future,
                                                                 file_name,
                                                                 output_container,
                                                                 echograms_container,
                                                                 chunks_sv_data,
                                                                 colormap)
            future_plot_task_denoised = task_plot_echograms_denoised.submit(future,
                                                                            file_name,
                                                                            output_container,
                                                                            echograms_container,
                                                                            chunks_sv_data,
                                                                            colormap)

            side_running_tasks.append(future_plot_task)
            side_running_tasks.append(future_plot_task_denoised)

            if apply_seabed_mask:
                future_plot_task = task_plot_echograms_seabed.submit(future,
                                                                     file_name,
                                                                     output_container,
                                                                     echograms_container,
                                                                     chunks_sv_data,
                                                                     colormap)
                side_running_tasks.append(future_plot_task)

        in_flight.append(future)

        # Throttle when max concurrent tasks reached
        if len(in_flight) >= batch_size:
            finished = next(as_completed(in_flight))
            in_flight.remove(finished)

    # Wait for remaining tasks
    for future_task in in_flight + side_running_tasks:
        future_task.result()

    if save_to_netcdf:
        future_zip = trigger_netcdf_flow.submit(
            file_list=netcdf_outputs,
            container=output_container
        )
        future_zip.wait()

    if os.path.exists('/tmp/oceanstream/netcdfdata'):
        shutil.rmtree('/tmp/oceanstream/netcdfdata', ignore_errors=True)

    logging.info("All batches have been processed.")


@flow(task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS))
def load_and_process_files_to_zarr(source_directory: str,
                                   cruise_id: str,
                                   load_from_blobstorage: bool,
                                   source_container: str,
                                   get_list_from_db: bool,
                                   start_datetime: Optional[datetime],
                                   end_datetime: Optional[datetime],
                                   save_to_blobstorage: bool,
                                   output_container: str,
                                   save_to_directory: bool,
                                   output_directory: str,
                                   reprocess_options: Optional[ReprocessingOptions],
                                   plot_echograms: bool,
                                   compute_nasc: bool,
                                   compute_mvbs: bool,
                                   echograms_container: str,
                                   encode_mode: str,
                                   colormap: str,
                                   waveform_mode: str,
                                   depth_offset: float,
                                   chunks_ping_time: int,
                                   chunks_range_sample: Optional[int],
                                   mask_impulse_noise: Optional[MaskImpulseNoise] = None,
                                   mask_attenuated_signal: Optional[MaskAttenuatedSignal] = None,
                                   mask_transient_noise: Optional[TransientNoiseMask] = None,
                                   remove_background_noise: Optional[RemoveBackgroundNoise] = None,
                                   apply_seabed_mask: bool = False,
                                   save_to_netcdf: bool = False,
                                   batch_size: int = BATCH_SIZE):
    reprocess = reprocess_options.get('reprocess', False) if reprocess_options else False
    skip_processed = reprocess_options.get('skip_processed', False) if reprocess_options else False
    denoised = (mask_impulse_noise is not None or
                mask_transient_noise is not None or
                mask_attenuated_signal is not None or
                remove_background_noise is not None)

    files_list, files_data = _prepare_file_list(
        source_directory,
        cruise_id,
        load_from_blobstorage,
        source_container,
        get_list_from_db=get_list_from_db,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        reprocess=reprocess
    )

    print('source_directory:', source_directory)
    total_files = len(files_list)
    print(f"Total files to process: {total_files}")
    files_list_with_data = get_metadata_for_files(files_list, files_data)
    chunks_echodata = {
        'ping_time': chunks_ping_time,
        'range_sample': chunks_range_sample
    }

    chunks_sv_data = {
        'ping_time': chunks_ping_time,
        'depth': 1000
    }

    _process_files_list(files_list_with_data,
                        save_to_netcdf=save_to_netcdf,
                        denoised=denoised,
                        cruise_id=cruise_id,
                        plot_echograms=plot_echograms,
                        load_from_blobstorage=load_from_blobstorage,
                        source_container=source_container,
                        output_container=output_container,
                        reprocess=reprocess,
                        chunks_echodata=chunks_echodata,
                        chunks_sv_data=chunks_sv_data,
                        compute_nasc=compute_nasc,
                        compute_mvbs=compute_mvbs,
                        save_to_blobstorage=save_to_blobstorage,
                        save_to_directory=save_to_directory,
                        output_directory=output_directory,
                        encode_mode=encode_mode,
                        colormap=colormap,
                        skip_processed=skip_processed,
                        waveform_mode=waveform_mode,
                        depth_offset=depth_offset,
                        mask_impulse_noise=mask_impulse_noise,
                        mask_attenuated_signal=mask_attenuated_signal,
                        mask_transient_noise=mask_transient_noise,
                        remove_background_noise=remove_background_noise,
                        apply_seabed_mask=apply_seabed_mask,
                        echograms_container=echograms_container,
                        batch_size=batch_size
                        )


def _prepare_file_list(
    source_directory: str,
    cruise_id: str,
    load_from_blobstorage: bool,
    source_container: str,
    get_list_from_db: bool,
    start_datetime: Optional[datetime],
    end_datetime: Optional[datetime],
    reprocess: bool
):
    """Fetch the list of files to be processed."""
    files = None

    with PostgresDB() as db_connection:
        survey_service = SurveyService(db_connection)
        survey_id = survey_service.get_survey_by_cruise_id(cruise_id)
        if not survey_id:
            survey_id = survey_service.insert_survey(cruise_id)
            logging.info(f"Inserted new survey with cruise_id: {cruise_id}")

        if get_list_from_db:
            file_service = FileSegmentService(db_connection)
            condition = "" if reprocess else "AND processed IS NOT True"
            if start_datetime and end_datetime:
                condition += (
                    f" AND file_start_time > '{start_datetime}' "
                    f"AND file_end_time < '{end_datetime}'"
                )
            files = file_service.get_files_by_survey_id(survey_id, condition)
            if not files:
                logging.warning(f"No files found for survey_id: {survey_id} with condition: {condition}")
                return [], []

    if load_from_blobstorage:
        file_names = [f['file_name'] for f in files] if files else None
        zarr_paths = list_zarr_files(
            source_container,
            cruise_id=cruise_id,
            file_names=file_names,
        )
    elif files:
        # If files are provided and not loading from blob storage, construct paths
        zarr_paths = [Path(source_directory) / f"{file['file_name']}.zarr" for file in files]
    else:
        # If no files are provided, load local files from the source directory
        zarr_paths = load_local_files(source_directory, source_directory, '*.zarr')

    return zarr_paths, files


if __name__ == "__main__":
    with PostgresDB() as db:
        db.create_tables()

    client = Client(address=DASK_CLUSTER_ADDRESS)
    ensure_container_exists(PROCESSED_CONTAINER_NAME)

    try:
        # Start the flow
        load_and_process_files_to_zarr.serve(
            name='process-echodata-to-sv',
            parameters={
                'source_directory': RAW_DATA_LOCAL,
                'cruise_id': '',
                'load_from_blobstorage': False,
                'source_container': CONVERTED_CONTAINER_NAME,
                'get_list_from_db': False,
                'start_datetime': None,
                'end_datetime': None,
                'save_to_blobstorage': True,
                'output_container': PROCESSED_CONTAINER_NAME,
                'save_to_directory': False,
                'output_directory': '',
                'reprocess_options': None,
                'plot_echograms': False,
                'compute_nasc': True,
                'compute_mvbs': True,
                'echograms_container': WEBAPP_CONTAINER_NAME,
                'encode_mode': 'complex',
                'colormap': 'ocean_r',
                'waveform_mode': 'CW',
                'depth_offset': 0,
                'chunks_ping_time': 1000,
                'chunks_range_sample': 1000,
                'mask_transient_noise': None,
                'mask_impulse_noise': None,
                'mask_attenuated_signal': None,
                'remove_background_noise': None,
                'apply_seabed_mask': False,
                'save_to_netcdf': False,
                'batch_size': BATCH_SIZE
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception occurred: {str(e)}', exc_info=True)
        sys.exit(1)
