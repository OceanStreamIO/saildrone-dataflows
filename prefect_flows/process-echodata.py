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
from pydantic import BaseModel, Field
from dask.distributed import Client
from prefect import flow, task
from prefect.futures import as_completed
from prefect_dask import DaskTaskRunner, get_dask_client
from prefect.cache_policies import Inputs
from prefect.states import Completed
from prefect.artifacts import create_markdown_artifact
from prefect.deployments import run_deployment
import uuid

from saildrone.process import process_converted_file, plot_and_upload_echograms
from saildrone.utils import load_local_files
from saildrone.store import (
    FileSegmentService,
    PostgresDB,
    SurveyService,
    open_zarr_store,
    save_dataset_to_netcdf,
    ensure_container_exists,
    save_zarr_store,
    list_zarr_files,
    ExportService,
    get_container_base_url,
    get_blob_size,
)


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


class DenoiseOptions(BaseModel):
    def get(self, key, default_value=None):
        return getattr(self, key, default_value)


class ReprocessingOptions(DenoiseOptions):
    reprocess: bool = Field(default=False,
                            description='Load and reprocess the raw data for files that '
                                        'have been already marked as processed.')
    skip_processed: bool = Field(default=False,
                                 description='Skip processing and only generate outputs'
                                             ' (e.g., echograms, NetCDF files); requires previously processed files.')


class MaskImpulseNoise(DenoiseOptions):
    depth_bin: int = Field(default=10, description="Downsampling bin size along vertical range variable (`range_var`) in meters.")
    num_side_pings: int = Field(default=2, description="Number of side pings to look at for the two-side comparison.")
    threshold: float = Field(default=10, description="Impulse noise threshold value (in dB) for the two-side comparison.")
    range_var: str = Field(default='depth', description="Vertical Axis Range Variable. Can be either \"depth\" or \"echo_range\".")


class MaskAttenuatedSignal(DenoiseOptions):
    upper_limit_sl: int = Field(default=180, description="Upper limit of deep scattering layer line (m).")
    lower_limit_sl: int = Field(default=300, description="Lower limit of deep scattering layer line (m).")
    num_side_pings: int = Field(default=15, description="Number of preceding & subsequent pings defining the block.")
    threshold: float = Field(default=10, description="Attenuation signal threshold value (dB) for the ping-block comparison.")
    range_var: str = Field(default='depth', description="Vertical Axis Range Variable. Can be either `depth` or `echo_range`.")


class TransientNoiseMask(DenoiseOptions):
    operation: str = Field(default='nanmedian', description="Pooling function used in the pooled Sv aggregation, either 'nanmedian' or 'nanmean'.")
    depth_bin: int = Field(default=10, description="Bin size for depth calculation.")
    num_side_pings: int = Field(default=25, description="Number of side pings to include.")
    exclude_above: float = Field(default=250.0, description="Exclude data above this depth value.")
    threshold: float = Field(default=12.0, description="Transient noise threshold value (in dB) for the pooling comparison.")
    range_var: str = Field(default='depth', description="Vertical Range Variable. Can be either `depth` or `echo_range`.")


class RemoveBackgroundNoise(DenoiseOptions):
    ping_num: int = Field(default=5, description="Number of pings to obtain noise estimates")
    range_sample_num: int = Field(default=30, description="Number of range samples to consider.")
    background_noise_max: float = Field(default=-125, description="Maximum allowable background noise estimation (in dB).")
    SNR_threshold: float = Field(default=3.0, description="Signal-to-noise ratio threshold for background noise removal.")


@task
def trigger_netcdf_flow(container, file_list, zip_name):
    """Trigger the NetCDF combining flow."""
    flat_paths = [p for group in file_list for p in group if p]  # flatten and skip empty

    print('Triggering NetCDF flow with container:', flat_paths)

    state = run_deployment(
        name="generate-netcdf-zip-export/generate-netcdf-zip",
        parameters={
            "output_container": container,
            "file_list": flat_paths,
            "zip_name": zip_name,
        },
        timeout=0,
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
        saved = []

        zarr_path_denoised = f"{file_path}_denoised.zarr" if denoising_applied else None
        nc_file_path_denoised = f"{file_path}--denoised.nc" if denoising_applied else None

        ds = open_zarr_store(zarr_path, container_name=container_name, chunks=chunks, rechunk_after=True)
        nc_file_output_path = save_dataset_to_netcdf(
            ds,
            container_name=container_name,
            ds_path=nc_file_path,
            base_local_temp_path=NETCDF_ROOT_DIR,
            is_temp_dir=False,
        )
        saved.append({"path": nc_file_path, "size": os.path.getsize(nc_file_output_path)})

        if zarr_path_denoised:
            ds_denoised = open_zarr_store(zarr_path_denoised, container_name=container_name, chunks=chunks,
                                          rechunk_after=True)
            nc_file_denoised_output_path = save_dataset_to_netcdf(
                ds_denoised,
                container_name=container_name,
                ds_path=nc_file_path_denoised,
                base_local_temp_path=NETCDF_ROOT_DIR,
                is_temp_dir=False,
            )
            saved.append({"path": nc_file_path_denoised, "size": os.path.getsize(nc_file_denoised_output_path)})

        return saved
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
    task_run_name="process-{source_path.stem}",
)
def process_single_file(source_path: Path, chunks_echodata, **kwargs):
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
        reprocess = kwargs.get('reprocess')
        compute_nasc = kwargs.get('compute_nasc')
        compute_mvbs = kwargs.get('compute_mvbs')
        encode_mode = kwargs.get('encode_mode')
        colormap = kwargs.get('colormap')
        waveform_mode = kwargs.get('waveform_mode')
        depth_offset = kwargs.get('depth_offset')
        mask_transient_noise = kwargs.get('mask_transient_noise')
        mask_impulse_noise = kwargs.get('mask_impulse_noise')
        mask_attenuated_signal = kwargs.get('mask_attenuated_signal')
        remove_background_noise = kwargs.get('remove_background_noise')
        apply_seabed_mask = kwargs.get('apply_seabed_mask', False)
        skip_processed = kwargs.get('skip_processed', False)

        if skip_processed:
            denoised = (mask_impulse_noise is not None or
                        mask_transient_noise is not None or
                        mask_attenuated_signal is not None or
                        remove_background_noise is not None)
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
                                      cruise_id=cruise_id,
                                      output_path=output_path,
                                      chunks=chunks_echodata,
                                      load_from_blobstorage=load_from_blobstorage,
                                      converted_container_name=converted_container_name,
                                      save_to_blobstorage=save_to_blobstorage,
                                      save_to_directory=save_to_directory,
                                      processed_container_name=processed_container_name,
                                      reprocess=reprocess,
                                      depth_offset=depth_offset,
                                      compute_nasc=compute_nasc,
                                      compute_mvbs=compute_mvbs,
                                      gps_container_name=GPSDATA_CONTAINER_NAME,
                                      encode_mode=encode_mode,
                                      waveform_mode=waveform_mode,
                                      mask_transient_noise=mask_transient_noise,
                                      mask_impulse_noise=mask_impulse_noise,
                                      mask_attenuated_signal=mask_attenuated_signal,
                                      remove_background_noise=remove_background_noise,
                                      apply_seabed_mask=apply_seabed_mask
                                      )
    except Exception as e:
        print(f"Error processing file: {source_path.name}: ${str(e)}")

        markdown_report = f"""# Error report for {source_path.name}
        Error occurred while processing the file: {source_path}
        {str(e)}
        """
        create_markdown_artifact(markdown_report)

        return Completed(message="Task completed with errors")


def _prepare_file_list(
    source_directory: str,
    cruise_id: str,
    load_from_blobstorage: bool,
    source_container: str,
    get_list_from_db: bool,
    start_datetime: Optional[datetime],
    end_datetime: Optional[datetime],
    reprocess: bool,
) -> list[Union[str, Path]]:
    """Fetch the list of files to be processed."""
    file_names = None

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
            file_names = file_service.get_files_list_with_condition(survey_id, condition)

    if file_names and not load_from_blobstorage:
        return [Path(source_directory) / f"{fname}.raw" for fname in file_names]
    if file_names and load_from_blobstorage:
        return list_zarr_files(source_container, cruise_id=cruise_id, file_names=file_names)
    if load_from_blobstorage:
        return list_zarr_files(source_container, cruise_id=cruise_id)
    return load_local_files(source_directory, source_directory, '*.zarr')



def _process_files(
    files_list,
    batch_size,
    chunks_echodata,
    chunks_sv_data,
    **kwargs,
) -> tuple[list[dict], dict[str, list], list[tuple[str, any]]]:
    """Schedule processing tasks for a list of files."""
    in_flight = []
    side_tasks = []
    netcdf_outputs: list[tuple[str, any]] = []
    file_results: list[dict] = []
    nc_result_map: dict[str, list] = {}

    for src in files_list:
        file_name = src.stem if isinstance(src, Path) else src
        future = process_single_file.submit(src, chunks_echodata=chunks_echodata, **kwargs)

        if kwargs.get("save_to_netcdf"):
            future_nc_task = task_save_to_netcdf.submit(
                future,
                file_name,
                kwargs["output_container"],
                chunks_sv_data,
            )
            side_tasks.append(future_nc_task)
            netcdf_outputs.append((file_name, future_nc_task))

        if kwargs.get("plot_echograms"):
            future_plot_task = task_plot_echograms_normal.submit(
                future,
                file_name,
                kwargs["output_container"],
                kwargs["echograms_container"],
                chunks_sv_data,
                kwargs.get("colormap", "ocean_r"),
            )
            side_tasks.append(future_plot_task)

            future_plot_denoised = task_plot_echograms_denoised.submit(
                future,
                file_name,
                kwargs["output_container"],
                kwargs["echograms_container"],
                chunks_sv_data,
                kwargs.get("colormap", "ocean_r"),
            )
            side_tasks.append(future_plot_denoised)

            if kwargs.get("apply_seabed_mask"):
                future_plot_seabed = task_plot_echograms_seabed.submit(
                    future,
                    file_name,
                    kwargs["output_container"],
                    kwargs["echograms_container"],
                    chunks_sv_data,
                    kwargs.get("colormap", "ocean_r"),
                )
                side_tasks.append(future_plot_seabed)

        in_flight.append(future)

        if len(in_flight) >= batch_size:
            finished = next(as_completed(in_flight))
            res = finished.result()
            if isinstance(res, dict):
                file_results.append(res)
            in_flight.remove(finished)

    # wait for remaining main tasks
    for fut in in_flight:
        res = fut.result()
        if isinstance(res, dict):
            file_results.append(res)

    # wait for side tasks
    for fut in side_tasks:
        res = fut.result()
        if fut in [ft for _, ft in netcdf_outputs]:
            nc_result_map[[n for n, ft in netcdf_outputs if ft == fut][0]] = res

    return file_results, nc_result_map, netcdf_outputs


def _register_export(
    export_key: str,
    output_container: str,
    start_run: datetime,
    end_run: datetime,
    file_results: list[dict],
    nc_result_map: dict[str, list],
    denoise_params: dict,
    combined_zip_path: Optional[str],
    combined_zip_size: Optional[int],
) -> None:
    """Store export details and processed file references in the DB."""
    base_url = get_container_base_url(output_container)

    with PostgresDB() as db_connection:
        export_service = ExportService(db_connection)
        export_id, _ = export_service.insert_export(
            container_name=output_container,
            base_url=base_url,
            start_date=start_run,
            end_date=end_run,
            num_files=len(file_results),
            denoise_params=denoise_params,
            combined_netcdf_path=combined_zip_path,
            combined_netcdf_size=combined_zip_size,
            export_key=export_key,
        )

        for res in file_results:
            fid = res.get("file_id")
            if fid is None:
                continue
            nc_info = nc_result_map.get(res["file_name"], [])
            nc_entry = nc_info[0] if nc_info else None
            export_service.add_file(
                export_id,
                fid,
                res.get("echogram_files"),
                sv_zarr_path=res.get("location"),
                denoised_zarr_path=(
                    f"{res['cruise_id']}/{res['file_name']}/{res['file_name']}_denoised.zarr"
                    if res.get("denoised")
                    else None
                ),
                netcdf_path=nc_entry["path"] if nc_entry else None,
                netcdf_size=nc_entry["size"] if nc_entry else None,
            )


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

    start_run = datetime.utcnow()

    # Generate export key early so the NetCDF archive can be named accordingly
    export_key = f"{output_container}-{uuid.uuid4().hex[:6]}"

    files_list = _prepare_file_list(
        source_directory,
        cruise_id,
        load_from_blobstorage,
        source_container,
        get_list_from_db,
        start_datetime,
        end_datetime,
        reprocess,
    )

    print('source_directory:', source_directory)
    total_files = len(files_list)
    print(f"Total files to process: {total_files}")

    chunks_echodata = {
        'ping_time': chunks_ping_time,
        'range_sample': chunks_range_sample,
    }
    chunks_sv_data = {
        'ping_time': chunks_ping_time,
        'depth': 1000,
    }

    process_kwargs = dict(
        cruise_id=cruise_id,
        load_from_blobstorage=load_from_blobstorage,
        source_container=source_container,
        output_container=output_container,
        reprocess=reprocess,
        chunks_ping_time=chunks_ping_time,
        chunks_range_sample=chunks_range_sample,
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
        plot_echograms=plot_echograms,
        save_to_netcdf=save_to_netcdf,
    )

    file_results, nc_result_map, netcdf_outputs = _process_files(
        files_list,
        batch_size,
        chunks_echodata,
        chunks_sv_data,
        **process_kwargs,
    )

    combined_zip_path = None
    combined_zip_size = None
    if save_to_netcdf and file_results:
        first_start = min(datetime.fromisoformat(r["file_start_time"]) for r in file_results)
        last_end = max(datetime.fromisoformat(r["file_end_time"]) for r in file_results)
        duration_hours = (last_end - first_start).total_seconds() / 3600
        if duration_hours <= 24:
            future_zip = trigger_netcdf_flow.submit(
                file_list=[ft for _, ft in netcdf_outputs],
                container=output_container,
                zip_name=export_key,
            )
            future_zip.wait()
            combined_zip_path = f"{export_key}.zip"
            combined_zip_size = get_blob_size(output_container, combined_zip_path)

    if os.path.exists('/tmp/oceanstream/netcdfdata'):
        shutil.rmtree('/tmp/oceanstream/netcdfdata', ignore_errors=True)

    end_run = datetime.utcnow()

    denoise_params = {
        'mask_impulse_noise': mask_impulse_noise.dict() if mask_impulse_noise else None,
        'mask_attenuated_signal': mask_attenuated_signal.dict() if mask_attenuated_signal else None,
        'mask_transient_noise': mask_transient_noise.dict() if mask_transient_noise else None,
        'remove_background_noise': remove_background_noise.dict() if remove_background_noise else None,
        'apply_seabed_mask': apply_seabed_mask
    }

    _register_export(
        export_key=export_key,
        output_container=output_container,
        start_run=start_run,
        end_run=end_run,
        file_results=file_results,
        nc_result_map=nc_result_map,
        denoise_params=denoise_params,
        combined_zip_path=combined_zip_path,
        combined_zip_size=combined_zip_size,
    )

    logging.info(f"All batches have been processed. Export key: {export_key}")


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
