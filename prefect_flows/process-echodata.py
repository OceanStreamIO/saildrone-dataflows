import logging
import os
import sys
import traceback
from datetime import datetime

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

from saildrone.process import process_converted_file
from saildrone.store import ensure_container_exists, FileSegmentService
from saildrone.utils import load_local_files
from saildrone.store import PostgresDB, SurveyService, list_zarr_files

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

CHUNKS = {"ping_time": 500, "range_sample": -1}
CHUNKS_DENOISING = {"ping_time": 500, "depth": 500}
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
    depth_bin: int = Field(default=10, description="Donwsampling bin size along vertical range variable (`range_var`) in meters.")
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
def process_single_file(source_path: Path, **kwargs):
    try:
        cruise_id = kwargs.get('cruise_id')
        load_from_blobstorage = kwargs.get('load_from_blobstorage')
        source_container = kwargs.get('source_container')
        save_to_blobstorage = kwargs.get('save_to_blobstorage')
        output_container = kwargs.get('output_container')
        save_to_directory = kwargs.get('save_to_directory')
        output_directory = kwargs.get('output_directory')
        reprocess = kwargs.get('reprocess')
        plot_echograms = kwargs.get('plot_echograms')
        compute_nasc = kwargs.get('compute_nasc')
        compute_mvbs = kwargs.get('compute_mvbs')
        echograms_container = kwargs.get('echograms_container')
        encode_mode = kwargs.get('encode_mode')
        colormap = kwargs.get('colormap')
        waveform_mode = kwargs.get('waveform_mode')
        depth_offset = kwargs.get('depth_offset')
        chunks_ping_time = kwargs.get('chunks_ping_time')
        chunks_range_sample = kwargs.get('chunks_range_sample')
        mask_transient_noise = kwargs.get('mask_transient_noise')
        mask_impulse_noise = kwargs.get('mask_impulse_noise')
        mask_attenuated_signal = kwargs.get('mask_attenuated_signal')
        remove_background_noise = kwargs.get('remove_background_noise')

        chunks = {
            'ping_time': chunks_ping_time,
            'range_sample': chunks_range_sample
        }

        output_path = output_directory
        converted_container_name = None
        processed_container_name = None

        if save_to_directory is not True:
            output_path = None

        if load_from_blobstorage is True:
            converted_container_name = source_container

        if save_to_blobstorage is True:
            processed_container_name = output_container

        process_converted_file(source_path,
                               cruise_id=cruise_id,
                               output_path=output_path,
                               chunks=chunks,
                               load_from_blobstorage=load_from_blobstorage,
                               converted_container_name=converted_container_name,
                               save_to_blobstorage=save_to_blobstorage,
                               save_to_directory=save_to_directory,
                               processed_container_name=processed_container_name,
                               reprocess=reprocess,
                               depth_offset=depth_offset,
                               plot_echograms=plot_echograms,
                               compute_nasc=compute_nasc,
                               compute_mvbs=compute_mvbs,
                               echograms_container=echograms_container,
                               gps_container_name=GPSDATA_CONTAINER_NAME,
                               encode_mode=encode_mode,
                               colormap=colormap,
                               waveform_mode=waveform_mode,
                               mask_transient_noise=mask_transient_noise,
                               mask_impulse_noise=mask_impulse_noise,
                               mask_attenuated_signal=mask_attenuated_signal,
                               remove_background_noise=remove_background_noise,
                               chunks_denoising=CHUNKS_DENOISING
                               )
        print(f"Processed Sv for {source_path.name}")
    except Exception as e:
        print(f"Error processing file: {source_path.name}: ${str(e)}")

        markdown_report = f"""# Error report for {source_path.name}
        Error occurred while processing the file: {source_path}
        {str(e)}
        """
        create_markdown_artifact(markdown_report)

        return Completed(message="Task completed with errors")


def process_raw_data(files: List[Path], **kwargs) -> None:
    task_futures = []

    for source_path in files:
        future = process_single_file.submit(source_path, **kwargs)
        task_futures.append(future)

    # Wait for all tasks in the batch to complete
    for future in task_futures:
        future.result()


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
                                   reprocess: bool,
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
                                   batch_size: int = BATCH_SIZE):
    file_names = None
    with PostgresDB() as db_connection:
        survey_service = SurveyService(db_connection)

        # Check if a survey with the given cruise_id exists
        survey_id = survey_service.get_survey_by_cruise_id(cruise_id)

        if not survey_id:
            survey_id = survey_service.insert_survey(cruise_id)
            logging.info(f"Inserted new survey with cruise_id: {cruise_id}")

        if get_list_from_db:
            file_service = FileSegmentService(db_connection)
            if not reprocess:
                condition = 'AND processed IS NOT True'
            else:
                condition = ''

            if start_datetime and end_datetime:
                condition += f" AND file_start_time > '{start_datetime}' AND file_end_time < '{end_datetime}'"

            file_names = file_service.get_files_list_with_condition(survey_id, condition)

    if file_names and not load_from_blobstorage:
        files_list = [Path(source_directory) / f"{file_name}.raw" for file_name in file_names]
    elif file_names and load_from_blobstorage:
        files_list = list_zarr_files(source_container, cruise_id=cruise_id, file_names=file_names)
    elif load_from_blobstorage:
        files_list = list_zarr_files(source_container, cruise_id=cruise_id)
    else:
        files_list = load_local_files(source_directory, source_directory, '*.zarr')

    print('source_directory:', source_directory)
    total_files = len(files_list)
    print(f"Total files to process: {total_files}")

    in_flight = []

    for src in files_list:
        # Submit task
        future = process_single_file.submit(src,
                                            cruise_id=cruise_id,
                                            load_from_blobstorage=load_from_blobstorage,
                                            source_container=source_container,
                                            output_container=output_container,
                                            reprocess=reprocess,
                                            plot_echograms=plot_echograms,
                                            compute_nasc=compute_nasc,
                                            compute_mvbs=compute_mvbs,
                                            echograms_container=echograms_container,
                                            save_to_blobstorage=save_to_blobstorage,
                                            save_to_directory=save_to_directory,
                                            output_directory=output_directory,
                                            encode_mode=encode_mode,
                                            colormap=colormap,
                                            waveform_mode=waveform_mode,
                                            depth_offset=depth_offset,
                                            chunks_ping_time=chunks_ping_time,
                                            chunks_range_sample=chunks_range_sample,
                                            mask_impulse_noise=mask_impulse_noise,
                                            mask_attenuated_signal=mask_attenuated_signal,
                                            mask_transient_noise=mask_transient_noise,
                                            remove_background_noise=remove_background_noise,
                                            apply_seabed_mask=apply_seabed_mask
                                            )
        in_flight.append(future)

        # Throttle when max concurrent tasks reached
        if len(in_flight) >= batch_size:
            finished = next(as_completed(in_flight))
            in_flight.remove(finished)

    # Wait for remaining tasks
    for future_task in in_flight:
        future_task.result()

    # wait(in_flight)

    # Process files in batches
    # for i in range(0, total_files, batch_size):
    #     batch_files = files_list[i:i + batch_size]
    #     print(f"Processing batch {i // batch_size + 1}")
    #     process_raw_data(batch_files,
    #                      cruise_id=cruise_id,
    #                      load_from_blobstorage=load_from_blobstorage,
    #                      source_container=source_container,
    #                      output_container=output_container,
    #                      reprocess=reprocess,
    #                      plot_echograms=plot_echograms,
    #                      compute_nasc=compute_nasc,
    #                      compute_mvbs=compute_mvbs,
    #                      echograms_container=echograms_container,
    #                      save_to_blobstorage=save_to_blobstorage,
    #                      save_to_directory=save_to_directory,
    #                      output_directory=output_directory,
    #                      encode_mode=encode_mode,
    #                      colormap=colormap,
    #                      waveform_mode=waveform_mode,
    #                      depth_offset=depth_offset,
    #                      chunks_ping_time=chunks_ping_time,
    #                      chunks_range_sample=chunks_range_sample,
    #                      mask_impulse_noise=mask_impulse_noise,
    #                      mask_attenuated_signal=mask_attenuated_signal,
    #                      mask_transient_noise=mask_transient_noise,
    #                      remove_background_noise=remove_background_noise,
    #                      apply_seabed_mask=apply_seabed_mask
    #                      )

    logging.info("All batches have been processed.")


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
                'reprocess': False,
                'plot_echograms': False,
                'compute_nasc': True,
                'compute_mvbs': True,
                'echograms_container': WEBAPP_CONTAINER_NAME,
                'encode_mode': 'complex',
                'colormap': 'ocean_r',
                'waveform_mode': 'CW',
                'depth_offset': 0,
                'chunks_ping_time': 500,
                'chunks_range_sample': 500,
                'mask_transient_noise': None,
                'mask_impulse_noise': None,
                'mask_attenuated_signal': None,
                'remove_background_noise': None,
                'apply_seabed_mask': False,
                'batch_size': BATCH_SIZE
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception occurred: {str(e)}', exc_info=True)
        sys.exit(1)
