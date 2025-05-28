import logging
import os
import sys
import time
import traceback

from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from prefect import flow, task, get_client
from dask.distributed import Client
from prefect_dask import DaskTaskRunner
from prefect.cache_policies import Inputs
from prefect.states import Completed
from prefect.concurrency.sync import concurrency
from prefect.futures import as_completed

from saildrone.calibrate import apply_calibration as apply_calibration_fn
from saildrone.utils import load_local_files
from saildrone.store import PostgresDB, SurveyService, FileSegmentService, save_zarr_store

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
CALIBRATION_FILE = os.getenv('CALIBRATION_FILE')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 6))
MINIMUM_THROTTLE = int(os.getenv('MINIMUM_THROTTLE', 30))


@task(
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    cache_policy=input_cache_policy,
    retry_jitter_factor=0.1,
    refresh_cache=True,
    result_storage=None,
    log_prints=True,
    task_run_name="convertraw-{file_path.stem}",
)
def convert_single_file(file_path: Path,
                        cruise_id=None,
                        survey_db_id=None,
                        store_to_directory=None,
                        output_directory=None,
                        reprocess=None,
                        apply_calibration=None,
                        calibration_file=None,
                        store_to_blobstorage=None,
                        blobstorage_container=None,
                        chunks=None):
    load_dotenv()

    try:
        converted_container_name = None
        output_path = None
        if store_to_blobstorage:
            converted_container_name = blobstorage_container

        if store_to_directory and output_directory:
            output_path = output_directory

        if apply_calibration is not True:
            calibration_file = None

        print(f"Converting file: {file_path}, cruise_id: {cruise_id}, reprocess: {reprocess}")

        file_id = convert_file_and_save(
            file_path,
            cruise_id=cruise_id,
            survey_db_id=survey_db_id,
            sonar_model='EK80',
            calibration_file=calibration_file,
            output_path=output_path,
            reprocess=reprocess,
            converted_container_name=converted_container_name,
            chunks=chunks
        )

        return file_id

    except Exception as e:
        print(f"Error processing file: {file_path.name}" + str(e))
        traceback.print_exc()

        return Completed(message="Task completed with errors")


def convert_file_and_save(file_path: Path, cruise_id=None, survey_db_id=None, sonar_model='EK80',
                          calibration_file=None, output_path=None,
                          reprocess=None, converted_container_name=None, chunks=None) -> (int, str, str):
    file_name = file_path.stem
    print('Starting conversion for file:', file_name)

    with PostgresDB() as db_connection:
        file_segment_service = FileSegmentService(db_connection)

        # Check if the file has already been processed
        if file_segment_service.is_file_converted(file_name) and not reprocess:
            print(f'Skipping already converted file: {file_name}')
            return None

        file_info = file_segment_service.get_file_info(file_name)
        print(f"File info for {file_name}: {file_info}")

        try:
            echodata, zarr_path = convert_raw_file_to_echodata(file_name,
                                                               file_path,
                                                               cruise_id=cruise_id,
                                                               calibration_file=calibration_file,
                                                               container_name=converted_container_name,
                                                               sonar_model=sonar_model,
                                                               chunks=chunks)
        except Exception as e:
            print(f'Error converting file {file_path}: {e}')
            file_segment_service.update_file_record(file_info['id'], failed=True, error_details=str(e))

            return None

        if output_path is not None:
            output_zarr_path = f"{output_path}/{file_name}.zarr"
            echodata.to_zarr(output_zarr_path, overwrite=True)

        file_info = file_segment_service.get_file_info(file_name)

        print(f"File info {file_name}: {file_info}")

        if file_info is not None:
            file_id = file_info['id']

            if file_info['converted'] is True:
                return file_id

            file_segment_service.update_file_record(
                file_id,
                size=file_path.stat().st_size,
                location=str(file_path),
                last_modified=time.ctime(file_path.stat().st_mtime),
                converted=True
            )
        else:
            file_id = file_segment_service.insert_file_record(
                file_name,
                size=file_path.stat().st_size,
                location=str(file_path),
                survey_db_id=survey_db_id,
                last_modified=time.ctime(file_path.stat().st_mtime),
                converted=True
            )

        return file_id


def convert_raw_file_to_echodata(file_name, file_path, calibration_file=None,
                                 cruise_id=None, container_name=None, sonar_model='EK80', chunks=None):
    from echopype.convert.api import open_raw

    echodata = open_raw(file_path, sonar_model=sonar_model)

    if echodata.beam is None:
        return echodata, None

    if calibration_file:
        echodata = apply_calibration_fn(echodata, calibration_file)

    if cruise_id:
        zarr_path = f"{cruise_id}/{file_name}.zarr"
    else:
        zarr_path = f"{file_name}.zarr"

    if container_name is not None:
        if chunks is not None:
            echodata = echodata.chunk(chunks)

        save_zarr_store(echodata, container_name=container_name, zarr_path=zarr_path)

    return echodata, zarr_path


@flow(
    log_prints=True,
    task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS)
)
def load_and_convert_files_to_zarr(source_directory: str,
                                   get_list_from_db: bool,
                                   cruise_id: str,
                                   store_to_directory: Optional[bool],
                                   apply_calibration: Optional[bool],
                                   calibration_file: Optional[str] = None,
                                   reprocess: Optional[bool] = False,
                                   output_directory: Optional[str] = None,
                                   store_to_blobstorage: Optional[bool] = False,
                                   blobstorage_container: Optional[str] = None,
                                   chunks_ping_time: int = 2000,
                                   chunks_range_sample: int = -1,
                                   batch_size: int = BATCH_SIZE
                                   ) -> None:
    raw_files = []
    with PostgresDB() as db_connection:
        survey_service = SurveyService(db_connection)

        # Check if a survey with the given cruise_id exists
        survey_db_id = survey_service.get_survey_by_cruise_id(cruise_id)

        if not survey_db_id:
            print(f"No survey found with cruise ID: {cruise_id}")
            return

        if get_list_from_db:
            file_service = FileSegmentService(db_connection)
            if not reprocess:
                condition = 'AND converted IS NOT True'
            else:
                condition = ''

            print(f'Survey ID: {survey_db_id}')
            file_names = file_service.get_files_list_with_condition(survey_db_id, condition)
            raw_files = [Path(source_directory) / f"{file_name}.raw" for file_name in file_names]

    if not get_list_from_db:
        raw_files = load_local_files(source_directory, source_directory)

    chunks = {
        "ping_time": chunks_ping_time,
        "range_sample": chunks_range_sample
    }

    total_files = len(raw_files)
    print(f"Total files to process: {total_files}")

    in_flight = []
    for file_path in raw_files:
        future = convert_single_file.submit(file_path,
                                            cruise_id=cruise_id,
                                            survey_db_id=survey_db_id,
                                            reprocess=reprocess,
                                            store_to_directory=store_to_directory,
                                            output_directory=output_directory,
                                            apply_calibration=apply_calibration,
                                            calibration_file=calibration_file,
                                            store_to_blobstorage=store_to_blobstorage,
                                            blobstorage_container=blobstorage_container,
                                            chunks=chunks)
        in_flight.append(future)

        if len(in_flight) >= batch_size:
            done = next(as_completed(in_flight))
            try:
                done.result()
            finally:
                in_flight.remove(done)

    for future_task in in_flight:
        future_task.result()

    print("All files have been processed.")


if __name__ == "__main__":
    with PostgresDB() as db:
        db.create_tables()

    client = Client(address=DASK_CLUSTER_ADDRESS)

    try:
        # Start the flow
        load_and_convert_files_to_zarr.serve(
            name='convert-raw-files-to-zarr',
            parameters={
                'source_directory': RAW_DATA_LOCAL,
                'get_list_from_db': False,
                'cruise_id': '',
                'store_to_directory': True,
                'apply_calibration': True,
                'calibration_file': CALIBRATION_FILE,
                'reprocess': False,
                'output_directory': ECHODATA_OUTPUT_PATH,
                'store_to_blobstorage': False,
                'blobstorage_container': os.getenv('CONVERTED_CONTAINER_NAME'),
                'chunks_ping_time': 2000,
                'chunks_range_sample': -1,
                'batch_size': BATCH_SIZE
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception occurred: {str(e)}', exc_info=True)
        sys.exit(1)
