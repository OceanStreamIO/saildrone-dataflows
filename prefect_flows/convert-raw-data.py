import logging
import os
import sys
import time

from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from prefect import flow, task
from dask.distributed import Client
from prefect_dask import DaskTaskRunner
from prefect.cache_policies import Inputs
from prefect.states import Completed

from saildrone.process import convert_file_and_save
from saildrone.store import ensure_container_exists
from saildrone.utils import load_local_files
from saildrone.store import PostgresDB, SurveyService, FileSegmentService

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

AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
if not AZURE_STORAGE_CONNECTION_STRING:
    logging.error('AZURE_STORAGE_CONNECTION_STRING environment variable not set.')
    sys.exit(1)


@task(
    retries=10,
    retry_delay_seconds=[10, 30, 60],
    cache_policy=input_cache_policy,
    retry_jitter_factor=0.1,
    refresh_cache=True,
    result_storage=None,
    task_run_name="convert-{file_path.stem}",
)
def convert_single_file(file_path: Path, cruise_id=None, store_to_directory=None, output_directory=None,
                        apply_calibration=None, store_to_blobstorage=None, blobstorage_container=None,
                        sonar_model='EK80'):
    load_dotenv()

    calibration_file = os.getenv('CALIBRATION_FILE')

    try:
        converted_container_name = None
        output_path = None
        if store_to_blobstorage:
            converted_container_name = blobstorage_container

        if store_to_directory and output_directory:
            output_path = output_directory

        if apply_calibration is not True:
            calibration_file = None

        convert_file_and_save(file_path, cruise_id, sonar_model,
                              calibration_file=calibration_file,
                              converted_container_name=converted_container_name,
                              output_path=output_path)
    except Exception as e:
        print(f"Error processing file: {file_path.name}" + str(e))

        return Completed(message="Task completed with errors")


def convert_raw_data(files: List[Path], cruise_id=None, store_to_directory=None, output_directory=None,
                     apply_calibration=None, store_to_blobstorage=None, blobstorage_container=None):
    task_futures = []

    for file_path in files:
        future = convert_single_file.submit(file_path,
                                            cruise_id=cruise_id,
                                            store_to_directory=store_to_directory,
                                            output_directory=output_directory,
                                            apply_calibration=apply_calibration,
                                            store_to_blobstorage=store_to_blobstorage,
                                            blobstorage_container=blobstorage_container)
        task_futures.append(future)

    # Wait for all tasks in the batch to complete
    for future in task_futures:
        future.result()


@flow(task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS))
def load_and_convert_files_to_zarr(source_directory: str,
                                   get_list_from_db: bool,
                                   cruise_id: str, survey_name: str,
                                   vessel: str, start_port: str, end_port: str, start_date: str, end_date: str,
                                   description: Optional[str],
                                   store_to_directory: Optional[bool],
                                   apply_calibration: Optional[bool],
                                   output_directory: Optional[str],
                                   store_to_blobstorage: Optional[bool],
                                   blobstorage_container: Optional[str],
                                   batch_size: int) -> None:
    raw_files = []
    with PostgresDB() as db_connection:
        survey_service = SurveyService(db_connection)

        # Check if a survey with the given cruise_id exists
        survey_id = survey_service.get_survey_by_cruise_id(cruise_id)

        if survey_id:
            # Update the existing survey record
            survey_service.update_survey(survey_id, survey_name, vessel, start_port, end_port, start_date, end_date,
                                         description)
            logging.info(f"Updated survey with cruise_id: {cruise_id}")
        else:
            # Insert a new survey record
            survey_id = survey_service.insert_survey(cruise_id, survey_name, vessel, start_port, end_port, start_date,
                                                     end_date, description)
            logging.info(f"Inserted new survey with cruise_id: {cruise_id}")

        if get_list_from_db:
            file_service = FileSegmentService(db_connection)
            file_names = file_service.get_files_with_condition(survey_id, 'converted IS NOT True')
            raw_files = [Path(source_directory) / file_name for file_name in file_names]

    if not get_list_from_db:
        raw_files = load_local_files(source_directory, source_directory)

    total_files = len(raw_files)
    print(f"Total files to process: {total_files}")

    # Process files in batches
    for i in range(0, total_files, batch_size):
        batch_files = raw_files[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}")
        convert_raw_data(batch_files,
                         cruise_id=cruise_id,
                         store_to_directory=store_to_directory,
                         output_directory=output_directory,
                         store_to_blobstorage=store_to_blobstorage,
                         apply_calibration=apply_calibration,
                         blobstorage_container=blobstorage_container)

    logging.info("All batches have been processed.")


if __name__ == "__main__":
    with PostgresDB() as db:
        db.create_tables()

    client = Client(address=DASK_CLUSTER_ADDRESS)

    ensure_container_exists(CONVERTED_CONTAINER_NAME)
    ensure_container_exists(PROCESSED_CONTAINER_NAME)

    try:
        # Start the flow
        load_and_convert_files_to_zarr.serve(
            name='convert-raw-files-to-zarr',
            parameters={
                'source_directory': RAW_DATA_LOCAL,
                'get_list_from_db': False,
                'cruise_id': '',
                'survey_name': '',
                'vessel': '',
                'start_port': '',
                'end_port': '',
                'start_date': '2024-05-01',
                'end_date': '2024-06-30',
                'description': '',
                'store_to_directory': True,
                'apply_calibration': True,
                'output_directory': ECHODATA_OUTPUT_PATH,
                'store_to_blobstorage': False,
                'blobstorage_container': os.getenv('CONVERTED_CONTAINER_NAME'),
                'batch_size': BATCH_SIZE
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception occurred: {str(e)}', exc_info=True)
        sys.exit(1)
