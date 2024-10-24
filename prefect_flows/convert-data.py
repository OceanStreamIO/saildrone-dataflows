import logging
import os
import sys
import time

from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from prefect import flow, task
from dask.distributed import Client
from prefect_dask import DaskTaskRunner, get_dask_client
from prefect.cache_policies import Inputs

from saildrone.process import process_file
from saildrone.store import save_zarr_store, ensure_container_exists
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
    retries=3,
    retry_delay_seconds=1,
    cache_policy=input_cache_policy,
    task_run_name="process-{file_path.stem}",
)
def process_single_file(file_path: Path, survey_id=None, sonar_model='EK80') -> None:
    print('Processing file:', file_path)

    process_file(file_path, survey_id, sonar_model,
                 calibration_file=CALIBRATION_FILE,
                 output_path=ECHODATA_OUTPUT_PATH,
                 chunks={"ping_time": 1000, "range_sample": -1},
                 processed_container_name=PROCESSED_CONTAINER_NAME)

    print(f"Processed Sv for {file_path.name}")


def convert_raw_data(files: List[Path], survey_id) -> None:
    task_futures = []
    print('Processing files:', files)
    for file_path in files:
        future = process_single_file.submit(file_path, survey_id)
        task_futures.append(future)

    # Wait for all tasks in the batch to complete
    for future in task_futures:
        future.result()


@flow(task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS))
def load_and_convert_files_to_zarr(source_directory: str, map_to_directory: str, cruise_id: str, survey_name: str,
                                   vessel: str, start_port: str, end_port: str, start_date: str, end_date: str,
                                   description: Optional[str], batch_size: int) -> None:
    """
    Load raw files from the source directory, insert/update survey record, and convert them to Zarr format.

    Args:
        source_directory (str): The directory containing the raw files.
        map_to_directory (str): The directory to map the raw files to.
        cruise_id (str): The unique ID of the cruise.
        survey_name (str): The name of the survey.
        vessel (str): The vessel used in the survey.
        start_port (str): The start port of the survey.
        end_port (str): The end port of the survey.
        start_date (str): The start date of the survey in the format YYYY-MM-DD.
        end_date (str): The end date of the survey in the format YYYY-MM-DD.
        description (Optional[str]): Optional description of the survey.
        batch_size (int): The number of files to process in each batch.
    """

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

    raw_files = load_local_files(source_directory, map_to_directory)
    total_files = len(raw_files)
    print(f"Total files to process: {total_files}")

    # Process files in batches
    for i in range(0, total_files, batch_size):
        batch_files = raw_files[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}")
        convert_raw_data(batch_files, survey_id)

    logging.info("All batches have been processed.")


if __name__ == "__main__":
    client = Client(address=DASK_CLUSTER_ADDRESS)

    ensure_container_exists(CONVERTED_CONTAINER_NAME)
    ensure_container_exists(PROCESSED_CONTAINER_NAME)

    try:
        # Start the flow
        load_and_convert_files_to_zarr.serve(
            name='convert-raw-files-to-zarr',
            parameters={
                'source_directory': RAW_DATA_LOCAL,
                'map_to_directory': RAW_DATA_LOCAL,
                'cruise_id': '',
                'survey_name': '',
                'vessel': '',
                'start_port': '',
                'end_port': '',
                'start_date': '2024-05-01',
                'end_date': '2024-06-30',
                'description': '',
                'batch_size': BATCH_SIZE
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception occurred: {str(e)}', exc_info=True)
        sys.exit(1)
