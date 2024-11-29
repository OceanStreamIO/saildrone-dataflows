import logging
import os
import sys
import traceback

from pathlib import Path
from typing import List, Optional, Union, TypedDict

from dotenv import load_dotenv
from prefect import flow, task
from dask.distributed import Client
from prefect_dask import DaskTaskRunner, get_dask_client
from prefect.cache_policies import Inputs
from prefect.states import Completed

from saildrone.process import process_converted_file
from saildrone.store import ensure_container_exists
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
GPSDATA_CONTAINER_NAME = os.getenv('GPSDATA_CONTAINER_NAME')
CALIBRATION_FILE = os.getenv('CALIBRATION_FILE')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 6))

CHUNKS = {"ping_time": 500, "range_sample": -1}

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
    task_run_name="process-{source_path.stem}",
)
def process_single_file(source_path: Path,
                        cruise_id=None,
                        source_container=None,
                        output_container=None,
                        load_from_blobstorage=None,
                        reprocess=False,
                        save_to_blobstorage=None,
                        save_to_directory=None,
                        chunks_ping_time=None,
                        chunks_range_sample=None) -> None:
    try:
        chunks = {
            'ping_time': chunks_ping_time,
            'range_sample': chunks_range_sample
        }

        output_path = ECHODATA_OUTPUT_PATH
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
                               reprocess=reprocess,
                               gps_container_name=GPSDATA_CONTAINER_NAME,
                               save_to_blobstorage=save_to_blobstorage,
                               save_to_directory=save_to_directory,
                               processed_container_name=processed_container_name)
        print(f"Processed Sv for {source_path.name}")
    except Exception as e:
        print(f"Error processing file: {source_path.name}: ${str(e)}")

        return Completed(message="Task completed with errors")


def process_raw_data(files: List[Path],
                     cruise_id=None,
                     source_container=None,
                     output_container=None,
                     save_to_blobstorage=None,
                     load_from_blobstorage=None,
                     save_to_directory=None,
                     chunks_ping_time=None,
                     chunks_range_sample=None,
                     reprocess=None) -> None:
    task_futures = []

    for source_path in files:
        future = process_single_file.submit(source_path,
                                            cruise_id=cruise_id,
                                            source_container=source_container,
                                            output_container=output_container,
                                            save_to_blobstorage=save_to_blobstorage,
                                            load_from_blobstorage=load_from_blobstorage,
                                            save_to_directory=save_to_directory,
                                            chunks_ping_time=chunks_ping_time,
                                            chunks_range_sample=chunks_range_sample,
                                            reprocess=reprocess)
        task_futures.append(future)

    # Wait for all tasks in the batch to complete
    for future in task_futures:
        future.result()


@flow(task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS))
def load_and_process_files_to_zarr(source_directory: str, map_to_directory: str, cruise_id: str, survey_name: str,
                                   vessel: str, start_port: str, end_port: str, start_date: str, end_date: str,
                                   description: Optional[str],
                                   load_from_blobstorage: bool,
                                   save_to_blobstorage: bool,
                                   save_to_directory: bool,
                                   source_container: str,
                                   output_container: str,
                                   reprocess: bool,
                                   chunks_ping_time: int,
                                   chunks_range_sample: int,
                                   batch_size: int) -> None:
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
        load_from_blobstorage (bool): Whether to load the raw files from Azure Blob Storage.
        save_to_blobstorage (bool): Whether to save the converted files to Azure Blob Storage.
        save_to_directory (bool): Whether to save the converted files to a local directory.
        source_container (str): The name of the Azure Blob Storage container containing the raw files.
        output_container (str): The name of the Azure Blob Storage container to save the converted files.
        reprocess (bool): Whether to reprocess the files if they have already been processed.
        chunks_ping_time (int): The chunk size for the ping_time dimension.
        chunks_range_sample (int): The chunk size for the range_sample dimension.
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

    if load_from_blobstorage:
        files_list = list_zarr_files(source_container, cruise_id=cruise_id)
    else:
        files_list = load_local_files(source_directory, map_to_directory)

    total_files = len(files_list)

    # Process files in batches
    for i in range(0, total_files, batch_size):
        batch_files = files_list[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}")
        process_raw_data(batch_files,
                         cruise_id=cruise_id,
                         load_from_blobstorage=load_from_blobstorage,
                         source_container=source_container,
                         output_container=output_container,
                         reprocess=reprocess,
                         save_to_blobstorage=save_to_blobstorage,
                         save_to_directory=save_to_directory,
                         chunks_ping_time=chunks_ping_time,
                         chunks_range_sample=chunks_range_sample,
                         )

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
                'map_to_directory': RAW_DATA_LOCAL,
                'cruise_id': '',
                'survey_name': '',
                'vessel': '',
                'start_port': '',
                'end_port': '',
                'start_date': '2024-05-01',
                'end_date': '2024-06-30',
                'description': '',
                'load_from_blobstorage': False,
                'save_to_blobstorage': True,
                'save_to_directory': False,
                'source_container': CONVERTED_CONTAINER_NAME,
                'output_container': PROCESSED_CONTAINER_NAME,
                'reprocess': False,
                'chunks_ping_time': 500,
                'chunks_range_sample': -1,
                'batch_size': BATCH_SIZE
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception occurred: {str(e)}', exc_info=True)
        sys.exit(1)
