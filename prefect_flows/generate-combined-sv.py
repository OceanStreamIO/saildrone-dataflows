import logging
import os
import sys
import traceback

from typing import List, Optional, Union
from dotenv import load_dotenv
from dask.distributed import Client
from prefect import flow, task
from prefect_dask import DaskTaskRunner, get_dask_client
from prefect.cache_policies import Inputs
from prefect.artifacts import create_markdown_artifact

from saildrone.store import FileSegmentService
from saildrone.process.concat import merge_location_data, optimize_zarr_store, concatenate_and_rechunk
from saildrone.store import (PostgresDB, SurveyService, list_zarr_files, open_zarr_store, generate_container_name,
                             ensure_container_exists, save_zarr_store)

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
def process_single_file(file, file_name, source_container_name, cruise_id, chunks, temp_container_name, file_index, total):
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

    try:
        print(f"Processing file {zarr_path} with frequencies {file_freqs}")

        # Open the Zarr store lazily with Dask
        ds = open_zarr_store(zarr_path, cruise_id=cruise_id, container_name=source_container_name, chunks=chunks)

        # Merge location data
        ds = merge_location_data(ds, location_data)

        # Save the dataset to a temporary Zarr store
        category = "short_pulse" if file_freqs == "38000.0,200000.0" else "long_pulse" if file_freqs == "38000.0" else "exported_ds"

        temp_path = f"{file_name}_{file_index}.zarr"
        zarr_store = save_zarr_store(ds, container_name=temp_container_name, zarr_path=temp_path)
        # optimize_zarr_store(temp_path)

        return zarr_store, category
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

        raise e


@task(
    task_run_name="process-batch-{batch_index}",
)
def process_batch(batch_files, source_container_name, cruise_id, chunks, temp_container_name, batch_index):
    """
    Submit individual file processing as futures and return the results.
    """
    futures = []

    for idx, file in enumerate(batch_files):
        future = process_single_file.submit(file, file['file_name'], source_container_name, cruise_id, chunks,
                                            temp_container_name, idx, len(batch_files))
        futures.append(future)

    results = {
        "short_pulse": [],
        "long_pulse": [],
        "exported_ds": []
    }

    # Collect results as they complete
    for future in futures:
        try:
            zarr_store, category = future.result()
            results[category].append(zarr_store)
        except Exception as e:
            print(f"Error processing file: {e}")

    return results


def concatenate_zarr_files(files, source_container_name, cruise_id=None, chunks=None, batch_size=10, temp_container_name=None):
    ds_list = {
        "short_pulse": [],
        "long_pulse": [],
        "exported_ds": []
    }

    futures = []
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        future = process_batch.submit(batch_files, source_container_name, cruise_id, chunks, temp_container_name, i)
        futures.append(future)

    for future in futures:
        batch_results = future.result()
        # Accumulate paths
        ds_list["short_pulse"].extend(batch_results["short_pulse"])
        ds_list["long_pulse"].extend(batch_results["long_pulse"])
        ds_list["exported_ds"].extend(batch_results["exported_ds"])

    short_pulse_ds = concatenate_and_rechunk(ds_list["short_pulse"], chunks=chunks) if ds_list["short_pulse"] else None
    long_pulse_ds = concatenate_and_rechunk(ds_list["long_pulse"], chunks=chunks) if ds_list["long_pulse"] else None
    exported_ds = concatenate_and_rechunk(ds_list["exported_ds"], chunks=chunks) if ds_list["exported_ds"] else None

    return short_pulse_ds, long_pulse_ds, exported_ds


@flow(task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS))
def load_and_process_files_to_zarr(cruise_id: str,
                                   source_container: str,
                                   output_container: str,
                                   chunks_ping_time: int,
                                   chunks_depth: Optional[int],
                                   batch_size: int = BATCH_SIZE):
    with PostgresDB() as db_connection:
        survey_service = SurveyService(db_connection)

        # Check if a survey with the given cruise_id exists
        survey_id = survey_service.get_survey_by_cruise_id(cruise_id)

        if not survey_id:
            survey_id = survey_service.insert_survey(cruise_id)
            logging.info(f"Inserted new survey with cruise_id: {cruise_id}")

        file_service = FileSegmentService(db_connection)
        files_list = file_service.get_files_by_survey_id(survey_id)

    total_files = len(files_list)
    print(f"Total files to process: {total_files}")

    chunks = {
        'ping_time': chunks_ping_time,
        'depth': chunks_depth
    }

    temp_container_name = generate_container_name(cruise_id)
    ensure_container_exists(temp_container_name)

    short_pulse_ds, long_pulse_ds, exported_ds = concatenate_zarr_files(
        files_list,
        source_container,
        cruise_id=cruise_id,
        batch_size=batch_size,
        temp_container_name=temp_container_name,
        chunks=chunks)

    if short_pulse_ds:
        save_zarr_store(short_pulse_ds, container_name=output_container, zarr_path=f"{cruise_id}/short_pulse.zarr")

    if long_pulse_ds:
        save_zarr_store(long_pulse_ds, container_name=output_container, zarr_path=f"{cruise_id}/long_pulse.zarr")

    if exported_ds:
        save_zarr_store(exported_ds, container_name=output_container, zarr_path=f"{cruise_id}/{cruise_id}.zarr")

    logging.info("All batches have been processed.")


if __name__ == "__main__":
    client = Client(address=DASK_CLUSTER_ADDRESS)

    try:
        # Start the flow
        load_and_process_files_to_zarr.serve(
            name='generate-combined-sv',
            parameters={
                'cruise_id': '',
                'source_container': PROCESSED_CONTAINER_NAME,
                'output_container': '',
                'chunks_ping_time': 500,
                'chunks_depth': 500,
                'batch_size': BATCH_SIZE
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception occurred: {str(e)}', exc_info=True)
        sys.exit(1)
