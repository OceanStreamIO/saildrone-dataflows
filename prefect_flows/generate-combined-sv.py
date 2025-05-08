import logging
import os
import sys
import traceback
import xarray as xr

from typing import List, Optional, Union, NamedTuple
from dotenv import load_dotenv
from dask import delayed, compute
from dask.distributed import Client
from prefect import flow, task
from prefect_dask import DaskTaskRunner, get_dask_client
from prefect.cache_policies import Inputs
from prefect.states import Completed

from prefect.artifacts import create_markdown_artifact

from saildrone.store import FileSegmentService
from saildrone.process.concat import merge_location_data, concatenate_and_rechunk
from saildrone.store import (PostgresDB, SurveyService, open_zarr_store, generate_container_name,
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


def get_write_mode(key: str, write_mode_dict: dict) -> str:
    mode = write_mode_dict.setdefault(key, 'w')
    write_mode_dict[key] = 'a'
    return mode


class FileResult(NamedTuple):
    zarr_path: str
    category: str


@delayed
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
        print(f"Processing file {file_name} with frequencies {file_freqs}; {file_index + 1}/{total}")

        # Open the Zarr store lazily with Dask
        ds = open_zarr_store(zarr_path, cruise_id=cruise_id, container_name=source_container_name, chunks=chunks)

        # Merge location data
        ds = merge_location_data(ds, location_data)

        # Save the dataset to a temporary Zarr store
        category = "short_pulse" if file_freqs == "38000.0,200000.0" else "long_pulse" if file_freqs == "38000.0" else "exported_ds"

        temp_path = f"{file_name}_{file_index}.zarr"
        zarr_path = save_zarr_store(ds, container_name=temp_container_name, zarr_path=temp_path)
        # optimize_zarr_store(temp_path)

        return FileResult(zarr_path=zarr_path, category=category)
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

        return FileResult(zarr_path=None, category=None)


def process_batch(batch_files, source_container_name, cruise_id, chunks, temp_container_name, batch_index):
    """
    Submit individual file processing as futures and return the results.
    """
    results = {
        "short_pulse": [],
        "long_pulse": [],
        "exported_ds": []
    }

    delayed_results = []

    for idx, file in enumerate(batch_files):
        delayed_result = process_single_file(
            file, file['file_name'], source_container_name, cruise_id, chunks,
            temp_container_name, idx, len(batch_files)
        )
        delayed_results.append(delayed_result)

    with get_dask_client() as dask_client:
        futures = dask_client.compute(delayed_results)
        computed_results = dask_client.gather(futures)

        for result in computed_results:
            if result.zarr_path is not None:
                results[result.category].append(result.zarr_path)

        return results


def concatenate_zarr_files(files, source_container_name, output_container, cruise_id=None, chunks=None,
                           temp_container_name=None,
                           write_mode_dict=None,
                           batch_index=None):

    ds_list = process_batch(files, source_container_name, cruise_id, chunks, temp_container_name, batch_index)

    for category, delayed_datasets in ds_list.items():
        valid_datasets = [d for d in delayed_datasets if d is not None]
        if not valid_datasets:
            continue

        combined_ds = concatenate_and_rechunk(valid_datasets, container_name=temp_container_name, chunks=chunks)
        zarr_path = f"{cruise_id}/{category}.zarr" if category != "exported_ds" else f"{cruise_id}/{cruise_id}.zarr"
        mode = get_write_mode(category, write_mode_dict)
        print(f"Saving {category} dataset with mode: {mode}")
        print(f"  - dims: {combined_ds.dims}")
        print(f"  - variables: {list(combined_ds.data_vars)}")
        print(f"  - chunks: {combined_ds.chunks}")

        save_zarr_store(combined_ds,
                        container_name=output_container,
                        zarr_path=zarr_path,
                        mode=mode,
                        append_dim="ping_time")


@flow(task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS))
def generate_combined_sv(cruise_id: str,
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
    batch_index = 0
    write_mode_dict = {}

    for i in range(0, total_files, batch_size):
        batch_files = files_list[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}")

        try:
            concatenate_zarr_files(
                batch_files,
                source_container,
                output_container,
                cruise_id=cruise_id,
                temp_container_name=temp_container_name,
                chunks=chunks,
                write_mode_dict=write_mode_dict,
                batch_index=batch_index
            )
            batch_index += 1
        except Exception as e:
            logging.error(f"Error saving Zarr store: {e}")
            markdown_report = f"""# Error saving Zarr store: {e}
            {str(e)}
            ## Error details
            - **Error Message**: {str(e)}
            - **Traceback**: {traceback.format_exc()}
            """

            create_markdown_artifact(markdown_report)

            raise e

    logging.info("All batches have been processed.")


if __name__ == "__main__":
    client = Client(address=DASK_CLUSTER_ADDRESS)

    try:
        generate_combined_sv.serve(
            name='generate-combined-sv',
            parameters={
                'cruise_id': '',
                'source_container': PROCESSED_CONTAINER_NAME,
                'output_container': '',
                'chunks_ping_time': 500,
                'chunks_depth': 500,
                'batch_size': 4
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception occurred: {str(e)}', exc_info=True)
        sys.exit(1)
