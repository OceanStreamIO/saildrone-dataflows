import logging
import os
import sys
import traceback
from datetime import datetime

from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from prefect import flow, task
from dask.distributed import Client
from prefect_dask import DaskTaskRunner, get_dask_client
from prefect.cache_policies import Inputs
from prefect.artifacts import create_markdown_artifact

from echopype import open_converted, combine_echodata as ep_combine_echodata

from saildrone.store import (FileSegmentService, PostgresDB, SurveyService, open_zarr_store,
                             save_dataset_to_netcdf, ensure_container_exists, save_zarr_store, list_zarr_files,
                             generate_container_name)

from saildrone.utils import load_local_files
from saildrone.store import PostgresDB, FileSegmentService

input_cache_policy = Inputs()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

load_dotenv()

RAW_DATA_MOUNT = os.getenv('RAW_DATA_MOUNT')
RAW_DATA_LOCAL = os.getenv('RAW_DATA_LOCAL')
DASK_CLUSTER_ADDRESS = os.getenv('DASK_CLUSTER_ADDRESS')
COMBINED_CONTAINER_NAME = os.getenv('COMBINED_CONTAINER_NAME')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 10))

AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
if not AZURE_STORAGE_CONNECTION_STRING:
    logging.error('AZURE_STORAGE_CONNECTION_STRING environment variable not set.')
    sys.exit(1)


@task
def process_converted_file(converted_file: str, chunks=None):
    """
    Processes a file only if it hasn't already been converted.
    """
    with PostgresDB() as db_connection:
        file_segment_service = FileSegmentService(db_connection)

        file_info = file_segment_service.get_file_info(converted_file)

        if file_info is None:
            return None

        if file_info['size'] < 6648203:
            logging.info(f"File {converted_file} is too small to process.")
            return None

    return open_converted(converted_file, chunks=chunks)


# @task(cache_policy=input_cache_policy)
# def combine_echodata(echodata_files, combined_zarr_path, ed_combined_name) -> None:
#     # Open (lazy-load) Zarr stores containing EchoData Objects, and lazily combine them
#
#     with get_dask_client() as client:
#         ed_future_list = []
#         for converted_file in echodata_files:
#             ed_future = client.submit(
#                 open_converted,
#                 converted_raw_path=converted_file,
#                 chunks={}
#             )
#             ed_future_list.append(ed_future)
#
#         ed_list = client.gather(ed_future_list)
#         ed_combined = ep_combine_echodata(ed_list)
#
#         # Save the combined EchoData object to a new Zarr store
#         # The appending operation only happens when relevant data needs to be save to disk
#         ed_combined.to_zarr(
#             combined_zarr_path / ed_combined_name,
#             overwrite=True,
#             compute=True,
#         )

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
def load_and_combine_zarr_stores(cruise_id: str,
                                 source_container: str,
                                 start_datetime: Optional[datetime],
                                 end_datetime: Optional[datetime],
                                 output_container: str,
                                 combined_zarr_name: str,
                                 chunks_ping_time: int,
                                 chunks_range_sample: Optional[int],
                                 batch_size: int) -> None:
    file_names = None
    with PostgresDB() as db_connection:
        survey_service = SurveyService(db_connection)

        # Check if a survey with the given cruise_id exists
        survey_id = survey_service.get_survey_by_cruise_id(cruise_id)

        if not survey_id:
            survey_id = survey_service.insert_survey(cruise_id)
            logging.info(f"Inserted new survey with cruise_id: {cruise_id}")

        file_service = FileSegmentService(db_connection)
        condition = ""

        if start_datetime and end_datetime:
            condition += f" AND file_start_time > '{start_datetime}' AND file_end_time < '{end_datetime}'"
            file_names = file_service.get_files_list_with_condition(survey_id, condition)

    echodata_files = list_zarr_files(source_container, cruise_id=cruise_id, file_names=file_names)
    total_files = len(echodata_files)
    temp_container_name = generate_container_name(cruise_id)
    ensure_container_exists(temp_container_name)
    batch_index = 0
    write_mode_dict = {}

    for i in range(0, total_files, batch_size):
        batch_files = echodata_files[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}")

        try:
            for converted_file in sorted(echodata_zarr_path.glob("*.zarr")):
                ed_future = client.submit(
                    ep.open_converted,
                    converted_raw_path=converted_file,
                    chunks={}
                )
                ed_future_list.append(ed_future)
            ed_list = client.gather(ed_future_list)
            ed_combined = ep.combine_echodata(ed_list)

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
        # Start the flow
        load_and_combine_zarr_stores.serve(
            name='combine-zarr-stores',
            parameters={
                'cruise_id': '',
                'source_container': 'converted',
                'start_datetime': None,
                'end_datetime': None,
                'output_container': '',
                'combined_zarr_name': 'saildrone2023.zarr',
                'chunks_ping_time': 1000,
                'chunks_range_sample': 1000,
                'batch_size': BATCH_SIZE
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception occurred: {str(e)}', exc_info=True)
        sys.exit(1)
