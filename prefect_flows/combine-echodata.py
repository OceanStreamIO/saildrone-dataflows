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
from prefect.states import Completed
from echopype import open_converted, combine_echodata as ep_combine_echodata

from saildrone.store import save_zarr_store, ensure_container_exists
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


@flow(task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS))
def load_and_combine_zarr_stores(source_directory: str,
                                 map_to_directory: str,
                                 output_zarr_path: str,
                                 container_name: str,
                                 survey_id: str,
                                 combined_zarr_name: str,
                                 description: Optional[str],
                                 batch_size: int) -> None:
    """
    Load raw files from the source directory, insert/update survey record, and convert them to Zarr format.
    """

    echodata_files = load_local_files(source_directory, map_to_directory, '*.zarr')
    # combine_echodata(echodata_files, Path(output_zarr_path), combined_zarr_name)
    combined_zarr_path = Path(output_zarr_path)

    with get_dask_client() as client:
        ed_future_list = []
        for converted_file in echodata_files:
            ed_future = client.submit(
                process_converted_file,
                converted_raw_path=converted_file,
                chunks={}
            )
            ed_future_list.append(ed_future)

        ed_list = client.gather(ed_future_list)
        ed_combined = ep_combine_echodata(ed_list)

        # Save the combined EchoData object to a new Zarr store
        # The appending operation only happens when relevant data needs to be save to disk
        if container_name != '':
            save_zarr_store(ed_combined, combined_zarr_name, survey_id=survey_id,
                                         container_name=container_name)
        else:
            ed_combined.to_zarr(
                combined_zarr_path / combined_zarr_name,
                overwrite=True,
                compute=True,
            )


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
                'survey_id': '',
                'output_container': '',
                'combined_zarr_name': 'saildrone2023.zarr',
                'chunks_ping_time': 1000,
                'chunks_range_sample': 1000,
                'description': '',
                'batch_size': BATCH_SIZE
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception occurred: {str(e)}', exc_info=True)
        sys.exit(1)
