import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from prefect import flow, task
from dask.distributed import Client
from prefect_flows import convert_raw_data
from prefect_dask import DaskTaskRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

load_dotenv()

RAW_DATA_MOUNT = os.getenv('RAW_DATA_MOUNT')
RAW_DATA_LOCAL = os.getenv('RAW_DATA_LOCAL')
DASK_CLUSTER_ADDRESS = os.getenv('DASK_CLUSTER_ADDRESS')
CONVERTED_CONTAINER_NAME = os.getenv('CONVERTED_CONTAINER_NAME')
BATCH_SIZE = 10

AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
if not AZURE_STORAGE_CONNECTION_STRING:
    logging.error('AZURE_STORAGE_CONNECTION_STRING environment variable not set.')
    sys.exit(1)


@flow(task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS))
def load_and_convert_files_to_zarr(source_directory, container_name, survey_id, batch_size) -> None:
    mounted_folder = Path(source_directory)
    raw_files = sorted(mounted_folder.glob("*.raw"))
    total_files = len(raw_files)
    print(f"Total files to process: {total_files}")

    # Process files in batches
    for i in range(0, total_files, batch_size):
        batch_files = raw_files[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}")
        convert_raw_data(batch_files, container_name, survey_id)

    logging.info("All batches have been processed.")


if __name__ == "__main__":
    client = Client(address=DASK_CLUSTER_ADDRESS)

    load_and_convert_files_to_zarr.serve(name="convert-raw-files-to-zarr",
                                         parameters={
                                            'source_directory': RAW_DATA_LOCAL,
                                            'container_name': 'converted',
                                            'survey_id': '',
                                            'batch_size': '10'
                                        })
