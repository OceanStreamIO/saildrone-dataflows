import logging
import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from prefect import flow, task
from dask.distributed import Client
from prefect_flows import process_batch
from prefect_dask import DaskTaskRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

load_dotenv()
DASK_CLUSTER_ADDRESS = os.getenv('DASK_CLUSTER_ADDRESS')
client = Client(address=DASK_CLUSTER_ADDRESS)

BATCH_SIZE = 10
AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
if not AZURE_STORAGE_CONNECTION_STRING:
    logging.error('AZURE_STORAGE_CONNECTION_STRING environment variable not set.')

    sys.exit(1)


@flow(task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS))
def load_and_process_files_in_batches(directory: Path) -> None:
    # List all .raw files in the directory
    raw_files = sorted(directory.glob("*.raw"))
    total_files = len(raw_files)
    print(f"Total files to process: {total_files}")

    # Process files in batches
    for i in range(0, total_files, BATCH_SIZE):
        batch_files = raw_files[i:i + BATCH_SIZE]
        print(f"Processing batch {i // BATCH_SIZE + 1}")
        process_batch(batch_files)


if __name__ == "__main__":
    data_directory = Path("./raw-data")
    load_and_process_files_in_batches(data_directory)
