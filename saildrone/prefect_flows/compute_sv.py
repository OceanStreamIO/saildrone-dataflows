import logging
import os
import sys
from pathlib import Path
from typing import List

import echopype as ep
from dotenv import load_dotenv
from prefect import flow, task
from prefect_dask import DaskTaskRunner, get_dask_client
from dask.distributed import Client
from prefect.cache_policies import Inputs

input_cache_policy = Inputs()

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


@task(
    retries=3,
    retry_delay_seconds=1,
    cache_policy=input_cache_policy,
)
def process_file(file_path: Path) -> None:
    with get_dask_client():
        print('Processing file:', file_path)
        echodata = ep.open_raw(file_path, sonar_model='EK60')

        # zarr_path = file_path.with_suffix('.zarr')
        # echodata.to_zarr(zarr_path)
        # sv_dataset = ep.calibrate.compute_Sv(echodata, waveform_mode='CW', encode_mode='power').compute()
        # sv_path = f'{zarr_path.parent}/{zarr_path.stem}_Sv.zarr'
        # sv_dataset.to_zarr(sv_path, mode='w')
        print(f"Processed Sv for {file_path.name}")


@flow(task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS))
def process_raw_data(files: List[Path]) -> None:
    task_futures = []
    for file_path in files:
        future = process_file.submit(file_path)
        task_futures.append(future)

    # Wait for all tasks in the batch to complete
    for future in task_futures:
        future.result()