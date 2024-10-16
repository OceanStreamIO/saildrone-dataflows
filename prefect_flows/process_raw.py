import logging
import os
import sys
from pathlib import Path
from typing import List

from echopype.calibrate import compute_Sv
from echopype.convert.api import open_raw
from store import ensure_container_exists

from dotenv import load_dotenv
from prefect import flow, task
from prefect_dask import DaskTaskRunner, get_dask_client
from dask.distributed import Client
from prefect.cache_policies import Inputs

from store import save_zarr_store

input_cache_policy = Inputs()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

load_dotenv()
DASK_CLUSTER_ADDRESS = os.getenv('DASK_CLUSTER_ADDRESS')
client = Client(address=DASK_CLUSTER_ADDRESS)


@task(
    retries=3,
    retry_delay_seconds=1,
    cache_policy=input_cache_policy,
)
def convert_file(file_path: Path, container_name, survey_id=None, sonar_model='EK80') -> None:
    with get_dask_client():
        print('Processing file:', file_path)
        echodata = open_raw(file_path, sonar_model=sonar_model)

        if survey_id:
            zarr_path = f"{survey_id}/{file_path.stem}.zarr"
        else:
            zarr_path = f"{file_path.stem}.zarr"

        save_zarr_store(echodata, container_name=container_name, zarr_path=zarr_path)

        # sv_dataset = ep.calibrate.compute_Sv(echodata, waveform_mode='CW', encode_mode='power').compute()
        # sv_path = f'{zarr_path.parent}/{zarr_path.stem}_Sv.zarr'
        # sv_dataset.to_zarr(sv_path, mode='w')
        print(f"Processed Sv for {file_path.name}")


@flow(task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS))
def convert_raw_data(files: List[Path], container_name, survey_id) -> None:
    ensure_container_exists(container_name)
    task_futures = []
    for file_path in files:
        future = convert_file.submit(file_path, container_name, survey_id)
        task_futures.append(future)

    # Wait for all tasks in the batch to complete
    for future in task_futures:
        future.result()