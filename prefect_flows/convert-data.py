import logging
import os
import sys
from echopype.convert.api import open_raw
from echopype.calibrate import compute_Sv
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from prefect import flow, task
from dask.distributed import Client
from prefect_dask import DaskTaskRunner, get_dask_client
from prefect.cache_policies import Inputs

from saildrone.process_data import apply_calibration
from saildrone.store import save_zarr_store, ensure_container_exists

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
CONVERTED_CONTAINER_NAME = os.getenv('CONVERTED_CONTAINER_NAME')
PROCESSED_CONTAINER_NAME = os.getenv('PROCESSED_CONTAINER_NAME')
CALIBRATION_FILE = os.getenv('CALIBRATION_FILE')
client = Client(address=DASK_CLUSTER_ADDRESS)
BATCH_SIZE = 6

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
def convert_file(file_path: Path, survey_id=None, sonar_model='EK80') -> None:
    print('Processing file:', file_path)

    with get_dask_client():
        echodata = open_raw(file_path, sonar_model=sonar_model)
        echodata = apply_calibration(echodata, CALIBRATION_FILE)

        if survey_id:
            zarr_path = f"{survey_id}/{file_path.stem}.zarr"
        else:
            zarr_path = f"{file_path.stem}.zarr"

        save_zarr_store(echodata, container_name=CONVERTED_CONTAINER_NAME, zarr_path=zarr_path)

        sv_dataset = compute_Sv(echodata, waveform_mode='CW', encode_mode='complex').compute()
        save_zarr_store(sv_dataset, container_name=PROCESSED_CONTAINER_NAME, zarr_path=f"{survey_id}/{file_path.stem}/{file_path.stem}_Sv.zarr")
        print(f"Processed Sv for {file_path.name}")


def convert_raw_data(files: List[Path], survey_id) -> None:
    task_futures = []
    print('Processing files:', files)
    for file_path in files:
        future = convert_file.submit(file_path, survey_id)
        task_futures.append(future)

    # Wait for all tasks in the batch to complete
    for future in task_futures:
        future.result()


@flow(task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS))
def load_and_convert_files_to_zarr(source_directory, survey_id, batch_size) -> None:
    mounted_folder = Path(source_directory)
    raw_files = sorted(mounted_folder.glob("*.raw"))
    total_files = len(raw_files)
    print(f"Total files to process: {total_files}")

    # Process files in batches
    for i in range(0, total_files, batch_size):
        batch_files = raw_files[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}")
        convert_raw_data(batch_files, survey_id)

    logging.info("All batches have been processed.")


if __name__ == "__main__":
    ensure_container_exists(CONVERTED_CONTAINER_NAME)
    ensure_container_exists(PROCESSED_CONTAINER_NAME)
    load_and_convert_files_to_zarr.serve(name="convert-raw-files-to-zarr",
                                         parameters={
                                            'source_directory': RAW_DATA_LOCAL,
                                            'survey_id': '',
                                            'batch_size': BATCH_SIZE
                                        })
