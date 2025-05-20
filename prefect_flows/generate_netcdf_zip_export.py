import logging
import os
import shutil
import sys

from typing import List, Optional, Union
from dotenv import load_dotenv

from prefect import flow, task
from prefect.cache_policies import Inputs
from saildrone.store import (FileSegmentService, PostgresDB, SurveyService, open_zarr_store, generate_container_name,
                             ensure_container_exists, save_zarr_store, zip_and_save_netcdf_files,
                             save_dataset_to_netcdf)

input_cache_policy = Inputs()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

load_dotenv()

NETCDF_ROOT_DIR = '/mnt/saildronedata'
DEFAULT_TASK_TIMEOUT = 7_200  # 2 hours
MAX_RUNTIME_SECONDS = 3_300


@task(log_prints=True)
def zip_netcdf_outputs(nc_file_paths, zip_name, container_name):
    print('Flat paths:', nc_file_paths)
    #zip_and_save_netcdf_files(flat_paths, zip_name, container_name, tmp_dir=NETCDF_ROOT_DIR + '/tmp')
    logging.info(f"Uploaded archive {zip_name} to container {container_name}")


@flow(log_prints=True)
def generate_netcdf_zip_export(output_container: str,
                               file_list: List[str]
                               ):
    total_files = len(file_list)
    logging.info(f"Total files to process: {total_files}")
    print(f"Total files to process: {total_files}")

    future_zip = zip_netcdf_outputs.submit(
        nc_file_paths=file_list,
        zip_name=f"{output_container}.zip",
        container_name=output_container
    )
    future_zip.wait()

    if os.path.exists('/tmp/oceanstream/netcdfdata'):
        shutil.rmtree('/tmp/oceanstream/netcdfdata', ignore_errors=True)

    print("All files have been processed.")


if __name__ == "__main__":
    try:
        generate_netcdf_zip_export.serve(
            name='generate-netcdf-zip',
            parameters={
                'output_container': '',
                'file_list': []
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception occurred: {str(e)}', exc_info=True)
        sys.exit(1)
