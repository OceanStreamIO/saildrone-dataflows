import logging
import os
import sys
import time
from pathlib import Path
from typing import List

# from echopype.calibrate import compute_Sv
# from echopype.convert.api import open_raw
from saildrone.store import ensure_container_exists, save_zarr_store

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

