import logging
import os
import shutil
import sys
import traceback
# TODO: unused imports below \u2014 clean up: dask, get_client, process_converted_file,
# get_files_list, FileSegmentService, SurveyService, ensure_container_exists, list_zarr_files
import dask

from datetime import datetime
from dask.distributed import get_client

from pathlib import Path
from typing import List, Optional, Union
from echopype.clean import remove_background_noise as ep_remove_background_noise
from dotenv import load_dotenv
from dask.distributed import Client
from prefect import flow, task
from prefect.futures import as_completed
from prefect_dask import DaskTaskRunner, get_dask_client
from prefect.cache_policies import Inputs
from saildrone.process import process_converted_file, plot_and_upload_echograms, get_files_list, apply_denoising
from saildrone.store import (FileSegmentService, PostgresDB, SurveyService, open_zarr_store,
                             save_dataset_to_netcdf, ensure_container_exists, save_zarr_store, list_zarr_files)

input_cache_policy = Inputs()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

load_dotenv()

DASK_CLUSTER_ADDRESS = os.getenv('DASK_CLUSTER_ADDRESS')

NETCDF_ROOT_DIR = '/mnt/saildronedata'
DEFAULT_TASK_TIMEOUT = 7_200  # 2 hours
MAX_RUNTIME_SECONDS = 3_300


@task(log_prints=True)
def denoise_zarr(
    zarr_src: str,
    zarr_dest: str,
    container_name: str,
    mask_impulse_noise,
    mask_attenuated_signal,
    mask_transient_noise,
    remove_background_noise,
    apply_seabed_mask: bool,
    chunks=None,
    category: str = None
):

    ds = open_zarr_store(
        zarr_src,
        container_name=container_name,
        chunks=chunks,
        rechunk_after=True,
    )

    print('Opened Zarr dataset:', ds)
    sv_dataset_denoised = apply_denoising(
        ds,
        mask_impulse_noise=mask_impulse_noise,
        mask_attenuated_signal=mask_attenuated_signal,
        mask_transient_noise=mask_transient_noise,
        pulse_length=category,
        remove_background_noise=None
    )

    print("Denoising complete", sv_dataset_denoised)
    print('Running background noise removal...')

    def _mask_one(ch_ds):
        freq = str(int(ch_ds["frequency_nominal"]))
        opts = remove_background_noise.get(freq)

        return ep_remove_background_noise(
            ch_ds,
            ping_num=opts["ping_window"],
            range_sample_num=opts["range_window"],
            SNR_threshold=opts["SNR_threshold"],
            background_noise_max=opts["background_noise_max"],
        )["Sv"]

    if remove_background_noise is not None:
        sv_clean = sv_dataset_denoised.groupby("channel").map(_mask_one)
        sv_clean.name = "Sv"

        # overwrite in-place
        sv_dataset_denoised["Sv"] = sv_clean
        print("Background noise removal complete.")

    save_zarr_store(sv_dataset_denoised, container_name=container_name, zarr_path=zarr_dest)

    print(f"Saved denoised dataset to {zarr_dest}")

    return zarr_dest


@flow(task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS))
def apply_denoising_flow(
    zarr_path_source: str,
    zarr_path_output: str,
    container_name: str,
    mask_impulse_noise=None,
    mask_attenuated_signal=None,
    mask_transient_noise=None,
    remove_background_noise=None,
    apply_seabed_mask: bool = False,
    chunks=None,
    category: str = ''
):
    future = denoise_zarr.submit(
        zarr_src=zarr_path_source,
        zarr_dest=zarr_path_output,
        container_name=container_name,
        mask_impulse_noise=mask_impulse_noise,
        mask_attenuated_signal=mask_attenuated_signal,
        mask_transient_noise=mask_transient_noise,
        remove_background_noise=remove_background_noise,
        apply_seabed_mask=apply_seabed_mask,
        chunks=chunks,
        category=category
    )

    output = future.result()
    print(f"Completed denoising: wrote {output}")

    return output


if __name__ == "__main__":
    client = Client(address=DASK_CLUSTER_ADDRESS)

    try:
        # Start the flow
        apply_denoising_flow.serve(
            name='apply-denoising-flow',
            parameters={
                'zarr_path_source': '',
                'zarr_path_output': '',
                'container_name': '',
                'mask_impulse_noise': None,
                'mask_attenuated_signal': None,
                'mask_transient_noise': None,
                'remove_background_noise': None,
                'apply_seabed_mask': False,
                'chunks': None,
                'category': ''
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception occurred: {str(e)}', exc_info=True)
        sys.exit(1)
