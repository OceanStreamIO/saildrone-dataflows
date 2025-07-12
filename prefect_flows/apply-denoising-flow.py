import logging
import os
import shutil
import sys
import traceback
import dask

from datetime import datetime
from dask.distributed import get_client

from pathlib import Path
from typing import List, Optional, Union
from dotenv import load_dotenv
from dask.distributed import Client
from prefect import flow, task
from prefect.futures import as_completed
from prefect_dask import DaskTaskRunner, get_dask_client
from prefect.cache_policies import Inputs
from saildrone.process import process_converted_file, plot_and_upload_echograms, get_files_list, apply_denoising
from saildrone.process.plot import plot_and_upload_masks
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
def denoise_task(
    sv_dataset,
    mask_impulse_noise,
    mask_attenuated_signal,
    mask_transient_noise,
    remove_background_noise,
    drop_pings: bool,
):
    """
    Run the de/denoising pipeline entirely on the Dask cluster.
    """
    # 1) Build masks & apply
    sv_den, mask_dict = apply_denoising(
        sv_dataset,
        mask_impulse_noise=mask_impulse_noise,
        mask_attenuated_signal=mask_attenuated_signal,
        mask_transient_noise=mask_transient_noise,
        remove_background_noise=remove_background_noise,
        drop_pings=drop_pings,
    )

    # 2) Materialize while you're still in the DaskTaskRunner context
    #    so nothing lazy escapes to the driver.
    sv_den = sv_den.compute()
    mask_dict = {
        k: v.compute() if hasattr(v, "compute") else v
        for k, v in mask_dict.items()
    }

    return sv_den, mask_dict


@flow(task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS))
def apply_denoising_flow(
    zarr_path_source: str,
    zarr_path_output: str,
    container_name: str,
    file_base_name: str,
    upload_path: str,
    plot_echograms: bool = False,
    title_template: str = '',
    colormap: str = 'ocean_r',
    mask_impulse_noise=None,
    mask_attenuated_signal=None,
    mask_transient_noise=None,
    remove_background_noise=None,
    apply_seabed_mask: bool = False,
    chunks=None
):
    sv_dataset = open_zarr_store(zarr_path_source,
                                 container_name=container_name, chunks=chunks, rechunk_after=True)

    sv_dataset_denoised, mask_dict = denoise_task.submit(
        sv_dataset,
        mask_impulse_noise,
        mask_attenuated_signal,
        mask_transient_noise,
        remove_background_noise,
        False
    ).result()

    print("Denoising complete", sv_dataset_denoised)

    # sv_dataset_denoised, mask_dict = apply_denoising(sv_dataset,
    #                                                  mask_impulse_noise=mask_impulse_noise,
    #                                                  mask_attenuated_signal=mask_attenuated_signal,
    #                                                  mask_transient_noise=mask_transient_noise,
    #                                                  remove_background_noise=remove_background_noise,
    #                                                  drop_pings=False)

    save_zarr_store(sv_dataset_denoised, container_name=container_name, zarr_path=zarr_path_output)
    print(f"Saved denoised dataset to {zarr_path_output}")

    if plot_echograms:
        plot_and_upload_echograms(
            sv_dataset_denoised,
            file_base_name=file_base_name,
            save_to_blobstorage=True,
            upload_path=upload_path,
            cmap=colormap,
            container_name=container_name,
            title_template=title_template,
        )

        plot_and_upload_masks(
            mask_dict,
            sv_dataset_denoised,
            file_base_name=file_base_name + '--mask',
            upload_path=upload_path,
            container_name=container_name,
        )


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
                'file_base_name': '',
                'upload_path': '',
                'plot_echograms': False,
                'title_template': '',
                'colormap': 'ocean_r',
                'mask_impulse_noise': None,
                'mask_attenuated_signal': None,
                'mask_transient_noise': None,
                'remove_background_noise': None,
                'apply_seabed_mask': False,
                'chunks': None,
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception occurred: {str(e)}', exc_info=True)
        sys.exit(1)
