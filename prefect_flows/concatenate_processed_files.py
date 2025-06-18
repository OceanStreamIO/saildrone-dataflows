import logging
import os
import shutil
import sys
import traceback
from collections import defaultdict

from datetime import datetime, timedelta
from typing import List, Optional, Union
from dotenv import load_dotenv

from dask.distributed import Client

from prefect import flow, task
from prefect.cache_policies import Inputs
from prefect.futures import as_completed, PrefectFuture

from saildrone.process import plot_and_upload_echograms, apply_denoising
from saildrone.process.concat import concatenate_and_rechunk
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

DASK_CLUSTER_ADDRESS = os.getenv('DASK_CLUSTER_ADDRESS')
CATEGORY_CONFIG = {
    "short_pulse": {
        "freq_key": "38000.0,200000.0",
        "zarr_name": "short_pulse{denoised}.zarr",
        "nc_name": "short_pulse{denoised}.nc",
        "file_base": "short_pulse{denoised}",
    },
    "long_pulse": {
        "freq_key": "38000.0",
        "zarr_name": "long_pulse{denoised}.zarr",
        "nc_name": "long_pulse{denoised}.nc",
        "file_base": "long_pulse{denoised}",
    },
    "exported_ds": {
        "freq_key": None,  # catch-all
        "zarr_name": "{batch_key}{denoised}.zarr",
        "nc_name": "{batch_key}{denoised}.nc",
        "file_base": "{batch_key}{denoised}"
    }
}

CHUNKS = {"ping_time": 1000, "depth": -1}
NETCDF_ROOT_DIR = os.getenv('NETCDF_ROOT_DIR', '/tmp/oceanstream/netcdfdata')
DEFAULT_TASK_TIMEOUT = 7_200  # 2 hours
MAX_RUNTIME_SECONDS = 3_300


@task(
    log_prints=True,
    task_run_name="compute_batch_nasc--{batch_key}"
)
def compute_batch_nasc(batch_results, batch_key, cruise_id, container_name, compute_nasc_options,
                       plot_echograms=False, save_to_netcdf=False, colormap='ocean_r', chunks=None):
    results = {}

    return results


@task(
    log_prints=True,
    task_run_name="compute_batch_mvbs--{batch_key}"
)
def compute_batch_mvbs(batch_results, batch_key, cruise_id, container_name, compute_mvbs_options,
                       plot_echograms=False, save_to_netcdf=False, colormap='ocean_r', chunks=None):
    results = {}

    return results


@task(
    log_prints=True,
    task_run_name="concatenate_batch_files--{batch_key}"
)
def concatenate_batch_files(batch_key, cruise_id, files, container_name, plot_echograms, save_to_netcdf,
                            colormap, **kwargs):
    """Run NASC, MVBS, … for one calendar batch."""
    # 1) bucket files by category
    batch_results = {cat: [] for cat in CATEGORY_CONFIG}
    chunks = kwargs.get('chunks', None)

    for file_info in files:
        freqs = file_info["file_freqs"]
        for category, cfg in CATEGORY_CONFIG.items():
            if cfg["freq_key"] is None or freqs == cfg["freq_key"]:
                path = f"{cruise_id}/{file_info['file_name']}/{file_info['file_name']}.zarr"
                batch_results[category].append(path)
                break

    def _process_category(cat: str):
        paths = batch_results[cat]
        if not paths:
            return

        section = CATEGORY_CONFIG[cat]
        print('Concatenating files for category:', cat, f'with {len(paths)} paths:')
        ds = concatenate_and_rechunk(paths, container_name=container_name, chunks=chunks)
        print(f"Finished concatenating {cat} dataset:", ds)

        # save Zarr
        zarr_path = f"{batch_key}/{section['zarr_name']}".format(batch_key=batch_key, denoised='')
        save_zarr_store(ds, container_name=container_name, zarr_path=zarr_path)

        # optional echograms
        if plot_echograms:
            plot_and_upload_echograms(
                ds,
                file_base_name=section["file_base"].format(batch_key=batch_key, denoised=''),
                save_to_blobstorage=True,
                depth_var="depth",
                upload_path=batch_key,
                cmap=colormap,
                container_name=container_name,
            )

        ##########################################################
        print('5) Applying denoising')
        try:
            sv_dataset_denoised = apply_denoising(ds, chunks_denoising=chunks, **kwargs)
        except Exception as e:
            print(f"Error applying denoising to {zarr_path}: {str(e)}")
            traceback.print_exc()
            sv_dataset_denoised = None

        print('5) Denoising applied', sv_dataset_denoised)

        if sv_dataset_denoised is not None:
            zarr_path_denoised = f"{batch_key}/{section['zarr_name']}".format(batch_key=batch_key, denoised='--denoised')
            save_zarr_store(sv_dataset_denoised, container_name=container_name, zarr_path=zarr_path_denoised)
            print('6) Saved denoised dataset to Zarr store:', zarr_path)

            if plot_echograms:
                plot_and_upload_echograms(
                    ds,
                    file_base_name=section["file_base"].format(batch_key=batch_key, denoised='--denoised'),
                    save_to_blobstorage=True,
                    depth_var="depth",
                    upload_path=batch_key,
                    cmap=colormap,
                    container_name=container_name,
                )

            if save_to_netcdf:
                nc_file_path_denoised = f"{batch_key}/{section['nc_name']}".format(batch_key=batch_key, denoised='--denoised')
                save_dataset_to_netcdf(
                    sv_dataset_denoised,
                    container_name=container_name,
                    ds_path=nc_file_path_denoised,
                    base_local_temp_path=NETCDF_ROOT_DIR,
                    is_temp_dir=False,
                )

                print('7) Saved denoised dataset to NetCDF:', nc_file_path_denoised)
        ##########################################################

        # optional NetCDF
        if save_to_netcdf:
            nc_path = f"{batch_key}/{section['nc_name']}".format(batch_key=batch_key, denoised='')
            save_dataset_to_netcdf(
                ds,
                container_name=container_name,
                ds_path=nc_path,
                base_local_temp_path=NETCDF_ROOT_DIR,
                is_temp_dir=False,
            )

    # 2) run through each category
    for category in CATEGORY_CONFIG:
        _process_category(category)

    logging.info(
        f"Running batch aggregation for key: {batch_key}, with {len(files)} files."
    )
    return batch_results


@flow(log_prints=True)
def concatenate_processed_files(files_list=None,
                                days_to_combine=1,
                                cruise_id='',
                                container_name='',
                                plot_echograms=False,
                                save_to_netcdf=False,
                                colormap='ocean_r',
                                compute_nasc_options=None,
                                compute_mvbs_options=None,
                                mask_impulse_noise=None,
                                mask_attenuated_signal=None,
                                mask_transient_noise=None,
                                remove_background_noise=None,
                                apply_seabed_mask=None,
                                chunks=None
                                ):
    by_batch = defaultdict(list)
    in_flight = []
    side_tasks = []

    for source_path, file_record in files_list:
        ts = datetime.fromisoformat(file_record["file_start_time"])
        key = _batch_key(ts, days_to_combine)
        by_batch[key].append(file_record)

    batch_size = 4
    batches = by_batch.items()
    print(f"Total batches to process: {len(batches)}")

    for key, files in batches:
        print('Processing batch:', key, 'with', len(files), 'files.')
        future = concatenate_batch_files.submit(key, cruise_id, files, container_name, plot_echograms,
                                                save_to_netcdf, colormap,
                                                mask_impulse_noise=mask_impulse_noise,
                                                mask_attenuated_signal=mask_attenuated_signal,
                                                mask_transient_noise=mask_transient_noise,
                                                remove_background_noise=remove_background_noise,
                                                apply_seabed_mask=apply_seabed_mask,
                                                chunks=chunks)

        in_flight.append(future)

        if compute_nasc_options:
            future_nasc_task = compute_batch_nasc.submit(future, key, cruise_id, container_name,
                                                         compute_nasc_options=compute_nasc_options,
                                                         plot_echograms=plot_echograms,
                                                         save_to_netcdf=save_to_netcdf,
                                                         colormap=colormap,
                                                         chunks=chunks)
            side_tasks.append(future_nasc_task)

        if compute_mvbs_options:
            future_mvbs_task = compute_batch_mvbs.submit(future, key, cruise_id, container_name,
                                                         compute_mvbs_options=compute_mvbs_options,
                                                         plot_echograms=plot_echograms,
                                                         save_to_netcdf=save_to_netcdf,
                                                         colormap=colormap,
                                                         chunks=chunks)
            side_tasks.append(future_mvbs_task)

        # Throttle when max concurrent tasks reached
        if len(in_flight) >= batch_size:
            finished = next(as_completed(in_flight))
            in_flight.remove(finished)

    # all_futures = agg_in_flight + agg_side_tasks
    for remaining in in_flight:
        remaining.result()

    if os.path.exists(NETCDF_ROOT_DIR):
        shutil.rmtree(NETCDF_ROOT_DIR, ignore_errors=True)

    print("All files have been processed.")


def _batch_key(ts: datetime, width_days: int) -> str:
    """
    Anchor `ts` to the start of its `width`-day window and return a
    filename-safe key.
      width == 1  →  '2023-08-08'
      width >  1  →  '2023-08-08_to_2023-08-10'   (inclusive range)
    """
    anchor = datetime(ts.year, ts.month, ts.day)  # midnight of that day

    if width_days == 1:
        return f"{anchor:%Y-%m-%d}"

    anchor -= timedelta(days=(anchor - datetime.min).days % width_days)
    end = anchor + timedelta(days=width_days - 1)

    return f"{anchor:%Y-%m-%d}_to_{end:%Y-%m-%d}"


if __name__ == "__main__":
    try:
        concatenate_processed_files.serve(
            name='concatenate_processed_files',
            parameters={
                'files_list': [],
                'days_to_combine': 1,
                'cruise_id': '',
                'container_name': '',
                'plot_echograms': False,
                'save_to_netcdf': False,
                'colormap': 'ocean_r',
                'compute_nasc_options': None,
                'compute_mvbs_options': None,
                'mask_impulse_noise': None,
                'mask_attenuated_signal': None,
                'mask_transient_noise': None,
                'remove_background_noise': None,
                'apply_seabed_mask': None,
                'chunks': None,
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception occurred: {str(e)}', exc_info=True)
        sys.exit(1)
