import logging
import os
import sys
import traceback

import numpy as np
import xarray as xr

from dotenv import load_dotenv
from prefect import flow, task
from dask.distributed import Client
from prefect_dask import DaskTaskRunner, get_dask_client
from prefect.cache_policies import Inputs
from prefect.artifacts import create_link_artifact, create_markdown_artifact
from prefect.states import Completed

from saildrone.process import apply_corrections_ds
from saildrone.process.plot import plot_noise_mask, plot_sv_data
from saildrone.process.concat import merge_location_data, optimize_zarr_store, concatenate_and_rechunk, \
    cleanup_temp_folders
from saildrone.denoise import get_impulse_noise_mask, get_attenuation_mask, create_multichannel_mask
from saildrone.store import (PostgresDB, SurveyService, FileSegmentService, open_zarr_store,
                             upload_folder_to_blob_storage, save_dataset_to_netcdf,
                             save_zarr_store, generate_container_name, ensure_container_exists,
                             generate_container_access_url)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
input_cache_policy = Inputs()


def concatenate_zarr_files(files, source_container_name, chunks=None, batch_size=10, path_template="/tmp/oceanstream"):
    temp_paths = {
        "short_pulse": [],
        "long_pulse": [],
        "exported_ds": []
    }

    futures = []
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        future = process_batch.submit(batch_files, source_container_name, chunks, i, path_template)
        futures.append(future)

    for future in futures:
        batch_results = future.result()
        print('Batch results:', batch_results)
        # Accumulate paths
        temp_paths["short_pulse"].extend(batch_results["short_pulse"])
        temp_paths["long_pulse"].extend(batch_results["long_pulse"])
        temp_paths["exported_ds"].extend(batch_results["exported_ds"])

    short_pulse_ds = concatenate_and_rechunk(temp_paths["short_pulse"], chunks=chunks) if temp_paths["short_pulse"] else None
    long_pulse_ds = concatenate_and_rechunk(temp_paths["long_pulse"], chunks=chunks) if temp_paths["long_pulse"] else None
    exported_ds = concatenate_and_rechunk(temp_paths["exported_ds"], chunks=chunks) if temp_paths["exported_ds"] else None

    # Cleanup temporary folders
    cleanup_temp_folders(temp_paths["short_pulse"] + temp_paths["long_pulse"] + temp_paths["exported_ds"])

    return short_pulse_ds, long_pulse_ds, exported_ds


@task(
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    cache_policy=input_cache_policy,
    retry_jitter_factor=0.1,
    refresh_cache=True,
    result_storage=None,
    task_run_name="process-{file_name}",
)
def process_single_file(file, file_name, source_container_name, chunks, path_template, file_index):
    """
    Process a single file for Dask Futures: open it, merge location data, save the dataset, and return its path and frequency category.
    """
    location, _, file_id, location_data, file_freqs, file_start_time, file_end_time = file
    print(f"Processing file {location} with frequencies {file_freqs}")

    # Open the Zarr store lazily with Dask
    ds = open_zarr_store(location, container_name=source_container_name, chunks=chunks)

    # Merge location data
    ds = merge_location_data(ds, location_data)

    # Save the dataset to a temporary Zarr store
    category = "short_pulse" if file_freqs == "38000.0,200000.0" else "long_pulse" if file_freqs == "38000.0" else "exported_ds"
    temp_path = f"{path_template}/{category}_file_{file_index}.zarr"

    print('Writing to', temp_path)
    ds.to_zarr(temp_path, mode="w")

    optimize_zarr_store(temp_path)

    return temp_path, category


@task(
    task_run_name="process-batch-{batch_index}",
)
def process_batch(batch_files, source_container_name, chunks, batch_index, path_template):
    """
    Submit individual file processing as futures and return the results.
    """
    futures = []

    for idx, file in enumerate(batch_files):
        future = process_single_file.submit(file, file[1], source_container_name, chunks, path_template, idx)
        futures.append(future)

    results = {
        "short_pulse": [],
        "long_pulse": [],
        "exported_ds": []
    }

    # Collect results as they complete
    for future in futures:
        try:
            temp_path, category = future.result()
            results[category].append(temp_path)
        except Exception as e:
            print(f"Error processing file: {e}")

    return results


def get_files_by_cruise_id(cruise_id, coordinates=None):
    with PostgresDB() as db_connection:
        survey_service = SurveyService(db_connection)
        survey_id = survey_service.get_survey_by_cruise_id(cruise_id)

        if survey_id is None:
            raise ValueError(f'Survey with cruise_id {cruise_id} not found.')

        polygon = f"POLYGON(({', '.join([f'{lon} {lat}' for lon, lat in coordinates])}))"
        file_service = FileSegmentService(db_connection)
        files = file_service.get_files_by_polygon_and_survey(polygon, survey_id)

        if not files:
            raise ValueError("No files found matching the given criteria.")

        return files