import logging
import os
import sys
import time
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

from saildrone.store import (PostgresDB, SurveyService, FileSegmentService, open_zarr_store,
                             upload_folder_to_blob_storage, save_dataset_to_netcdf,
                             save_zarr_store, generate_container_name, ensure_container_exists,
                             generate_container_access_url)

load_dotenv()

input_cache_policy = Inputs()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

DASK_CLUSTER_ADDRESS = os.getenv('DASK_CLUSTER_ADDRESS')
CONVERTED_CONTAINER_NAME = os.getenv('CONVERTED_CONTAINER_NAME')
PROCESSED_CONTAINER_NAME = os.getenv('PROCESSED_CONTAINER_NAME')
CHUNKS = {"ping_time": 1000, "range_sample": -1}
BATCH_SIZE = os.getenv('BATCH_SIZE_FOR_EXPORT', 10)


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


def export_processed_data_task(cruise_id: str, coordinates=None, container_name=None, filters=None, export_format='netcdf'):
    files = get_files_by_cruise_id(cruise_id, coordinates)




@task(
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    retry_jitter_factor=0.1,
    refresh_cache=True,
    task_run_name="plot-echograms"
)
def plot_sv_data_task(sv_path=None, container_name=None, file_name=None):
    output_path = f'/tmp/oceanstream/echograms/{container_name}'
    os.makedirs(output_path, exist_ok=True)

    try:
        if isinstance(sv_path, list):
            for sv_item in sv_path:
                print('Plotting echogram:', sv_item)
                ds_Sv = open_zarr_store(f'{sv_item}.zarr', container_name=container_name, chunks=CHUNKS)
                plot_sv_data(ds_Sv, file_base_name=sv_item, output_path=output_path, depth_var='depth')
        elif sv_path:
            ds_Sv = open_zarr_store(sv_path, container_name=container_name, chunks=CHUNKS)
            plot_sv_data(ds_Sv, file_name, output_path=output_path, depth_var='depth')

        print(f"Uploading echograms to blob storage: {container_name}")
        upload_folder_to_blob_storage(output_path, container_name, 'echograms')
    except Exception as e:
        print(f'Error plotting echograms: {e}')
        traceback.print_exc()



@flow(task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS))
def export_processed(cruise_id, coordinates=None, filters=None, batch_size=10, export_format='zarr', depth_offset=None, container_name=None):
    if not coordinates:
        raise ValueError("Coordinates are required for spatial queries.")

    files = get_files_by_cruise_id(cruise_id, coordinates)
    sv_dataset_list = []
    if container_name is None:
        container_name = generate_container_name(cruise_id)

    ensure_container_exists(container_name, public_access='container')

    short_pulse_ds = open_zarr_store('short_pulse_data.zarr', container_name=container_name, chunks=CHUNKS)
    sv_data, sv_denoised = process_sv_dataset(short_pulse_ds, container_name, filters, 'short_pulse', depth_offset)
    if sv_data:
        sv_dataset_list.append("short_pulse_data")

    short_pulse_ds, long_pulse_ds, exported_ds = concatenate_zarr_files(
        files,
        source_container_name=PROCESSED_CONTAINER_NAME,
        batch_size=batch_size,
        chunks=CHUNKS)

    # if export_format == 'netcdf':

    if short_pulse_ds:
        pass
      if long_pulse_ds:
        sv_data, sv_denoised = process_sv_dataset(long_pulse_ds, container_name, filters, 'long_pulse', depth_offset)
        if sv_data:
            sv_dataset_list.append("long_pulse_data")

    if long_pulse_ds:
        if depth_offset is not None:
            long_pulse_ds = apply_corrections_ds(long_pulse_ds, depth_offset=depth_offset)

        save_dataset_to_netcdf(long_pulse_ds, container_name=container_name, ds_path="long_pulse_data.nc")
        save_zarr_store(long_pulse_ds, container_name=container_name, zarr_path="long_pulse_data.zarr")
        sv_dataset_list.append("long_pulse_data")

    if exported_ds:
        if depth_offset is not None:
            exported_ds = apply_corrections_ds(exported_ds, depth_offset=depth_offset)
        save_dataset_to_netcdf(exported_ds, container_name=container_name, ds_path="exported_data.nc")
        save_zarr_store(exported_ds, container_name=container_name, zarr_path="exported_data.zarr")
        sv_dataset_list.append("exported_data")

    access_link = generate_container_access_url(container_name)
    create_link_artifact(
        key=f"{container_name}-link",
        link=access_link,
        link_text="Export link",
        description="Link to download the exported data."
    )

    plot_sv_data_task(sv_dataset_list, container_name=container_name)
    # future = plot_sv_data_task.submit(sv_dataset_list, container_name=container_name)
    # future.wait()


def process_sv_dataset(ds, container_name, filters, ds_name, depth_offset=None):
    from echopype.mask import apply_mask

    os.makedirs(f"/tmp/oceanstream/echograms/{container_name}", exist_ok=True)
    corrected_ds = ds
    corrected_ds_denoised = None

    # if depth_offset is not None:
    #     corrected_ds = apply_corrections_ds(ds, depth_offset=depth_offset)

    # save_dataset_to_netcdf(corrected_ds, container_name=container_name, ds_path=f"{ds_name}_data.nc")
    # save_zarr_store(corrected_ds, container_name=container_name, zarr_path=f"{ds_name}_data.zarr")

    return corrected_ds, corrected_ds_denoised


if __name__ == "__main__":
    client = Client(address=DASK_CLUSTER_ADDRESS)
    print('Dask client connected to ', DASK_CLUSTER_ADDRESS, client)

    try:
        export_processed.serve(
            name='export-processed-data',
            parameters={
                'cruise_id': '',
                'coordinates': [],
                'filters': {},
                'batch_size': BATCH_SIZE,
                'depth_offset': None,
                'export_format': 'zarr'
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception occurred: {str(e)}', exc_info=True)
        sys.exit(1)
