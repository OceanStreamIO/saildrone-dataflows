import logging
import os
import sys
import time
import xarray as xr

import pandas as pd
from dotenv import load_dotenv
from prefect import flow, task
from dask.distributed import Client
from prefect_dask import DaskTaskRunner, get_dask_client
from prefect.cache_policies import Inputs
from prefect.artifacts import create_link_artifact
from prefect.states import Completed

from saildrone.process import convert_file_and_save, plot_sv_data
from saildrone.store import (PostgresDB, SurveyService, FileSegmentService, open_zarr_store,
                             upload_folder_to_blob_storage,
                             save_datasets_to_netcdf, generate_container_name, ensure_container_exists,
                             generate_container_access_url, create_blob_service_client)


input_cache_policy = Inputs()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

load_dotenv()

DASK_CLUSTER_ADDRESS = os.getenv('DASK_CLUSTER_ADDRESS')
CONVERTED_CONTAINER_NAME = os.getenv('CONVERTED_CONTAINER_NAME')
PROCESSED_CONTAINER_NAME = os.getenv('PROCESSED_CONTAINER_NAME')
CHUNKS = {"ping_time": 500, "range_sample": -1}


@flow(task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS))
def export_processed(cruise_id='', coordinates=None):
    if not coordinates:
        raise ValueError("Coordinates are required for spatial queries.")

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

        short_pulse = []
        long_pulse = []

        for file in files:
            location, file_name, file_id, location_data, file_freqs, file_start_time, file_end_time = file
            zarr_path = location

            # Load the zarr store as an xarray dataset
            ds = open_zarr_store(zarr_path, container_name=PROCESSED_CONTAINER_NAME, chunks=CHUNKS)

            # Merge location data into the dataset
            ds = merge_location_data(ds, location_data)

            # Categorize datasets by file frequency
            if file_freqs == "38000.0,200000.0":
                short_pulse.append(ds)
            elif file_freqs == "38000.0":
                long_pulse.append(ds)

        container_name = generate_container_name(cruise_id)
        ensure_container_exists(container_name)

        output_path = f'/tmp/echograms/{container_name}'
        os.makedirs(output_path, exist_ok=True)

        short_pulse_datasets = [
            ds.rename({"source_filenames": f"source_filenames_{i}"})
            for i, ds in enumerate(short_pulse)
        ]
        short_pulse_ds = xr.merge(short_pulse_datasets) if short_pulse_datasets else xr.Dataset()

        plot_sv_data(short_pulse_ds, f"{cruise_id}--short-pulse", output_path=output_path)

        long_pulse_datasets = [
            ds.rename({"source_filenames": f"source_filenames_{i}"})
            for i, ds in enumerate(long_pulse)
        ]
        long_pulse_ds = xr.merge(long_pulse_datasets) if long_pulse_datasets else xr.Dataset()
        plot_sv_data(short_pulse_ds, f"{cruise_id}--long-pulse", output_path=output_path)

        upload_folder_to_blob_storage(output_path, container_name, 'echograms')
        save_datasets_to_netcdf(short_pulse_ds, long_pulse_ds, container_name)

        access_link = generate_container_access_url(container_name)
        create_link_artifact(
            key=f"{container_name}-link",
            link=access_link,
            link_text="Export link",
            description="Link to download the exported data."
        )

        return short_pulse_ds, long_pulse_ds


def merge_location_data(dataset: xr.Dataset, location_data) -> xr.Dataset:
    """
    Merge location data into the xarray dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        The xarray dataset to update.
    location_data : list
        A list of dictionaries containing location data.

    Returns
    -------
    xr.Dataset
        Updated dataset with location data added.
    """
    # Convert location_data to a Pandas DataFrame
    location_df = pd.DataFrame(location_data)

    # Convert timestamp strings to datetime objects
    location_df['dt'] = pd.to_datetime(location_df['dt'])

    # Create xarray variables from the location data
    dataset['latitude'] = xr.DataArray(location_df['lat'].values, dims='time',
                                       coords={'time': location_df['dt'].values})
    dataset['longitude'] = xr.DataArray(location_df['lon'].values, dims='time',
                                        coords={'time': location_df['dt'].values})
    dataset['speed_knots'] = xr.DataArray(location_df['knt'].values, dims='time',
                                          coords={'time': location_df['dt'].values})

    return dataset


if __name__ == "__main__":
    try:
        client = Client(address=DASK_CLUSTER_ADDRESS)
        export_processed.serve(
            name='export-processed-data',
            parameters={
                'cruise_id': '',
                'coordinates': []
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception occurred: {str(e)}', exc_info=True)
        sys.exit(1)
