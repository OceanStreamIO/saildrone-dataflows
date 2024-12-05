import logging
import os
import sys
import time
import xarray as xr

from pathlib import Path
from typing import List, Optional
import pandas as pd
from dotenv import load_dotenv
from prefect import flow, task
from dask.distributed import Client
from prefect_dask import DaskTaskRunner, get_dask_client
from prefect.cache_policies import Inputs
from prefect.states import Completed

from saildrone.process import convert_file_and_save, plot_sv_data
from saildrone.utils import load_local_files
from saildrone.store import PostgresDB, SurveyService, FileSegmentService, open_zarr_store

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
            print(ds)

            # Merge location data into the dataset
            ds = merge_location_data(ds, location_data)

            # Categorize datasets by file frequency
            if file_freqs == "38000.0,200000.0":
                short_pulse.append(ds)
            elif file_freqs == "38000.0":
                long_pulse.append(ds)

        short_pulse_datasets = [
            ds.rename({"source_filenames": f"source_filenames_{i}"})
            for i, ds in enumerate(short_pulse)
        ]
        short_pulse_ds = xr.merge(short_pulse_datasets) if short_pulse_datasets else xr.Dataset()
        plot_sv_data(short_pulse_ds, f"{cruise_id}--short-pulse", "output/short_pulse")

        long_pulse_datasets = [
            ds.rename({"source_filenames": f"source_filenames_{i}"})
            for i, ds in enumerate(long_pulse)
        ]
        long_pulse_ds = xr.merge(long_pulse_datasets) if long_pulse_datasets else xr.Dataset()
        plot_sv_data(short_pulse_ds, f"{cruise_id}--long-pulse", "output/long_pulse")

        save_datasets_to_netcdf(short_pulse_ds, long_pulse_ds)

        return short_pulse_ds, long_pulse_ds


def save_datasets_to_netcdf(
    short_pulse_ds: xr.Dataset,
    long_pulse_ds: xr.Dataset,
    short_pulse_path: str = "output/short_pulse_data.nc",
    long_pulse_path: str = "output/long_pulse_data.nc",
    compression_level: int = 5
):
    """
    Save short and long pulse Xarray datasets to NetCDF files efficiently with Dask.

    Parameters
    ----------
    short_pulse_ds : xr.Dataset
        The Xarray dataset for short pulse data.
    long_pulse_ds : xr.Dataset
        The Xarray dataset for long pulse data.
    short_pulse_path : str, optional
        The output file path for the short pulse dataset (default is "short_pulse_data.nc").
    long_pulse_path : str, optional
        The output file path for the long pulse dataset (default is "long_pulse_data.nc").
    compression_level : int, optional
        The zlib compression level (default is 5).

    Returns
    -------
    Tuple[str, str]
        Paths of the saved NetCDF files for short and long pulse datasets.
    """

    with get_dask_client() as client:
        # Define compression encoding for short pulse dataset
        short_pulse_encoding = get_variable_encoding(short_pulse_ds, compression_level)
        long_pulse_encoding = get_variable_encoding(long_pulse_ds, compression_level)

        # Save short pulse dataset to NetCDF
        print(f"Saving short pulse dataset to {short_pulse_path}...")
        short_pulse_ds.to_netcdf(
            path=short_pulse_path,
            format="NETCDF4",
            engine="netcdf4",
            encoding=short_pulse_encoding,
            compute=True,
        )
        print(f"Short pulse dataset saved to {short_pulse_path}.")

        # Save long pulse dataset to NetCDF
        print(f"Saving long pulse dataset to {long_pulse_path}...")
        long_pulse_ds.to_netcdf(
            path=long_pulse_path,
            format="NETCDF4",
            engine="netcdf4",
            encoding=long_pulse_encoding,
            compute=True,
        )
        print(f"Long pulse dataset saved to {long_pulse_path}.")

        # Close the Dask client
        client.close()
        print("Dask client closed.")

        return short_pulse_path, long_pulse_path


def get_variable_encoding(ds: xr.Dataset, compression_level):
    """Generate encoding dictionary for dataset variables."""
    encoding = {}
    for var in ds.data_vars:
        if ds[var].dtype.kind in {"U", "S", "O"}:  # String or object types
            # No compression or chunking for unsupported types
            encoding[var] = {}
        else:
            # Apply compression for numeric types
            encoding[var] = {
                "zlib": True,
                "complevel": compression_level,
            }
    return encoding


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
