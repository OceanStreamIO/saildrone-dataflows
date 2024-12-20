import os
import shutil

import numpy as np
import pandas as pd
import xarray as xr
import zarr


def merge_location_data(dataset: xr.Dataset, location_data):
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


def save_temp_zarr(ds, path_template, batch_index):
    """
    Save an intermediate batch to a temporary Zarr store.
    """
    if ds:
        temp_path = f"{path_template}_batch_{batch_index}.zarr"
        ds.to_zarr(temp_path, mode="w")
        optimize_zarr_store(temp_path)  # Optimize chunk sizes and consolidate metadata
        return temp_path

    return None


def optimize_zarr_store(zarr_path):
    """
    Optimize the Zarr store by consolidating metadata.
    """
    zarr.consolidate_metadata(zarr_path)


def cleanup_temp_folders(temp_paths):
    """
    Delete all temporary folders used for intermediate results.
    """
    for path in temp_paths:
        if os.path.exists(path):
            shutil.rmtree(path)


def rechunk_datasets(datasets, chunks):
    """
    Ensure all datasets have consistent chunks before concatenation.
    """
    return [ds.chunk(chunks) for ds in datasets]


def concatenate_and_rechunk(paths, dim="ping_time", chunks=None):
    """
    Concatenate datasets from Zarr paths along the given dimension and rechunk.
    """
    if paths and chunks:
        datasets = [xr.open_zarr(path, chunks=chunks) for path in paths]
        sorted_datasets = sorted(datasets, key=lambda ds: ds[dim].min().values)

        sorted_datasets = [
            ds.rename({"source_filenames": f"source_filenames_{i}"})
            for i, ds in enumerate(sorted_datasets)
        ]

        # Concatenate along the specified dimension
        concatenated_ds = xr.merge(sorted_datasets)

        # if 'frequency_nominal' in concatenated_ds:
        #     frequency_1d = concatenated_ds.frequency_nominal.values[0, :]
        #     concatenated_ds = concatenated_ds.assign_coords(frequency=('channel', frequency_1d))
        # else:
        #     print('frequency_nominal not in concatenated_ds')

        return concatenated_ds.chunk(chunks)

    return None




