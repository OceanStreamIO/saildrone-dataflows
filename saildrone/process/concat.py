import os
import shutil

import numpy as np
import pandas as pd
import xarray as xr
import zarr


def merge_location_data(dataset: xr.Dataset, location_data):
    """Merge location data into the dataset while ensuring it's a variable, not just an attribute."""
    # Convert location_data to a Pandas DataFrame
    location_df = pd.DataFrame(location_data)

    # Convert timestamp strings to datetime objects
    location_df['dt'] = pd.to_datetime(location_df['dt'])

    # Determine which time dimension to use
    time_dim = "ping_time" if "ping_time" in dataset.dims else "time" if "time" in dataset.dims else None
    if not time_dim:
        return dataset  # Return without merging if no time dimension exists

    # Interpolate location data to match dataset time
    target_times = dataset[time_dim].values

    lat_interp = np.interp(
        np.array(pd.to_datetime(target_times).astype(int)),
        np.array(location_df["dt"].astype(int)),
        location_df["lat"]
    )

    lon_interp = np.interp(
        np.array(pd.to_datetime(target_times).astype(int)),
        np.array(location_df["dt"].astype(int)),
        location_df["lon"]
    )

    speed_interp = np.interp(
        np.array(pd.to_datetime(target_times).astype(int)),
        np.array(location_df["dt"].astype(int)),
        location_df["knt"]
    )

    # Ensure latitude is stored as a variable, not just an attribute
    dataset['latitude'] = xr.DataArray(lat_interp, dims=time_dim, coords={time_dim: target_times})
    dataset['longitude'] = xr.DataArray(lon_interp, dims=time_dim, coords={time_dim: target_times})
    dataset['speed_knots'] = xr.DataArray(speed_interp, dims=time_dim, coords={time_dim: target_times})

    # Debugging: Print dataset variables after merging

    return dataset


def xmerge_location_data(dataset: xr.Dataset, location_data):
    """Merge location data into the dataset while ensuring time alignment using interpolation."""
    # Convert location_data to a Pandas DataFrame
    location_df = pd.DataFrame(location_data)

    # Convert timestamp strings to datetime objects
    location_df['dt'] = pd.to_datetime(location_df['dt'])

    if "ping_time" in dataset.dims:
        # Interpolate location data to match 'ping_time'
        target_times = dataset["ping_time"].values

        lat_interp = np.interp(
            np.array(pd.to_datetime(target_times).astype(int)),
            np.array(location_df["dt"].astype(int)),
            location_df["lat"]
        )

        lon_interp = np.interp(
            np.array(pd.to_datetime(target_times).astype(int)),
            np.array(location_df["dt"].astype(int)),
            location_df["lon"]
        )

        speed_interp = np.interp(
            np.array(pd.to_datetime(target_times).astype(int)),
            np.array(location_df["dt"].astype(int)),
            location_df["knt"]
        )

        dataset['latitude'] = xr.DataArray(lat_interp, dims="ping_time", coords={"ping_time": target_times})
        dataset['longitude'] = xr.DataArray(lon_interp, dims="ping_time", coords={"ping_time": target_times})
        dataset['speed_knots'] = xr.DataArray(speed_interp, dims="ping_time", coords={"ping_time": target_times})

    else:
        # Default behavior: Assign location data based on its own timestamps
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
        concatenated_ds = xr.concat(sorted_datasets, dim=dim)

        if 'frequency_nominal' in concatenated_ds:
            freq = concatenated_ds.frequency_nominal

            if freq.ndim == 2:
                # Ensure unique across files, collapse to 1D if possible
                unique_rows = np.unique(freq.values, axis=0)
                if unique_rows.shape[0] == 1:
                    frequency_1d = unique_rows[0]
                else:
                    print("Multiple frequency_nominal rows found; using first row.")
                    frequency_1d = freq.values[0]
            else:
                frequency_1d = freq.values

            frequency_1d = np.asarray(frequency_1d).astype(np.float64)
            concatenated_ds = concatenated_ds.assign_coords({
                "frequency": ("channel", frequency_1d)
            })
        else:
            print('frequency_nominal not in concatenated_ds')

        return concatenated_ds.chunk(chunks)

    return None




