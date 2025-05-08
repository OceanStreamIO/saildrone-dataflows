import os
import shutil
import numpy as np
import pandas as pd
import xarray as xr
import zarr


def merge_location_data(dataset: xr.Dataset, location_data) -> xr.Dataset:
    # Convert location_data list of dicts to DataFrame
    location_df = pd.DataFrame(location_data)
    location_df['dt'] = pd.to_datetime(location_df['dt'])

    # Create a 1D time-aligned xarray Dataset
    nav_ds = xr.Dataset({
        "latitude": ("time", location_df["lat"].values),
        "longitude": ("time", location_df["lon"].values),
        "speed_knots": ("time", location_df["knt"].values)
    }, coords={"time": location_df["dt"].values})

    if "ping_time" in dataset.coords:
        nav_interp = nav_ds.interp(time=dataset["ping_time"], kwargs={"fill_value": "extrapolate"})
    else:
        nav_interp = nav_ds

    for var in nav_interp.data_vars:
        dataset[var] = nav_interp[var]

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
    if not paths or not chunks:
        return None

    datasets = []
    for path in paths:
        ds = xr.open_zarr(path, chunks=None)

        # Drop range_sample if present (e.g. redundant indexing axis)
        for var in ["range_sample", "source_filenames"]:
            if var in ds:
                ds = ds.drop_vars(var)

        datasets.append(ds)

    datasets.sort(key=lambda ds: ds[dim].min().values)

    # Concatenate along the specified dimension
    concatenated_ds = xr.concat(datasets, dim=dim)

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





