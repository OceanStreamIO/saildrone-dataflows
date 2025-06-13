import os
import shutil
import pandas as pd
import xarray as xr
import zarr

from saildrone.store import open_zarr_store
from saildrone.store.utils import fix_chunking


def merge_location_data(ds: xr.Dataset, location_data: list[dict]) -> xr.Dataset:
    df = pd.DataFrame(location_data)
    df["dt"] = (
        pd.to_datetime(df["dt"], utc=True, errors="coerce")  # parse to UTC
        .dt.tz_localize(None)
    )
    df = df.set_index("dt").sort_index()
    nav = df[["lat", "lon", "knt"]].to_xarray()
    nav = nav.rename({
        "dt": "ping_time",
        "lat": "latitude",
        "lon": "longitude",
        "knt": "speed_knots",
    })

    if "ping_time" in ds.coords:
        nav = nav.interp(
            ping_time=ds["ping_time"],
            method="nearest",
            kwargs={"fill_value": "extrapolate"},
        )

    for v in ["latitude", "longitude", "speed_knots"]:
        if v in ds.data_vars:
            ds = ds.drop_vars(v)
        if v in ds.coords:
            ds = ds.reset_coords(v, drop=True)

    if "time" in ds:
        ds = ds.drop_vars("time")

    if "time" in ds.coords:
        ds = ds.reset_coords("time", drop=True)

    merged = xr.merge([ds, nav], compat="override")
    merged = merged.reset_coords(
        ["latitude", "longitude", "speed_knots"],
        drop=False,  # keep them, just demote to data vars
    )

    if "time" in merged:
        merged = merged.drop_vars("time")

    if "time" in merged.coords:
        merged = merged.reset_coords("time", drop=True)

    return merged


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


def concatenate_and_rechunk(paths, container_name, dim="ping_time", chunks=None):
    if not paths or not chunks:
        return None

    datasets = []
    for path in paths:
        ds = open_zarr_store(path, container_name=container_name, chunks=None)

        # Drop range_sample if present (e.g. redundant indexing axis)
        for var in ["range_sample", "source_filenames"]:
            if var in ds:
                ds = ds.drop_vars(var)

        for v in ("latitude", "longitude", "speed_knots"):
            if v in ds.coords:
                ds = ds.reset_coords(v)

        datasets.append(ds)

    datasets.sort(key=lambda ds: ds[dim].min().values)

    # Concatenate along the specified dimension
    concatenated_ds = xr.concat(
        datasets,
        dim=dim,
        data_vars="all",
        coords="minimal",
        compat="override",
        join="outer")

    if "frequency_nominal" in concatenated_ds:
        freq_1d = concatenated_ds["frequency_nominal"].mean("ping_time")
        concatenated_ds = concatenated_ds.drop_vars("frequency_nominal")
        concatenated_ds["frequency_nominal"] = freq_1d

        # optional: also expose it as a coordinate
        concatenated_ds = concatenated_ds.assign_coords(
            frequency=("channel", freq_1d.values)
        )
    else:
        print('frequency_nominal not in concatenated_ds')

    concatenated_ds = concatenated_ds.chunk(chunks)
    concatenated_ds = fix_chunking(concatenated_ds)

    return concatenated_ds
