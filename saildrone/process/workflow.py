import logging

import numpy as np
import xarray as xr

from xarray import Dataset
from pathlib import Path
from prefect_dask import get_dask_client

from echopype.calibrate import compute_Sv as sv_computation
from saildrone.store import PostgresDB, FileSegmentService
from saildrone.store import save_zarr_store as save_zarr_to_blobstorage, open_converted as open_from_blobstorage


def process_file(file_path: Path, survey_id=None, sonar_model='EK80', calibration_file=None, output_path=None,
                 converted_container_name=None, processed_container_name=None, chunks=None) -> (Dataset, str, str):
    file_name = file_path.stem
    sv_zarr_path = None


def process_converted_file(source_path: Path = None,
                           survey_id=None,
                           output_path=None,
                           converted_container_name=None,
                           processed_container_name=None,
                           chunks=None) -> (Dataset, str, str):
    if isinstance(source_path, Path):
        file_name = source_path.stem
    else:
        file_name = source_path

    sv_zarr_path = None
    zarr_store = None

    with PostgresDB() as db_connection:
        file_segment_service = FileSegmentService(db_connection)

        # Check if the file has already been processed
        if file_segment_service.is_file_processed(file_name):
            logging.info(f'Skipping already processed file: {file_name}')
            return None, None, None

        with get_dask_client():
            zarr_path = None
            if converted_container_name is not None:
                zarr_path = f"{survey_id}/{file_name}.zarr"

            echodata = open_echodata(source_path=source_path,
                                     container_name=converted_container_name,
                                     zarr_path=zarr_path,
                                     chunks=chunks)
            output_zarr_path = None

            if echodata.beam is None:
                logging.info(f'No beam data found in file: {file_name}')
                return

            sv_dataset = compute_sv(echodata,
                                    container_name=converted_container_name,
                                    zarr_path=zarr_path,
                                    source_path=output_zarr_path)

            if processed_container_name is not None:
                sv_zarr_path = f"{survey_id}/{file_name}.zarr"
                zarr_store = save_zarr_to_blobstorage(sv_dataset, container_name=processed_container_name,
                                                      zarr_path=sv_zarr_path)
            elif output_path is not None:
                sv_zarr_path = f"{output_path}/{file_name}.zarr"
                sv_dataset.to_zarr(sv_zarr_path, mode='w')
                zarr_store = sv_zarr_path

            file_info = file_segment_service.get_file_info(file_name)
            file_segment_service.update_file_record(
                file_id=file_info['id'],
                processed=True
            )

            return sv_dataset, zarr_store, sv_zarr_path


def open_echodata(source_path=None, container_name=None, zarr_path=None, chunks=None):
    if source_path is not None:
        from echopype.echodata.api import open_converted

        return open_converted(source_path, chunks=chunks)

    return open_from_blobstorage(zarr_path, container_name=container_name, chunks=chunks)


def compute_sv(echodata, container_name=None, source_path=None, zarr_path=None, chunks=None):
    if chunks is not None:
        echodata = open_echodata(zarr_path=zarr_path, source_path=source_path, container_name=container_name,
                                 chunks=chunks)

    sv_dataset = sv_computation(echodata, waveform_mode='CW', encode_mode='complex')
    sv_dataset = sv_dataset.chunk({
        'channel': 2,
        'ping_time': 1000,
        'range_sample': 1000
    })

    return sv_dataset


def enrich_sv_dataset(sv: xr.Dataset, echodata, **kwargs) -> xr.Dataset:
    """
    Enhances the input `sv` dataset by adding depth, location, and split-beam angle information.

    Parameters:
    - sv (xr.Dataset): Volume backscattering strength (Sv) from the given echodata.
    - echodata (EchoData): An EchoData object holding the raw data.
    - **kwargs: Keyword arguments specific to `add_depth()`, `add_location()`, and `add_splitbeam_angle()`.

    Returns:
    - xr.Dataset: An enhanced dataset with depth, location, and split-beam angle.
    """
    from echopype.consolidate import add_location, add_splitbeam_angle, add_depth

    depth_keys = ["depth_offset", "tilt", "downward"]
    depth_args = {k: kwargs.get(k) for k in depth_keys}

    location_keys = ["nmea_sentence"]
    location_args = {k: kwargs.get(k) for k in location_keys}

    splitbeam_keys = [
        "waveform_mode",
        "encode_mode",
        "pulse_compression",
        "storage_options"
    ]
    splitbeam_args = {k: kwargs.get(k) for k in splitbeam_keys}

    try:
        sv = add_depth(sv, echodata, **depth_args)
    except (KeyError, ValueError) as e:
        logging.warning(f"Failed to add depth due to error: {str(e)}", exc_info=True)

    try:
        sv = add_location(sv, echodata, **location_args)
    except (KeyError, ValueError) as e:
        logging.warning(f"Failed to add location due to error: {str(e)}", exc_info=True)

    try:
        sv = add_splitbeam_angle(sv, echodata, **splitbeam_args)
    except (KeyError, ValueError) as e:
        logging.warning(f"Failed to add split-beam angle due to error: {str(e)}", exc_info=True)

    sv = apply_corrections_ds(sv, depth_offset=kwargs.get("depth_offset"))

    return sv


def apply_corrections_ds(dataset, depth_offset=None):
    # Remove empty pings
    dataset = dataset.dropna(dim='ping_time', how='all', subset=['Sv'])

    # Correct echo range if 'echo_range' is present
    if 'echo_range' in dataset and depth_offset is not None:
        dataset = correct_echo_range(dataset, depth_offset=depth_offset)

        if 'depth' not in dataset:
            dataset = dataset.rename({'range_sample': 'depth'})

    return dataset


def correct_echo_range(ds, depth_offset=6):
    # Replace channel and ping_time with their first elements
    first_channel = ds["channel"].values[0]
    first_ping_time = ds["ping_time"].values[0]

    # Slice the echo_range to get the desired range of values
    selected_echo_range = ds["echo_range"].sel(channel=first_channel, ping_time=first_ping_time)
    selected_echo_range = selected_echo_range.values.tolist()
    selected_echo_range = [value + depth_offset for value in selected_echo_range]

    # Find min and max ignoring NaNs
    min_val = np.nanmin(selected_echo_range)
    max_val = np.nanmax(selected_echo_range)

    # Assign the values to the depth coordinate
    ds = ds.assign_coords(range_sample=selected_echo_range)

    # Remove NaN values
    ds = ds.sel(range_sample=slice(min_val, max_val))

    return ds
