import logging
import sys

import xarray as xr

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


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
