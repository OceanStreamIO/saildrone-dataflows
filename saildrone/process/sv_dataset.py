import logging
import numpy as np
import xarray as xr

from echopype.calibrate import compute_Sv as sv_computation
from .echodata import open_echodata


def compute_sv(echodata, container_name=None, source_path=None, zarr_path=None, chunks=None, waveform_mode='CW',
               encode_mode='complex', depth_offset=0):
    if chunks is not None:
        echodata = open_echodata(zarr_path=zarr_path, source_path=source_path, container_name=container_name,
                                 chunks=chunks)

    sv_dataset = sv_computation(echodata, waveform_mode=waveform_mode, encode_mode=encode_mode)
    sv_dataset = enrich_sv_dataset(sv_dataset, echodata, depth_offset=depth_offset, waveform_mode=waveform_mode,
                                   encode_mode=encode_mode)
    sv_dataset = sv_dataset.chunk({
        'channel': 2,
        'ping_time': 1000
    })

    return sv_dataset


def choose_depth_flags(echodata, depth_offset=0, downward=True):
    platform = echodata["Platform"]

    # ────────────────────────────────────────────────────────────────────
    # 1.  vertical offsets: transducer_offset_z − water_level − vertical_offset
    # ────────────────────────────────────────────────────────────────────
    vert_vars = ["transducer_offset_z", "water_level", "vertical_offset"]
    have_all_vert = all(name in platform for name in vert_vars)

    use_platform_vertical_offsets = False
    if have_all_vert:
        vert_values = platform[vert_vars].to_array()
        use_platform_vertical_offsets = not vert_values.isnull().any()

    # If we can use the three offsets, ignore any user-supplied depth_offset
    depth_offset_kwarg = None if use_platform_vertical_offsets else depth_offset

    # ────────────────────────────────────────────────────────────────────
    # 2.  tilt correction: choose platform angles first, else beam angles
    # ────────────────────────────────────────────────────────────────────
    plat_angle_vars = ["platform_pitch", "platform_roll"]
    plat_angles_present = all(v in platform for v in plat_angle_vars) and not (
        platform[plat_angle_vars].to_array().isnull().any()
    )

    # Beam angles exist if any `beam_direction_x` variable is in the Sonar groups
    beam_angles_present = any(
        "beam_direction_x" in var for var in echodata["Sonar"].data_vars
    )

    use_platform_angles = plat_angles_present
    use_beam_angles = (not plat_angles_present) and beam_angles_present

    # ────────────────────────────────────────────────────────────────────
    # 3.  pack and return
    # ────────────────────────────────────────────────────────────────────
    return {
        "depth_offset": depth_offset_kwarg,
        "use_platform_vertical_offsets": use_platform_vertical_offsets,
        "use_platform_angles": use_platform_angles,
        "use_beam_angles": use_beam_angles,
        "downward": bool(downward),
    }


def enrich_sv_dataset(ds_Sv: xr.Dataset, echodata, **kwargs) -> xr.Dataset:
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

    location_keys = ["nmea_sentence"]
    location_args = {k: kwargs.get(k) for k in location_keys}

    splitbeam_keys = [
        "waveform_mode",
        "encode_mode"
    ]
    splitbeam_args = {k: kwargs.get(k) for k in splitbeam_keys}

    try:
        ds_Sv = add_location(ds_Sv, echodata, **location_args)
    except (KeyError, ValueError) as e:
        logging.warning(f"Failed to add location due to error: {str(e)}", exc_info=True)

    try:
        ds_Sv = add_splitbeam_angle(ds_Sv, echodata, to_disk=False, pulse_compression=False, **splitbeam_args)
    except (KeyError, ValueError) as e:
        logging.warning(f"Failed to add split-beam angle due to error: {str(e)}", exc_info=True)

    try:
        flags = choose_depth_flags(echodata)
        ds_Sv = add_depth(ds_Sv, echodata, **flags)
    except (KeyError, ValueError) as e:
        logging.warning(f"Failed to add depth due to error: {str(e)}", exc_info=True)

    ds_Sv = ds_Sv.dropna(dim='ping_time', how='all', subset=['Sv'])

    if "echo_range" in ds_Sv and "echo_range" not in ds_Sv.coords:
        # pick any ping/channel slice – every slice is identical
        # 1. extract a *single* 1-D vector and strip the wrapper
        er_1d = (
            ds_Sv["echo_range"]  # 3-D (channel × ping × sample)
            .isel(channel=0, ping_time=0)  # → 1-D but still a DataArray
            .data  # → bare NumPy array
        )

        # 2. add it as a coordinate on the same dimension as range_sample
        ds_Sv = ds_Sv.assign_coords(echo_range=("range_sample", er_1d))

    return ds_Sv


def apply_corrections_ds(dataset, depth_offset=None):
    # Remove empty pings
    dataset = dataset.dropna(dim='ping_time', how='all', subset=['Sv'])

    # Correct echo range if 'echo_range' is present
    if 'echo_range' in dataset and depth_offset is not None:
        try:
            dataset = correct_echo_range(dataset, depth_offset=depth_offset)
        except Exception as e:
            print(f"Error correcting echo range: {e}")

    return dataset


def correct_echo_range(ds: xr.Dataset, depth_offset: float = 0.0) -> xr.Dataset:
    """
    Correct the echo range values in a dataset by applying a depth offset and filtering invalid values.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing echo_range and range_sample dimensions
    depth_offset : float, optional
        Offset to add to the echo range values, by default 0.0

    Returns
    -------
    xr.Dataset
        Dataset with corrected depth values and filtered invalid entries

    Notes
    -----
    The function performs the following operations:
    1. Preserves the original range_sample values
    2. Applies depth offset to echo_range values
    3. Filters out invalid depth values
    4. Renames range_sample to depth
    """
    if "range_sample" not in ds.dims:
        return ds

    if "echo_range" not in ds:
        logging.warning("echo_range not found in dataset")
        return ds

    # Store original range_sample values
    ds = ds.assign(original_range_sample=("range_sample", ds["range_sample"].values))

    # Get first channel and ping_time - assuming these are constant for the range
    first_channel = ds["channel"].values[0]
    first_ping_time = ds["ping_time"].values[0]

    # Extract and correct echo range values using numpy operations
    selected_echo_range = ds["echo_range"].sel(channel=first_channel, ping_time=first_ping_time)
    corrected_depth = selected_echo_range.values + depth_offset

    # Find valid range using numpy operations
    min_val = np.nanmin(corrected_depth)
    max_val = np.nanmax(corrected_depth)

    # Update coordinates and rename
    ds = ds.assign_coords(range_sample=corrected_depth)
    ds = ds.rename({'range_sample': 'depth'})

    # Filter to valid depth range
    ds = ds.sel(depth=slice(min_val, max_val))

    # Remove any remaining NaN depths
    valid_depth_indices = ~np.isnan(ds["depth"].values)
    ds = ds.isel(depth=valid_depth_indices)

    # Restore original range_sample
    ds = ds.rename({"original_range_sample": "range_sample"})

    return ds
