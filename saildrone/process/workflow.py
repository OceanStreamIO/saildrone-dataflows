import logging
import os
import time
import traceback

import numpy as np
import pandas as pd
import xarray as xr

from xarray import Dataset
from pathlib import Path
from typing import Optional, Tuple
from echopype.mask import apply_mask
from echopype.clean import mask_transient_noise as mask_transient_noise_func
from echopype.calibrate import compute_Sv as sv_computation
from echopype.commongrid import compute_NASC, compute_MVBS

from saildrone.store import PostgresDB, FileSegmentService, SurveyService
from saildrone.store import (save_zarr_store as save_zarr_to_blobstorage, open_converted as open_from_blobstorage)
from saildrone.azure_iot import serialize_location_data
from saildrone.denoise import (get_impulse_noise_mask,
                               create_multichannel_mask,
                               get_attenuation_mask,
                               remove_background_noise as remove_background_noise_func)

#from .seabed import get_seabed_mask_multichannel
from .plot import plot_and_upload_echograms
from .process_gps import query_location_points_between_timestamps, extract_start_end_coordinates
from .location import extract_location_data
from ..store.nascpoint_service import NASCPointService


def process_converted_file(source_path: Path, cruise_id: str, reprocess: bool = False, **kwargs) -> dict:
    """Process a converted file and store the results in the database.

    Parameters:
    - source_path: Path
        The path to the converted file.
    - cruise_id: str
        The cruise ID associated with the file.
    - reprocess: bool
        Whether to reprocess the file if it has already been processed.
    - **kwargs: dict
        Additional keyword arguments for processing the file.

    Returns:
    - dict: A dictionary containing the processing status and file name.
    """

    start_time = time.time()
    file_id = None
    file_name = source_path.stem if isinstance(source_path, Path) else source_path

    with PostgresDB() as db_connection:
        file_segment_service = FileSegmentService(db_connection)
        survey_service = SurveyService(db_connection)

        file_info = file_segment_service.get_file_info(file_name)

        if file_info is None:
            raise RuntimeError(f"File '{file_name}' not found in the database.")

        survey_db_id = survey_service.get_survey_by_cruise_id(cruise_id)
        if survey_db_id is None:
            raise RuntimeError(f"Survey with cruse id '{cruise_id}' not found in the database.")

        # Check processing status
        if file_info["processed"] and not reprocess:
            logging.info(f"Skipping already processed file: {file_name}")
            return {"status": "skipped", "file_name": file_name}

        file_id = file_info["id"]

    # Process the file and handle potential errors
    if file_id is None:
        raise RuntimeError(f"File ID not found for file '{file_name}'")

    try:
        return _process_file_workflow(
            file_name=file_name,
            source_path=source_path,
            cruise_id=cruise_id,
            survey_db_id=survey_db_id,
            file_id=file_id,
            start_time=start_time,
            **kwargs
        )
    except Exception as e:
        error_message = f"Error processing file '{file_name}': {e}"
        logging.error(error_message)
        traceback.print_exc()

        # Update the database with the failure details
        with PostgresDB() as db_connection:
            file_segment_service = FileSegmentService(db_connection)
            file_segment_service.update_file_record(
                file_id=file_id,
                failed=True,
                error_details=str(e),
            )

        raise RuntimeError(error_message)


def _process_file_workflow(
    file_name=None,
    source_path=None,
    cruise_id=None,
    survey_db_id=None,
    file_id=None,
    start_time=None,
    **kwargs
) -> dict:
    """Core processing logic, encapsulating the main workflow."""
    load_from_blobstorage = kwargs.get("load_from_blobstorage", False)
    converted_container_name = kwargs.get("converted_container_name")
    colormap = kwargs.get("colormap", "ocean_r")
    chunks = kwargs.get("chunks")
    encode_mode = kwargs.get("encode_mode", "complex")
    waveform_mode = kwargs.get("waveform_mode", "CW")
    depth_offset = kwargs.get("depth_offset", 0)
    plot_echograms = kwargs.get("plot_echograms", False)
    output_path = kwargs.get("output_path")
    save_to_blobstorage = kwargs.get("save_to_blobstorage", False)
    echograms_container = kwargs.get("echograms_container")
    processed_container_name = kwargs.get("processed_container_name")
    save_to_directory = kwargs.get("save_to_directory", False)
    gps_container_name = kwargs.get("gps_container_name")
    compute_nasc_opt = kwargs.get("compute_nasc", False)
    compute_mvbs_opt = kwargs.get("compute_mvbs", False)
    apply_seabed_mask = kwargs.get("apply_seabed_mask", False)

    if load_from_blobstorage is True:
        zarr_path = str(source_path)
        source_path = None
        converted_container_name = None
    else:
        zarr_path = f"{cruise_id}/{file_name}.zarr" if converted_container_name else None

    #####################################################################
    # 1. Load echodata
    #####################################################################
    echodata = open_echodata(
        source_path=source_path,
        container_name=converted_container_name,
        zarr_path=zarr_path,
        chunks=chunks
    )

    if echodata.beam is None:
        error_message = f"No beam data found in file: {file_name}"
        logging.error(error_message)
        with PostgresDB() as db_connection:
            file_segment_service = FileSegmentService(db_connection)
            _update_file_failure(file_segment_service, file_id, error_message)
        raise RuntimeError(error_message)

    #####################################################################
    # 2. Sv computation
    #####################################################################
    try:
        sv_dataset = compute_sv(
            echodata,
            container_name=converted_container_name,
            encode_mode=encode_mode,
            waveform_mode=waveform_mode,
            zarr_path=zarr_path,
            depth_offset=depth_offset
        )
    except Exception as e:
        error_message = f"Failed to compute Sv for file '{file_name}'. Error: {str(e)}"
        logging.error(error_message)
        with PostgresDB() as db_connection:
            file_segment_service = FileSegmentService(db_connection)
            _update_file_failure(file_segment_service, file_id, error_message)
        raise RuntimeError(error_message)

    #####################################################################
    # 3. Plot echograms
    #####################################################################
    echogram_files = None

    if plot_echograms:
        echogram_files = plot_and_upload_echograms(sv_dataset,
                                                   cruise_id=cruise_id,
                                                   file_base_name=file_name,
                                                   output_path=output_path,
                                                   save_to_blobstorage=save_to_blobstorage,
                                                   depth_var="depth",
                                                   cmap=colormap,
                                                   container_name=echograms_container)

    sv_dataset_denoised = apply_denoising(sv_dataset, **kwargs)

    # if sv_dataset_denoised is not None and apply_seabed_mask:
    #     seabed_mask = get_seabed_mask_multichannel(sv_dataset_denoised)
    #     sv_dataset_denoised = apply_mask(sv_dataset_denoised, seabed_mask, var_name="Sv")

    if sv_dataset_denoised is not None and plot_echograms:
        echogram_files_denoised = plot_and_upload_echograms(sv_dataset_denoised,
                                                            cruise_id=cruise_id,
                                                            file_base_name=file_name,
                                                            file_name=f"{file_name}_denoised",
                                                            output_path=output_path,
                                                            save_to_blobstorage=save_to_blobstorage,
                                                            depth_var="depth",
                                                            cmap=colormap,
                                                            container_name=echograms_container)
        echogram_files.extend(echogram_files_denoised)
    #####################################################################
    # 4. Compute MVBS
    #####################################################################
    if compute_mvbs_opt:
        ds_MVBS = compute_MVBS(
            sv_dataset,
            range_var="depth",
            range_bin='1m',  # in meters
            ping_time_bin='5s',  # in seconds
        )

        _save_processed_data(
            ds_MVBS,
            cruise_id=cruise_id,
            base_file_name=file_name,
            file_name=f"{file_name}_mvbs",
            processed_container_name=processed_container_name,
            output_path=output_path,
            save_to_directory=save_to_directory,
            save_to_blobstorage=save_to_blobstorage,
        )

        if plot_echograms:
            echogram_files_mvbs = plot_and_upload_echograms(ds_MVBS,
                                                            cruise_id=cruise_id,
                                                            file_base_name=file_name,
                                                            file_name=f"{file_name}_mvbs",
                                                            output_path=output_path,
                                                            save_to_blobstorage=save_to_blobstorage,
                                                            depth_var="depth",
                                                            cmap=colormap,
                                                            container_name=echograms_container)
            echogram_files.extend(echogram_files_mvbs)

    #####################################################################
    # 5. Gather metadata
    #####################################################################
    payload = _prepare_payload(
        sv_dataset,
        file_name=file_name,
        start_time=start_time,
        survey_db_id=survey_db_id,
        echogram_files=echogram_files
    )

    #####################################################################
    # 6. Add location data if not present in the dataset or the database
    #####################################################################
    has_location_in_db = _has_location_data_in_db(file_id)
    has_location_data_ds = _has_location_data(sv_dataset)

    if not has_location_data_ds or not has_location_in_db:
        interp_lat, interp_lon, location_data, gps_data = _load_location_data(sv_dataset, gps_container_name, cruise_id)

        if not has_location_data_ds:
            sv_dataset = sv_dataset.assign_coords(latitude=("ping_time", interp_lat), longitude=("ping_time", interp_lon))

        if not has_location_in_db:
            location_data_str = serialize_location_data(location_data.to_dict(orient="records"))
            gps_result = extract_start_end_coordinates(gps_data)

            payload.update({
                "file_start_lat": gps_result["file_start_lat"],
                "file_start_lon": gps_result["file_start_lon"],
                "file_end_lat": gps_result["file_end_lat"],
                "file_end_lon": gps_result["file_end_lon"],
                "location_data": location_data_str
            })

    #####################################################################
    # 7. Compute NASC
    #####################################################################
    if compute_nasc_opt:
        ds_NASC = compute_NASC(
            sv_dataset,
            range_bin="10m",
            dist_bin="0.5nmi"
        )

        # Log-transform the NASC values for plotting
        ds_NASC["NASC_log"] = 10 * np.log10(ds_NASC["NASC"])
        ds_NASC["NASC_log"].attrs = {
            "long_name": "Log of NASC",
            "units": "m2 nmi-2"
        }
        _save_processed_data(
            ds_NASC,
            cruise_id=cruise_id,
            base_file_name=file_name,
            file_name=f"{file_name}_nasc",
            processed_container_name=processed_container_name,
            output_path=output_path,
            save_to_directory=save_to_directory,
            save_to_blobstorage=save_to_blobstorage,
        )

        dist_max = ds_NASC.attrs["distance_max"]
    else:
        ds_NASC = None
        dist_max = None

    #####################################################################
    # 8. Save the processed data to storage
    #####################################################################
    sv_zarr_path, zarr_store = _save_processed_data(
        sv_dataset,
        cruise_id=cruise_id,
        file_name=file_name,
        base_file_name=file_name,
        processed_container_name=processed_container_name,
        output_path=output_path,
        save_to_directory=save_to_directory,
        save_to_blobstorage=save_to_blobstorage,
    )

    if sv_dataset_denoised is not None:
        _save_processed_data(
            sv_dataset_denoised,
            cruise_id=cruise_id,
            base_file_name=file_name,
            file_name=f"{file_name}_denoised",
            processed_container_name=processed_container_name,
            output_path=output_path,
            save_to_directory=save_to_directory,
            save_to_blobstorage=save_to_blobstorage,
        )

    #####################################################################
    # 9. Update DB with processing results
    #####################################################################
    payload.update({
        "distance": dist_max,
        "location": sv_zarr_path,
        "denoised": sv_dataset_denoised is not None,
    })

    with PostgresDB() as db_connection:
        if ds_NASC:
            nasc_service = NASCPointService(db_connection)
            nasc_service.insert_nasc_points(file_id, survey_db_id, ds_NASC, average=False, clearTables=True)
            nasc_service.insert_nasc_points(file_id, survey_db_id, ds_NASC, average=True)

        file_segment_service = FileSegmentService(db_connection)
        file_segment_service.update_file_record(
            file_id=file_id, **payload, processed=True
        )

    return payload


def _save_processed_data(
    sv_dataset: Dataset,
    cruise_id: str = None,
    file_name: str = None,
    base_file_name: str = None,
    processed_container_name: Optional[str] = None,
    output_path: Optional[str] = None,
    save_to_directory: Optional[bool] = None,
    save_to_blobstorage: Optional[bool] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Save processed data to storage."""
    sv_dataset = sv_dataset.chunk({"channel": 1, "ping_time": 512, "depth": 1024})

    if processed_container_name and save_to_blobstorage:
        sv_zarr_path = base_file_name and f"{cruise_id}/{base_file_name}/{file_name}.zarr" or f"{cruise_id}/{file_name}.zarr"
        zarr_store = save_zarr_to_blobstorage(sv_dataset, container_name=processed_container_name,
                                              zarr_path=sv_zarr_path)
    elif output_path and save_to_directory:
        os.makedirs(f"{output_path}/{base_file_name}", exist_ok=True)
        sv_zarr_path = f"{output_path}/{base_file_name}/{file_name}.zarr"
        sv_dataset.to_zarr(sv_zarr_path, mode="w")
        zarr_store = sv_zarr_path
    else:
        sv_zarr_path, zarr_store = None, None

    return sv_zarr_path, zarr_store


def _has_location_data_in_db(file_id: int) -> bool:
    with PostgresDB() as db_connection:
        file_service = FileSegmentService(db_connection)

        return file_service.file_has_location_data(file_id)


def _has_location_data(sv_dataset: xr.Dataset) -> bool:
    has_lat_coord = "latitude" in sv_dataset.coords
    has_lon_coord = "longitude" in sv_dataset.coords
    has_lat_var = "latitude" in sv_dataset.data_vars
    has_lon_var = "longitude" in sv_dataset.data_vars

    return (has_lat_coord or has_lat_var) and (has_lon_coord or has_lon_var)


def _load_location_data(ds_Sv: xr.Dataset, gps_container_name, cruise_id) -> xr.Dataset:
    """
    Adds latitude and longitude coordinates to an Sv xarray dataset based on provided location data.

    Parameters:
    sv_ds (xr.Dataset): The Sv dataset containing a 'ping_time' dimension.

    Returns:
    xr.Dataset: The updated Sv dataset with added latitude ('lat') and longitude ('lon') variables.
    """
    ping_times = ds_Sv.coords["ping_time"].values
    ping_times_index = pd.DatetimeIndex(ping_times)

    gps_data = query_location_points_between_timestamps(
        ping_times_index[0].isoformat(), ping_times_index[-1].isoformat(), container_name=gps_container_name,
        survey_id=cruise_id
    )

    df = extract_location_data(gps_data)

    # Convert datetime strings to pandas datetime objects
    df["dt"] = pd.to_datetime(df["dt"])

    # Get ping_time from the Sv dataset and ensure it's in datetime format
    times_sv = pd.to_datetime(ds_Sv["ping_time"].values)

    # Perform linear interpolation for latitude and longitude
    interp_lat = np.interp(times_sv.astype(np.int64), df["dt"].astype(np.int64), df["lat"])
    interp_lon = np.interp(times_sv.astype(np.int64), df["dt"].astype(np.int64), df["lon"])

    return interp_lat, interp_lon, df, gps_data


def _prepare_payload(
    sv_dataset: Dataset,
    file_name: str = None,
    start_time: float = None,
    echogram_files: Optional[list] = None,
    survey_db_id: int = None
) -> dict:
    processing_time_ms = int((time.time() - start_time) * 1000)
    ping_times = sv_dataset.coords["ping_time"].values

    payload = {
        "file_name": file_name,
        "size": None,
        "last_modified": None,
        "survey_db_id": survey_db_id,
        "failed": False,
        "error_details": "",
        "processing_time_ms": processing_time_ms,
        "file_npings": len(sv_dataset["ping_time"].values),
        "file_nsamples": len(sv_dataset["range_sample"].values),
        "file_start_time": str(ping_times[0]),
        "file_end_time": str(ping_times[-1]),
        "file_freqs": ",".join(map(str, sv_dataset["frequency_nominal"].values)),
        "file_start_depth": float(sv_dataset["depth"].values[0]),
        "file_end_depth": float(sv_dataset["depth"].values[-1]),
        "echogram_files": echogram_files
    }

        # with PostgresDB() as db_connection:
        #     file_service = FileSegmentService(db_connection)
        #     has_location_data = file_service.file_has_location_data(file_id)
        #
        # if not has_location_data:
        #     gps_data = query_location_points_between_timestamps(
        #         ping_times_index[0].isoformat(), ping_times_index[-1].isoformat(),
        #         container_name=gps_container_name,
        #         survey_id=cruise_id
        #     )
        #
        #     gps_result = extract_start_end_coordinates(gps_data)
        #     location_data = extract_location_data(gps_data)
        #     location_data_str = serialize_location_data(location_data.to_dict(orient="records"))
        #
        #     payload.update({
        #         "file_start_lat": gps_result["file_start_lat"],
        #         "file_start_lon": gps_result["file_start_lon"],
        #         "file_end_lat": gps_result["file_end_lat"],
        #         "file_end_lon": gps_result["file_end_lon"],
        #         "location_data": location_data_str
        #     })

    return payload


def _update_file_failure(file_segment_service, file_id: int, error_message: str):
    """Log failure details in the database."""
    file_segment_service.update_file_record(
        file_id=file_id, failed=True, error_details=error_message
    )


def open_echodata(source_path=None, container_name=None, zarr_path=None, chunks=None):
    if source_path is not None:
        from echopype.echodata.api import open_converted

        return open_converted(source_path, chunks=chunks)

    return open_from_blobstorage(zarr_path, container_name=container_name, chunks=chunks)


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
        'ping_time': 1000,
        'depth': 1000
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
        "encode_mode"
    ]
    splitbeam_args = {k: kwargs.get(k) for k in splitbeam_keys}

    try:
        sv = add_location(sv, echodata, **location_args)
    except (KeyError, ValueError) as e:
        logging.warning(f"Failed to add location due to error: {str(e)}", exc_info=True)

    try:
        sv = add_splitbeam_angle(sv, echodata, to_disk=False, pulse_compression=False, **splitbeam_args)
    except (KeyError, ValueError) as e:
        logging.warning(f"Failed to add split-beam angle due to error: {str(e)}", exc_info=True)

    sv = apply_corrections_ds(sv, depth_offset=kwargs.get("depth_offset"))

    # try:
    #     sv = add_depth(sv, echodata, **depth_args)
    # except (KeyError, ValueError) as e:
    #     logging.warning(f"Failed to add depth due to error: {str(e)}", exc_info=True)

    return sv


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


def apply_denoising(sv_dataset, **kwargs):
    mask_impulse_noise = kwargs.get("mask_impulse_noise", False)
    mask_attenuated_signal = kwargs.get("mask_attenuated_signal", False)
    mask_transient_noise = kwargs.get("mask_transient_noise", False)
    remove_background_noise = kwargs.get("remove_background_noise", False)
    chunks_denoising = kwargs.get("chunks_denoising")

    sv_dataset_denoised = None

    #####################################################################
    # Step 1: Apply impulse noise mask
    #####################################################################
    if mask_impulse_noise:
        mask_channels = []

        params_impulse = {
            "depth_bin": mask_impulse_noise.get('depth_bin'),
            "num_side_pings": mask_impulse_noise.get('num_side_pings'),
            "impulse_noise_threshold": mask_impulse_noise.get('threshold'),
            "range_var": mask_impulse_noise.get('range_var')
        }

        for channel in sv_dataset.coords["channel"].values:
            impulse_noise_mask = get_impulse_noise_mask(sv_dataset, params_impulse, desired_channel=channel)
            mask_channels.append(impulse_noise_mask)
        multi_channel_mask = create_multichannel_mask(mask_channels, sv_dataset)
        sv_dataset_denoised = apply_mask(sv_dataset, multi_channel_mask, var_name="Sv")

    #####################################################################
    # Step 2: Apply attenuated signal mask
    #####################################################################
    if mask_attenuated_signal:
        mask_channels = []
        params_attn = {
            "upper_limit_sl": mask_attenuated_signal.get('upper_limit_sl'),
            "lower_limit_sl": mask_attenuated_signal.get('lower_limit_sl'),
            "num_side_pings": mask_attenuated_signal.get('num_side_pings'),
            "attenuation_signal_threshold": mask_attenuated_signal.get('threshold'),
            "range_var": mask_attenuated_signal.get('range_var')
        }

        if sv_dataset_denoised is None:
            sv_dataset_denoised = sv_dataset

        for channel in sv_dataset.coords["channel"].values:
            attn_signal_mask = get_attenuation_mask(sv_dataset_denoised, params_attn, desired_channel=channel)
            mask_channels.append(attn_signal_mask)
        multi_channel_mask = create_multichannel_mask(mask_channels, sv_dataset_denoised)
        sv_dataset_denoised = apply_mask(sv_dataset_denoised, multi_channel_mask, var_name="Sv")

    #####################################################################
    # Step 3: Apply transient noise mask
    #####################################################################
    if mask_transient_noise:
        threshold = f'{mask_transient_noise.get("threshold", 12.0)}dB'
        exclude_above = f'{mask_transient_noise.get("exclude_above", 250.0)}m'
        depth_bin = f'{mask_transient_noise.get("depth_bin", "10")}m'
        num_side_pings = mask_transient_noise.get('num_side_pings', 25)
        if sv_dataset_denoised is None:
            sv_dataset_denoised = sv_dataset

        transient_mask = mask_transient_noise_func(
            sv_dataset_denoised,
            func=mask_transient_noise.get('operation', 'nanmean'),
            depth_bin=depth_bin,
            num_side_pings=num_side_pings,
            exclude_above=exclude_above,
            transient_noise_threshold=threshold,
            range_var=mask_transient_noise.get('range_var', 'depth'),
            use_index_binning=True,
            chunk_dict=chunks_denoising
        )

        fill_value = np.nan
        sv_dataset_denoised['Sv'] = xr.where(
            transient_mask,
            fill_value,
            sv_dataset_denoised["Sv"]
        )
        # sv_dataset_denoised = apply_mask(sv_dataset, transient_mask, var_name="Sv")

    #####################################################################
    # Step 4: Remove background noise
    #####################################################################
    if remove_background_noise:
        if sv_dataset_denoised is None:
            sv_dataset_denoised = sv_dataset

        sv_dataset_denoised = remove_background_noise_func(sv_dataset_denoised,
                                                           ping_num=remove_background_noise.get('ping_num'),
                                                           SNR_threshold=remove_background_noise.get('SNR_threshold'),
                                                           range_sample_num=remove_background_noise.get('range_sample_num'),
                                                           background_noise_max=remove_background_noise.get('background_noise_max')
                                                           )

    return sv_dataset_denoised
