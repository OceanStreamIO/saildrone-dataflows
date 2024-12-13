import logging
import time
import traceback

import numpy as np
import pandas as pd
import xarray as xr

from xarray import Dataset
from pathlib import Path
from typing import Optional, Tuple

from echopype.calibrate import compute_Sv as sv_computation
from saildrone.store import PostgresDB, FileSegmentService, SurveyService
from saildrone.store import (save_zarr_store as save_zarr_to_blobstorage, open_converted as open_from_blobstorage,
                             plot_and_upload_echograms)
from saildrone.azure_iot import serialize_location_data

from .process_gps import query_location_points_between_timestamps, extract_start_end_coordinates
from .location import extract_location_data


def process_converted_file(
    source_path: Path,
    cruise_id: str,
    output_path: Optional[str] = None,
    load_from_blobstorage: bool = None,
    converted_container_name: Optional[str] = None,
    save_to_blobstorage: Optional[bool] = None,
    save_to_directory: Optional[bool] = None,
    processed_container_name: Optional[str] = None,
    gps_container_name: Optional[str] = None,
    plot_echograms=None,
    echograms_container=None,
    reprocess: bool = False,
    encode_mode='complex',
    waveform_mode='CW',
    chunks: Optional[dict] = None
) -> dict:
    start_time = time.time()
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

        # Process the file and handle potential errors
        try:
            return _process_file_workflow(
                file_name=file_name,
                load_from_blobstorage=load_from_blobstorage,
                source_path=source_path,
                cruise_id=cruise_id,
                survey_db_id=survey_db_id,
                output_path=output_path,
                converted_container_name=converted_container_name,
                processed_container_name=processed_container_name,
                plot_echograms=plot_echograms,
                echograms_container=echograms_container,
                gps_container_name=gps_container_name,
                chunks=chunks,
                encode_mode=encode_mode,
                waveform_mode=waveform_mode,
                save_to_directory=save_to_directory,
                save_to_blobstorage=save_to_blobstorage,
                file_segment_service=file_segment_service,
                start_time=start_time,
                file_id=file_info["id"]
            )
        except Exception as e:
            error_message = f"Error processing file '{file_name}': {e}"
            logging.error(error_message)
            traceback.print_exc()

            # Update the database with the failure details
            file_segment_service.update_file_record(
                file_id=file_info["id"],
                failed=True,
                error_details=str(e),
            )
            raise RuntimeError(error_message)


def _process_file_workflow(
    file_name: str = None,
    source_path: Path = None,
    cruise_id: str = None,
    survey_db_id: int = None,
    output_path: Optional[str] = None,
    load_from_blobstorage: bool = None,
    converted_container_name: Optional[str] = None,
    processed_container_name: Optional[str] = None,
    gps_container_name: Optional[str] = None,
    plot_echograms: Optional[bool] = None,
    echograms_container: Optional[str] = None,
    chunks: Optional[dict] = None,
    encode_mode: str = 'complex',
    waveform_mode: str = 'CW',
    file_segment_service=None,
    save_to_directory=None,
    save_to_blobstorage=None,
    start_time: float = None,
    file_id: int = None
) -> dict:
    """Core processing logic, encapsulating the main workflow."""
    if load_from_blobstorage is True:
        zarr_path = str(source_path)
        source_path = None
        converted_container_name = None
    else:
        zarr_path = f"{cruise_id}/{file_name}.zarr" if converted_container_name else None

    echodata = open_echodata(
        source_path=source_path,
        container_name=converted_container_name,
        zarr_path=zarr_path,
        chunks=chunks
    )

    if echodata.beam is None:
        error_message = f"No beam data found in file: {file_name}"
        logging.error(error_message)
        _update_file_failure(file_segment_service, file_id, error_message)
        raise RuntimeError(error_message)

    # Attempt Sv computation
    try:
        sv_dataset = compute_sv(
            echodata,
            container_name=converted_container_name,
            encode_mode=encode_mode,
            waveform_mode=waveform_mode,
            zarr_path=zarr_path
        )
    except Exception as e:
        error_message = f"Failed to compute Sv for file '{file_name}'. Error: {str(e)}"
        logging.error(error_message)
        _update_file_failure(file_segment_service, file_id, error_message)
        raise RuntimeError(error_message)

    echogram_files = None
    if plot_echograms:
        echogram_files = plot_and_upload_echograms(sv_dataset,
                                                   cruise_id=cruise_id,
                                                   file_base_name=file_name,
                                                   container_name=echograms_container)

    # Save the processed data
    sv_zarr_path, zarr_store = _save_processed_data(
        sv_dataset,
        cruise_id=cruise_id,
        file_name=file_name,
        processed_container_name=processed_container_name,
        output_path=output_path,
        save_to_directory=save_to_directory,
        save_to_blobstorage=save_to_blobstorage,
    )

    # Gather metadata and update the database
    payload = _prepare_payload(
        sv_dataset,
        file_name=file_name,
        file_id=file_id,
        sv_zarr_path=sv_zarr_path,
        cruise_id=cruise_id,
        start_time=start_time,
        survey_db_id=survey_db_id,
        gps_container_name=gps_container_name,
        echogram_files=echogram_files
    )
    file_segment_service.update_file_record(
        file_id=file_id, **payload, processed=True
    )
    return payload


def _save_processed_data(
    sv_dataset: Dataset,
    cruise_id: str = None,
    file_name: str = None,
    processed_container_name: Optional[str] = None,
    output_path: Optional[str] = None,
    save_to_directory: Optional[bool] = None,
    save_to_blobstorage: Optional[bool] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Save processed data to storage."""
    if processed_container_name and save_to_blobstorage:
        sv_zarr_path = f"{cruise_id}/{file_name}.zarr"
        zarr_store = save_zarr_to_blobstorage(sv_dataset, container_name=processed_container_name,
                                              zarr_path=sv_zarr_path)
    elif output_path and save_to_directory:
        sv_zarr_path = f"{output_path}/{file_name}.zarr"
        sv_dataset.to_zarr(sv_zarr_path, mode="w")
        zarr_store = sv_zarr_path
    else:
        sv_zarr_path, zarr_store = None, None

    return sv_zarr_path, zarr_store


def _prepare_payload(
    sv_dataset: Dataset,
    file_id: int = None,
    file_name: str = None,
    sv_zarr_path: Optional[str] = None,
    cruise_id: str = None,
    start_time: float = None,
    echogram_files: Optional[list] = None,
    gps_container_name: Optional[str] = None,
    survey_db_id: int = None
) -> dict:
    with PostgresDB() as db_connection:
        file_service = FileSegmentService(db_connection)
        has_location_data = file_service.file_has_location_data(file_id)

    processing_time_ms = int((time.time() - start_time) * 1000)

    ping_times = sv_dataset.coords["ping_time"].values
    ping_times_index = pd.DatetimeIndex(ping_times)

    payload = {
        "file_name": file_name,
        "size": None,
        "last_modified": None,
        "location": sv_zarr_path,
        "survey_db_id": survey_db_id,
        "failed": False,
        "error_details": "",
        "processing_time_ms": processing_time_ms,
        "file_npings": len(sv_dataset["ping_time"].values),
        "file_nsamples": len(sv_dataset["range_sample"].values),
        "file_start_time": str(ping_times[0]),
        "file_end_time": str(ping_times[-1]),
        "file_freqs": ",".join(map(str, sv_dataset["frequency_nominal"].values)),
        "file_start_depth": float(sv_dataset["range_sample"].values[0]),
        "file_end_depth": float(sv_dataset["range_sample"].values[-1]),
        "echogram_files": echogram_files
    }

    if not has_location_data:
        gps_data = query_location_points_between_timestamps(
            ping_times_index[0].isoformat(), ping_times_index[-1].isoformat(), container_name=gps_container_name,
            survey_id=cruise_id
        )

        gps_result = extract_start_end_coordinates(gps_data)
        location_data = extract_location_data(gps_data)
        location_data_str = serialize_location_data(location_data.to_dict(orient="records"))

        payload.update({
            "file_start_lat": gps_result["file_start_lat"],
            "file_start_lon": gps_result["file_start_lon"],
            "file_end_lat": gps_result["file_end_lat"],
            "file_end_lon": gps_result["file_end_lon"],
            "location_data": location_data_str
        })

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
               encode_mode='complex'):
    if chunks is not None:
        echodata = open_echodata(zarr_path=zarr_path, source_path=source_path, container_name=container_name,
                                 chunks=chunks)

    sv_dataset = sv_computation(echodata, waveform_mode=waveform_mode, encode_mode=encode_mode)
    sv_dataset = enrich_sv_dataset(sv_dataset, echodata, depth_offset=0)
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
