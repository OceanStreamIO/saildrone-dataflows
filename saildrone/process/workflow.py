import logging
import os
import shutil
import time
import traceback

import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from datetime import datetime

from xarray import Dataset
from pathlib import Path
from typing import Optional, Tuple, List

from echopype.commongrid import compute_NASC, compute_MVBS
from prefect.futures import as_completed

from saildrone.store import PostgresDB, FileSegmentService, SurveyService, open_zarr_store, get_azure_blob_filesystem
from saildrone.store import (save_zarr_store as save_zarr_to_blobstorage, list_zarr_files)
from saildrone.azure_iot import serialize_location_data
from saildrone.utils import load_local_files, get_metadata_for_files
from saildrone.denoise import (background_noise_mask, impulsive_noise_mask,
                               build_full_mask, apply_full_mask, transient_noise_mask, attenuation_mask)

from .seabed import mask_true_seabed
from .echodata import open_echodata
from .sv_dataset import compute_sv
from .plot import plot_and_upload_echograms, ensure_channel_labels, plot_and_upload_masks
from .process_gps import query_location_points_between_timestamps, extract_start_end_coordinates
from .location import extract_location_data
from ..store.nascpoint_service import NASCPointService

PARAMETER_NAMES = ("upper_limit_sl", "lower_limit_sl", "num_side_pings", "threshold")


def get_files_list(
        source_directory: str = None,
        cruise_id: str = None,
        load_from_blobstorage: bool = True,
        source_container: str = None,
        get_list_from_db: bool = True,
        start_datetime: Optional[datetime] = None,
        end_datetime: Optional[datetime] = None,
        reprocess: bool = True
):
    """Fetch the list of files to be processed."""
    files = None

    with PostgresDB() as db_connection:
        survey_service = SurveyService(db_connection)
        survey_id = survey_service.get_survey_by_cruise_id(cruise_id)
        if not survey_id:
            survey_id = survey_service.insert_survey(cruise_id)
            logging.info(f"Inserted new survey with cruise_id: {cruise_id}")

        if get_list_from_db:
            file_service = FileSegmentService(db_connection)
            condition = "" if reprocess else "AND processed IS NOT True"
            if start_datetime and end_datetime:
                condition += (
                    f" AND file_start_time > '{start_datetime}' "
                    f"AND file_end_time < '{end_datetime}'"
                )
            files = file_service.get_files_by_survey_id(survey_id, condition)
            if not files:
                logging.warning(f"No files found for survey_id: {survey_id} with condition: {condition}")
                return [], []

    if load_from_blobstorage:
        file_names = [f['file_name'] for f in files] if files else None
        zarr_paths = list_zarr_files(
            source_container,
            cruise_id=cruise_id,
            file_names=file_names,
        )
    elif files:
        # If files are provided and not loading from blob storage, construct paths
        zarr_paths = [Path(source_directory) / f"{file['file_name']}.zarr" for file in files]
    else:
        # If no files are provided, load local files from the source directory
        zarr_paths = load_local_files(source_directory, source_directory, '*.zarr')

    return get_metadata_for_files(zarr_paths, files)


def process_files_list(files_list_with_data, save_to_netcdf, denoised=None, **kwargs):
    in_flight = []
    side_running_tasks = []
    netcdf_outputs = []

    plot_echograms = kwargs.get('plot_echograms', False)
    chunks_sv_data = kwargs.get('chunks_sv_data', None)
    output_container = kwargs.get('output_container', None)
    echograms_container = kwargs.get('echograms_container', None)
    colormap = kwargs.get('colormap', 'ocean_r')
    apply_seabed_mask = kwargs.get('apply_seabed_mask', False)
    batch_size = kwargs.get('batch_size', 10)
    task_plot_echograms_normal = kwargs.get('task_plot_echograms_normal', None)
    task_plot_echograms_denoised = kwargs.get('task_plot_echograms_denoised', None)
    task_save_to_netcdf = kwargs.get('task_save_to_netcdf', None)
    process_single_file = kwargs.get('process_single_file', None)
    task_plot_echograms_seabed = kwargs.get('task_plot_echograms_seabed', None)
    trigger_netcdf_flow = kwargs.get('trigger_netcdf_flow', None)

    for source_path, file_record in files_list_with_data:
        location_data = file_record["location_data"] if 'location_data' in file_record else None
        file_name = file_record["file_name"]
        future = process_single_file.submit(source_path,
                                            file_name=file_name,
                                            denoised=denoised,
                                            location_data=location_data,
                                            **kwargs)

        if save_to_netcdf:
            future_nc_task = task_save_to_netcdf.submit(future, file_name, output_container, chunks_sv_data)
            side_running_tasks.append(future_nc_task)
            netcdf_outputs.append(future_nc_task)

        if plot_echograms:
            future_plot_task = task_plot_echograms_normal.submit(future,
                                                                 file_name,
                                                                 output_container,
                                                                 echograms_container,
                                                                 chunks_sv_data,
                                                                 colormap)
            future_plot_task_denoised = task_plot_echograms_denoised.submit(future,
                                                                            file_name,
                                                                            output_container,
                                                                            echograms_container,
                                                                            chunks_sv_data,
                                                                            colormap)

            side_running_tasks.append(future_plot_task)
            side_running_tasks.append(future_plot_task_denoised)

            if apply_seabed_mask:
                future_plot_task = task_plot_echograms_seabed.submit(future,
                                                                     file_name,
                                                                     output_container,
                                                                     echograms_container,
                                                                     chunks_sv_data,
                                                                     colormap)
                side_running_tasks.append(future_plot_task)

        in_flight.append(future)

        # Throttle when max concurrent tasks reached
        if len(in_flight) >= batch_size:
            finished = next(as_completed(in_flight))
            in_flight.remove(finished)

    # Wait for remaining tasks
    for future_task in in_flight + side_running_tasks:
        future_task.result()

    if save_to_netcdf:
        future_zip = trigger_netcdf_flow.submit(
            file_list=netcdf_outputs,
            container=output_container
        )
        future_zip.wait()

        if os.path.exists('/tmp/oceanstream/netcdfdata'):
            shutil.rmtree('/tmp/oceanstream/netcdfdata', ignore_errors=True)

    logging.info("All batches have been processed.")


def process_converted_file(source_path: Path, **kwargs) -> dict:
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
    file_name = kwargs.get("file_name", source_path.stem if isinstance(source_path, Path) else source_path)
    # with PostgresDB() as db_connection:
    #     file_segment_service = FileSegmentService(db_connection)
    #     survey_service = SurveyService(db_connection)
    #
    #     file_info = file_segment_service.get_file_info(file_name)
    #
    #     if file_info is None:
    #         raise RuntimeError(f"File '{file_name}' not found in the database.")
    #
    #     survey_db_id = survey_service.get_survey_by_cruise_id(cruise_id)
    #     if survey_db_id is None:
    #         raise RuntimeError(f"Survey with cruse id '{cruise_id}' not found in the database.")
    #
    #     # Check processing status
    #     if file_info["processed"] and not reprocess:
    #         logging.info(f"Skipping already processed file: {file_name}")
    #         return {"status": "skipped", "file_name": file_name}
    #
    #     file_id = file_info["id"]
    #
    # # Process the file and handle potential errors
    # if file_id is None:
    #     raise RuntimeError(f"File ID not found for file '{file_name}'")

    try:
        return _process_file_workflow(
            source_path=source_path,
            start_time=start_time,
            **kwargs
        )
    except Exception as e:
        error_message = f"Error processing file '{file_name}': {e}"
        logging.error(error_message)
        traceback.print_exc()

        # Update the database with the failure details
        # with PostgresDB() as db_connection:
        #     file_segment_service = FileSegmentService(db_connection)
        #     file_segment_service.update_file_record(
        #         file_id=file_id,
        #         failed=True,
        #         error_details=str(e),
        #     )

        raise RuntimeError(error_message)


def _process_file_workflow(
        file_name=None,
        source_path=None,
        start_time=None,
        **kwargs
) -> dict:
    """Core processing logic, encapsulating the main workflow."""
    cruise_id = kwargs.get("cruise_id", None)
    location_data = kwargs.get("location_data", None)
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

        # with PostgresDB() as db_connection:
        #     file_segment_service = FileSegmentService(db_connection)
        #     _update_file_failure(file_segment_service, file_id, error_message)

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

        sv_dataset = ensure_channel_labels(sv_dataset, add_freq=True)

        sv_dataset_denoised = apply_denoising(sv_dataset, **kwargs)
        sv_dataset_seabed = None

        if apply_seabed_mask:
            ds_Sv = sv_dataset_denoised if sv_dataset_denoised is not None else sv_dataset
            sv_dataset_seabed = mask_true_seabed(ds_Sv)

        depth_1d = (
            sv_dataset["depth"]
            .isel(channel=0, ping_time=0)
            .data
        )

        # attach it as a coordinate on range_sample
        sv_dataset = (
            sv_dataset
            .assign_coords(depth=("range_sample", depth_1d))
            .swap_dims({"range_sample": "depth"})
        )

        if sv_dataset_denoised is not None:
            sv_dataset_denoised = (
                sv_dataset_denoised
                .assign_coords(depth=("range_sample", depth_1d))
                .swap_dims({"range_sample": "depth"})
            )

        if sv_dataset_seabed is not None:
            sv_dataset_seabed = (
                sv_dataset_seabed
                .assign_coords(depth=("range_sample", depth_1d))
                .swap_dims({"range_sample": "depth"})
            )

    except Exception as e:
        error_message = f"Failed to compute Sv for file '{file_name}'. Error: {str(e)}"
        logging.error(error_message)
        traceback.print_exc()

        # with PostgresDB() as db_connection:
        #     file_segment_service = FileSegmentService(db_connection)
        #     _update_file_failure(file_segment_service, file_id, error_message)

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

    if sv_dataset_seabed is not None and plot_echograms:
        echogram_files_seabed = plot_and_upload_echograms(sv_dataset_seabed,
                                                          cruise_id=cruise_id,
                                                          file_base_name=file_name,
                                                          file_name=f"{file_name}_seabed",
                                                          output_path=output_path,
                                                          save_to_blobstorage=save_to_blobstorage,
                                                          depth_var="depth",
                                                          cmap=colormap,
                                                          container_name=echograms_container)
        echogram_files.extend(echogram_files_seabed)
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
        echogram_files=echogram_files,
        cruise_id=cruise_id
    )

    #####################################################################
    # 6. Add location data if not present in the dataset or the database
    #####################################################################
    has_location_in_db = location_data is not None
    has_location_data_ds = _has_location_data(sv_dataset)

    if not has_location_data_ds or not has_location_in_db:
        interp_lat, interp_lon, location_data, gps_data = _load_location_data_from_geoparquet(sv_dataset,
                                                                                              gps_container_name,
                                                                                              cruise_id)

        if not has_location_data_ds:
            sv_dataset = sv_dataset.assign_coords(latitude=("ping_time", interp_lat),
                                                  longitude=("ping_time", interp_lon))

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
        ds_NASC = compute_and_save_nasc(cruise_id, file_name, output_path, processed_container_name,
                                        save_to_blobstorage, save_to_directory, sv_dataset)

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

    if sv_dataset_seabed is not None:
        _save_processed_data(
            sv_dataset_seabed,
            cruise_id=cruise_id,
            base_file_name=file_name,
            file_name=f"{file_name}_seabed",
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
        "seabed_mask": sv_dataset_seabed is not None
    })

    # with PostgresDB() as db_connection:
    #     if ds_NASC:
    #         nasc_service = NASCPointService(db_connection)
    #         nasc_service.insert_nasc_points(file_id, survey_db_id, ds_NASC, average=False, clearTables=True)
    #         nasc_service.insert_nasc_points(file_id, survey_db_id, ds_NASC, average=True)
    #
    #     file_segment_service = FileSegmentService(db_connection)
    #     file_segment_service.update_file_record(
    #         file_id=file_id, **payload, processed=True
    #     )
    #
    # payload["file_id"] = file_id
    return payload


def compute_and_save_nasc(
        sv_dataset,
        compute_nasc_opts=None,
        cruise_id=None,
        file_name=None,
        output_path=None,
        zarr_path=None,
        container_name=None,
        save_to_blobstorage=True,
        save_to_directory=False
):
    compute_nasc_opts = compute_nasc_opts or {}
    range_bin = compute_nasc_opts.get("range_bin", "10m")
    dist_bin = compute_nasc_opts.get("dist_bin", "0.5nmi")
    closed = compute_nasc_opts.get("closed", "left")

    ds_NASC = compute_NASC(
        sv_dataset,
        range_bin=range_bin,
        dist_bin=dist_bin,
        closed=closed,
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
        file_name=f"{file_name}--nasc",
        processed_container_name=container_name,
        output_path=output_path,
        zarr_path=zarr_path,
        save_to_directory=save_to_directory,
        save_to_blobstorage=save_to_blobstorage,
    )
    return ds_NASC


def compute_and_save_mvbs(
        sv_dataset,
        compute_mvbs_opts=None,
        cruise_id=None,
        file_name=None,
        output_path=None,
        zarr_path=None,
        container_name=None,
        save_to_blobstorage=True,
        save_to_directory=False
):
    compute_mvbs_opts = compute_mvbs_opts or {}
    range_var = compute_mvbs_opts.get("range_var", "depth")
    range_bin = compute_mvbs_opts.get("range_bin", "20m")
    ping_time_bin = compute_mvbs_opts.get("ping_time_bin", "5s")
    closed = compute_mvbs_opts.get("closed", "left")

    ds_MVBS = compute_MVBS(
        sv_dataset,
        range_var=range_var,
        range_bin=range_bin,
        ping_time_bin=ping_time_bin,
        closed=closed
    )

    _save_processed_data(
        ds_MVBS,
        cruise_id=cruise_id,
        base_file_name=file_name,
        file_name=f"{file_name}--mvbs",
        processed_container_name=container_name,
        output_path=output_path,
        zarr_path=zarr_path,
        save_to_directory=save_to_directory,
        save_to_blobstorage=save_to_blobstorage,
    )
    return ds_MVBS


def _save_processed_data(
        sv_dataset: Dataset,
        cruise_id: str = None,
        file_name: str = None,
        base_file_name: str = None,
        processed_container_name: Optional[str] = None,
        output_path: Optional[str] = None,
        zarr_path: Optional[str] = None,
        save_to_directory: Optional[bool] = None,
        save_to_blobstorage: Optional[bool] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Save processed data to storage."""
    if processed_container_name and save_to_blobstorage:
        if zarr_path:
            sv_zarr_path = zarr_path
        else:
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


def _load_location_data_from_geoparquet(ds_Sv: xr.Dataset, gps_container_name, cruise_id) -> xr.Dataset:
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
        survey_db_id: int = None,
        cruise_id: str = None
) -> dict:
    processing_time_ms = int((time.time() - start_time) * 1000)
    ping_times = sv_dataset.coords["ping_time"].values

    payload = {
        "file_name": file_name,
        "size": None,
        "last_modified": None,
        "survey_db_id": survey_db_id,
        "cruise_id": cruise_id,
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


def apply_denoising(sv_dataset, **kwargs):
    impulse_noise_opts = kwargs.get("mask_impulse_noise", None)
    attenuated_signal_opts = kwargs.get("mask_attenuated_signal", None)
    transient_noise_opts = kwargs.get("mask_transient_noise", None)
    background_noise_opts = kwargs.get("remove_background_noise", None)
    drop_pings = kwargs.get("drop_pings", False)

    if not any([impulse_noise_opts, attenuated_signal_opts, transient_noise_opts, background_noise_opts]):
        return sv_dataset

    stages = {
    }

    if attenuated_signal_opts:
        stages["signal-attenuation"] = {
            "fn": attenuation_mask,
            "param_sets": attenuated_signal_opts
        }

    if impulse_noise_opts:
        stages["impulsive"] = {
            "fn": impulsive_noise_mask,
            "param_sets": impulse_noise_opts,
        }

    if transient_noise_opts:
        stages["transient"] = {
            "fn": transient_noise_mask,
            "param_sets": transient_noise_opts,
        }

    if background_noise_opts:
        stages["background"] = {
            "fn": background_noise_mask,
            "param_sets": background_noise_opts,
        }

    full_mask, stage_cubes = build_full_mask(sv_dataset, stages=stages, return_stage_masks=True)
    sv_dataset_denoised = apply_full_mask(sv_dataset, full_mask, drop_pings=drop_pings)

    mask_dict = {"full": full_mask, **dict(stage_cubes)}

    mask_vars = {
        f"mask_{key.replace(' ', '_').lower()}": arr.astype("bool")
        for key, arr in mask_dict.items()
    }

    mask_vars = {
        name: arr.broadcast_like(sv_dataset["Sv"]).assign_attrs(
            long_name=f"{key} quality-control mask (True = bad)"
        )
        for (name, arr), (key, _) in zip(mask_vars.items(), mask_dict.items())
    }

    sv_dataset_denoised = sv_dataset_denoised.merge(mask_vars, compat="no_conflicts")

    return sv_dataset_denoised, mask_dict
