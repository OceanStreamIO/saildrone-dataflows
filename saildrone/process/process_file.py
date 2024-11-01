import logging
import os
import time

from xarray import Dataset

from echopype.convert.api import open_raw
from echopype.calibrate import compute_Sv as sv_computation

from pathlib import Path
from saildrone.store import PostgresDB, SurveyService, FileSegmentService
from saildrone.process import apply_calibration
from saildrone.store import save_zarr_store as save_zarr_to_blobstorage, open_converted as open_from_blobstorage
from prefect_dask import get_dask_client


CHUNKS = {"ping_time": 1000, "range_sample": -1}


def process_file(file_path: Path, survey_id=None, sonar_model='EK80', calibration_file=None, output_path=None,
                 converted_container_name=None, processed_container_name=None, chunks=None) -> (Dataset, str, str):
    file_name = file_path.stem
    sv_zarr_path = None
    zarr_store = None

    with PostgresDB() as db_connection:
        file_segment_service = FileSegmentService(db_connection)

        # Check if the file has already been processed
        if file_segment_service.is_file_processed(file_name):
            logging.info(f'Skipping already processed file: {file_name}')
            return None, None, None

        with get_dask_client():
            echodata, zarr_path = convert_file(file_name, file_path, survey_id=survey_id,
                                               calibration_file=calibration_file,
                                               container_name=converted_container_name, sonar_model=sonar_model)

            output_zarr_path = None
            if output_path is not None:
                output_zarr_path = f"{output_path}/{file_name}/{file_name}.zarr"
                os.makedirs(f"{output_path}/{file_name}", exist_ok=True)
                echodata.to_zarr(output_zarr_path, overwrite=True)

            if echodata.beam is None:
                logging.info(f'No beam data found in file: {file_name}')
                return

            sv_dataset = compute_sv(echodata, container_name=converted_container_name, zarr_path=zarr_path,
                                    source_path=output_zarr_path, chunks=chunks)

            if processed_container_name is not None:
                sv_zarr_path = f"{survey_id}/{file_name}/{file_name}_Sv.zarr"
                zarr_store = save_zarr_to_blobstorage(sv_dataset, container_name=processed_container_name,
                                                      zarr_path=sv_zarr_path)

            elif output_path is not None:
                sv_zarr_path = f"{output_path}/{file_name}/{file_name}_Sv.zarr"
                sv_dataset.to_zarr(sv_zarr_path, mode='w')
                zarr_store = sv_zarr_path

            file_id = file_segment_service.insert_file_record(
                file_name=file_name,
                size=file_path.stat().st_size,
                location=str(file_path),
                last_modified=time.ctime(file_path.stat().st_mtime)
            )

            file_segment_service.mark_file_processed(file_id)

            return sv_dataset, zarr_store, sv_zarr_path


def convert_file_and_save(file_path: Path, survey_id=None, sonar_model='EK80', calibration_file=None, output_path=None,
                          converted_container_name=None) -> (int, str, str):
    file_name = file_path.stem
    sv_zarr_path = None
    zarr_store = None

    with PostgresDB() as db_connection:
        file_segment_service = FileSegmentService(db_connection)

        # Check if the file has already been processed
        if file_segment_service.is_file_converted(file_name):
            logging.info(f'Skipping already converted file: {file_name}')
            return None, None, None

        with get_dask_client():
            echodata, zarr_path = convert_file(file_name, file_path, survey_id=survey_id,
                                               calibration_file=calibration_file,
                                               container_name=converted_container_name, sonar_model=sonar_model)

            if output_path is not None:
                output_zarr_path = f"{output_path}/{file_name}.zarr"
                echodata.to_zarr(output_zarr_path, overwrite=True)

            file_id = file_segment_service.insert_file_record(
                file_name=file_name,
                size=file_path.stat().st_size,
                location=str(file_path),
                last_modified=time.ctime(file_path.stat().st_mtime)
            )

            file_segment_service.mark_file_converted(file_id)

            return file_id, zarr_store, sv_zarr_path


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


def convert_file(file_name, file_path, calibration_file=None,
                 survey_id=None, container_name=None, sonar_model='EK80'):
    echodata = open_raw(file_path, sonar_model=sonar_model)

    if echodata.beam is None:
        return echodata, None

    echodata = apply_calibration(echodata, calibration_file)

    if survey_id:
        zarr_path = f"{survey_id}/{file_name}.zarr"
    else:
        zarr_path = f"{file_name}.zarr"

    if container_name is not None:
        # echodata = echodata.chunk(CHUNKS)
        save_zarr_to_blobstorage(echodata, container_name=container_name, zarr_path=zarr_path)

    return echodata, zarr_path


def open_echodata(source_path=None, container_name=None, zarr_path=None, chunks=None):
    if source_path is not None:
        from echopype.echodata.api import open_converted

        return open_converted(source_path, chunks=chunks)

    return open_from_blobstorage(zarr_path, container_name=container_name, chunks=chunks)