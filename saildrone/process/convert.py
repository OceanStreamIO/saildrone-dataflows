import logging
import time

from echopype.convert.api import open_raw
from pathlib import Path
from prefect_dask import get_dask_client

from saildrone.store import PostgresDB, FileSegmentService
from saildrone.process import apply_calibration
from saildrone.store import save_zarr_store as save_zarr_to_blobstorage


CHUNKS = {"ping_time": 1000, "range_sample": -1}


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