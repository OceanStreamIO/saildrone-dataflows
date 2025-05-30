import logging
import time

from pathlib import Path
from prefect_dask import get_dask_client

from saildrone.store import PostgresDB, FileSegmentService
from saildrone.store import save_zarr_store as save_zarr_to_blobstorage


CHUNKS = {"ping_time": 2000, "range_sample": -1}


def convert_file_and_save(file_path: Path, cruise_id=None, survey_db_id=None, sonar_model='EK80',
                          calibration_file=None, output_path=None,
                          reprocess=None, converted_container_name=None, chunks=None) -> (int, str, str):
    file_name = file_path.stem
    sv_zarr_path = None
    zarr_store = None

    print('Starting conversion for file:', file_name)

    with PostgresDB() as db_connection:
        file_segment_service = FileSegmentService(db_connection)

        # Check if the file has already been processed
        if file_segment_service.is_file_converted(file_name) and not reprocess:
            print(f'Skipping already converted file: {file_name}')
            return None, None, None

        if file_segment_service.is_file_failed(file_name):
            print(f'Skipping failed file: {file_name}')
            return None, None, None

        file_info = file_segment_service.get_file_info(file_name)
        print(f"File info for {file_name}: {file_info}")

        with get_dask_client():
            print(f"convert_file_and_save: {file_name}, file_path: {file_path}, output_path: {output_path}, converted_container_name: {converted_container_name}")

            try:
                echodata, zarr_path = convert_file(file_name, file_path,
                                                   cruise_id=cruise_id,
                                                   calibration_file=calibration_file,
                                                   container_name=converted_container_name,
                                                   sonar_model=sonar_model, chunks=chunks)
            except Exception as e:
                print(f'Error converting file {file_name}: {e}')
                file_segment_service.update_file_record(file_info['id'], failed=True, error_details=str(e))

                return None, None, None

            if output_path is not None:
                output_zarr_path = f"{output_path}/{file_name}.zarr"
                echodata.to_zarr(output_zarr_path, overwrite=True)

            file_info = file_segment_service.get_file_info(file_name)

            print(f"File info {file_name}: {file_info}")

            if file_info is not None:
                if file_info['converted'] is True:
                    return None, None, None

                file_id = file_info['id']
                file_segment_service.update_file_record(
                    file_id,
                    size=file_path.stat().st_size,
                    location=str(file_path),
                    last_modified=time.ctime(file_path.stat().st_mtime),
                    converted=True
                )
            else:
                file_id = file_segment_service.insert_file_record(
                    file_name,
                    size=file_path.stat().st_size,
                    location=str(file_path),
                    survey_db_id=survey_db_id,
                    last_modified=time.ctime(file_path.stat().st_mtime),
                    converted=True
                )

            return file_id, zarr_store, sv_zarr_path


def convert_file(file_name, file_path, calibration_file=None,
                 cruise_id=None, container_name=None, sonar_model='EK80', chunks=None):

    from echopype.convert.api import open_raw
    from saildrone.calibrate import apply_calibration

    echodata = open_raw(file_path, sonar_model=sonar_model)

    print('Loaded echodata for file:', file_name, echodata)

    if echodata.beam is None:
        return echodata, None

    if calibration_file:
        echodata = apply_calibration(echodata, calibration_file)

    if cruise_id:
        zarr_path = f"{cruise_id}/{file_name}.zarr"
    else:
        zarr_path = f"{file_name}.zarr"

    if container_name is not None:
        if chunks is not None:
            echodata = echodata.chunk(chunks)

        save_zarr_to_blobstorage(echodata, container_name=container_name, zarr_path=zarr_path)

    return echodata, zarr_path
