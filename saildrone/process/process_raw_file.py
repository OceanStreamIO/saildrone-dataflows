import logging
import os
import time

from xarray import Dataset
from pathlib import Path
from prefect_dask import get_dask_client

from saildrone.store import PostgresDB, FileSegmentService
from saildrone.store import save_zarr_store as save_zarr_to_blobstorage
from .convert import convert_file
from .workflow import compute_sv

CHUNKS = {"ping_time": 1000, "range_sample": -1}


def process_raw_file(file_path: Path, survey_id=None, sonar_model='EK80', calibration_file=None, output_path=None,
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
