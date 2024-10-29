import pytest
import fsspec

from unittest.mock import patch, MagicMock
from pathlib import Path
from saildrone.store import PostgresDB, FileSegmentService
from saildrone.process import process_file


@pytest.fixture
def db_setup():
    with PostgresDB() as db:
        db.create_tables()
        yield db


@patch('saildrone.store.save_zarr_store')
@patch('prefect_dask.get_dask_client', return_value=MagicMock())
def test_process_file_success(mock_get_dask_client, mock_save_zarr_store, db_setup):
    db_setup.empty_files_table()

    file_info = Path('./test/data/SD_TPOS2023_v03-Phase0-D20230530-T220328-0.raw')
    file_name = 'SD_TPOS2023_v03-Phase0-D20230530-T220328-0'

    file_segment_service = FileSegmentService(db_setup)
    assert not file_segment_service.is_file_processed(file_name)
    sv_dataset, _, _ = process_file(
        file_path=file_info,
        survey_id='TPOS2023',
        sonar_model='EK80',
        calibration_file='./saildrone/utils/calibration_values.xlsx'
    )
    sv_path = f"test/processed/{file_name}_Sv.zarr"
    sv_dataset.to_zarr(sv_path, mode='w')

    assert file_segment_service.is_file_processed(file_name)

    db_setup.cursor.execute("SELECT * FROM files WHERE file_name = %s", (file_name,))
    file_record = db_setup.cursor.fetchone()
    assert file_record is not None
    assert file_record[1] == file_name


# @patch('prefect_dask.get_dask_client', return_value=MagicMock())
# def test_process_file_with_chunks_local(mock_save_zarr_store, db_setup):
#     db_setup.empty_files_table()
#
#     file_info = Path('./test/data/SD_TPOS2023_v03-Phase0-D20230530-T220328-0.raw')
#     file_name = 'SD_TPOS2023_v03-Phase0-D20230530-T220328-0'
#
#     file_segment_service = FileSegmentService(db_setup)
#     assert not file_segment_service.is_file_processed(file_name)
#
#     process_file(
#         file_path=file_info,
#         sonar_model='EK80',
#         chunks={"ping_time": 1000, "range_sample": -1},
#         output_path='test/processed',
#         calibration_file='./saildrone/utils/calibration_values.xlsx'
#     )
#
#     assert file_segment_service.is_file_processed(file_name)
#
#     db_setup.cursor.execute("SELECT * FROM files WHERE file_name = %s", (file_name,))
#     file_record = db_setup.cursor.fetchone()
#     assert file_record is not None
#     assert file_record[1] == file_name
#
#
# @patch('prefect_dask.get_dask_client', return_value=MagicMock())
# def test_process_file_local_to_blobstorage(mock_get_dask_client, db_setup):
#     db_setup.empty_files_table()
#     file_info = Path('./test/data/SD_TPOS2023_v03-Phase0-D20230530-T220328-0.raw')
#     file_name = 'SD_TPOS2023_v03-Phase0-D20230530-T220328-0'
#     survey_id = 'TPOXTEST'
#
#     file_segment_service = FileSegmentService(db_setup)
#     assert not file_segment_service.is_file_processed(file_name)
#     sv_dataset, zarr_store, sv_zarr_path = process_file(
#         file_path=file_info,
#         survey_id=survey_id,
#         processed_container_name='processed',
#         sonar_model='EK80',
#         output_path='test/processed',
#         chunks={"ping_time": 1000, "range_sample": -1},
#         calibration_file='./saildrone/utils/calibration_values.xlsx'
#     )
#
#     assert isinstance(zarr_store, fsspec.FSMap)
#     assert sv_zarr_path == f"{survey_id}/{file_name}/{file_name}_Sv.zarr"
#     assert file_segment_service.is_file_processed(file_name)
#
#     db_setup.cursor.execute("SELECT * FROM files WHERE file_name = %s", (file_name,))
#     file_record = db_setup.cursor.fetchone()
#     assert file_record is not None
#     assert file_record[1] == file_name
