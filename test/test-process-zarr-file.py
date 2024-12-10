import pandas as pd
import pytest
import time

import fsspec

import pandas as pd

from unittest.mock import patch, MagicMock
from pathlib import Path
from dask.distributed import Client, LocalCluster

from echopype.echodata.api import open_converted
from saildrone.process.workflow import compute_sv
from saildrone.store import PostgresDB, FileSegmentService
from saildrone.process import (process_converted_file, query_location_points_between_timestamps, plot_sv_data,
                               extract_start_end_coordinates)


GPS_OUTPUT_FOLDER = "./test/gps-processed"

@pytest.fixture
def db_setup():
    with PostgresDB() as db:
        db.create_tables()
        yield db


def test_process_file_success(db_setup):
    # db_setup.empty_files_table()

    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)

    file_name = 'SD_TPOS2023_v03-Phase0-D20230530-T173550-0'
    file_info = Path(f'./test/zarr/{file_name}.zarr')
    file_segment_service = FileSegmentService(db_setup)
    # file_segment_service.insert_file_record(file_name, size=file_info.stat().st_size, converted=True)

    # assert not file_segment_service.is_file_processed(file_name)
    echodata = open_converted(file_info)
    sv_dataset = compute_sv(echodata)
    plot_sv_data(sv_dataset, file_name, output_path='test/processed')

    # Calculate processing times
    # processing_time_ms = int((time.time() - start_time) * 1000)
    # ping_times = sv_dataset.coords['ping_time'].values
    # ping_times_index = pd.DatetimeIndex(ping_times)
    # day_date = ping_times_index[0].date()
    # total_recording_time = (ping_times_index[-1] - ping_times_index[0]).total_seconds()
    # first_ping_time = ping_times_index[0].time()
    # start_time = first_ping_time.isoformat()
    #
    # file_npings = len(sv_dataset["ping_time"].values)
    # file_nsamples = len(sv_dataset["range_sample"].values)
    # file_start_time = str(sv_dataset["ping_time"].values[0])
    # file_end_time = str(sv_dataset["ping_time"].values[-1])
    # file_freqs = ",".join(map(str, sv_dataset["frequency_nominal"].values))
    # file_start_depth = str(sv_dataset["range_sample"].values[0])
    # file_end_depth = str(sv_dataset["range_sample"].values[-1])
    #
    # gps_data = query_location_points_between_timestamps(GPS_OUTPUT_FOLDER, file_start_time, file_end_time)
    # gps_result = extract_start_end_coordinates(gps_data)
    #
    # payload = {
    #     "file_name": file_name,
    #     "zarr_path_converted": '',
    #     "zarr_path_sv": '',
    #     "date": day_date.isoformat(),
    #     "duration": total_recording_time,
    #     "file_npings": file_npings,
    #     "file_nsamples": file_nsamples,
    #     "file_start_time": file_start_time,
    #     "file_end_time": file_end_time,
    #     "file_freqs": file_freqs,
    #     "file_start_depth": file_start_depth,
    #     "file_end_depth": file_end_depth,
    #     "file_start_lat": gps_result['file_start_lat'],
    #     "file_start_lon": gps_result['file_start_lat'],
    #     "file_end_lat": gps_result['file_start_lat'],
    #     "file_end_lon": gps_result['file_start_lat'],
    #     "start_time": start_time,
    #     "dataset_id": None,
    #     "campaign_id": None,
    #     "processing_time_ms": processing_time_ms,
    #     "gps_data": None
    # }

    print(sv_dataset)
    # print(payload)

    client.close()
    cluster.close()
    #
    # sv_path = f"test/processed/{file_name}_Sv.zarr"
    # sv_dataset.to_zarr(sv_path, mode='w')
    #
    # plot_sv_data(sv_dataset, file_name, output_path='test/processed')
    # assert file_segment_service.is_file_processed(file_name)
    #
    # db_setup.cursor.execute("SELECT * FROM files WHERE file_name = %s", (file_name,))
    # file_record = db_setup.cursor.fetchone()
    # assert file_record is not None
    # assert file_record[1] == file_name
