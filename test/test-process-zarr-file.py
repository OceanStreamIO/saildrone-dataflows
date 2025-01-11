import pandas as pd
import pytest
import xarray as xr
import numpy as np
import time

import fsspec

import pandas as pd

from unittest.mock import patch, MagicMock
from pathlib import Path
from dask.distributed import Client, LocalCluster

from echopype.echodata.api import open_converted
from echopype.mask import apply_mask
from echopype.commongrid import compute_MVBS, compute_NASC
from echopype.clean import mask_transient_noise
from saildrone.denoise import (get_impulse_noise_mask,
                               create_multichannel_mask,
                               get_attenuation_mask,
                               remove_background_noise,
                               get_transient_noise_mask)
from saildrone.process.plot import plot_noise_mask
from saildrone.process.workflow import compute_sv
from saildrone.store import PostgresDB, FileSegmentService
from saildrone.process import (process_converted_file, query_location_points_between_timestamps, plot_sv_data,
                               extract_start_end_lat_lon,
                               extract_start_end_coordinates)


GPS_OUTPUT_FOLDER = "./test/gps-processed"

# @pytest.fixture
# def db_setup():
#     with PostgresDB() as db:
#         db.create_tables()
#         yield db


@pytest.mark.skip
def test_process_file_apply_denoising():
    # db_setup.empty_files_table()

    filters = {
        "mask_impulse_noise": {
            "enabled": False,
            "depth_bin": 5,
            "num_side_pings": 2,
            "impulse_noise_threshold": 10,
            "range_var": "depth",
            "use_index_binning": False
        },
        "mask_attenuated_signal": {
            "enabled": False,
            "upper_limit_sl": 180,
            "lower_limit_sl": 300,
            "num_side_pings": 15,
            "attenuation_signal_threshold": 10,
            "start": 0,
            "range_var": "depth"
        },
        "mask_transient_noise": {
            "enabled": False,
            # "r0": 200,
            # "r1": 1000,
            # "n": 30,
            # "thr": [3, 1],
            # "roff": 250,
            # "jumps": 5,
            # "maxts": -35,
            # "start": 0
            "operation": "mean",
            "depth_bin": 5,
            "num_side_pings": 20,
            "exclude_above": 250,
            "transient_noise_threshold": 10,
            # "range_var": "depth",
            # "use_index_binning": True
        },

        "remove_background_noise": {
            "enabled": False,
            "ping_num": 20,
            "range_sample_num": 50,
            "background_noise_max": None,
            "SNR_threshold": 3.0
        }
    }

    start_time = time.time()
    cluster = LocalCluster(n_workers=2, threads_per_worker=1, memory_limit='12GB')
    client = Client(cluster)

    file_name = 'SE2204_-D20220704-T162237'
    file_info = Path(f'./test/reka-shipboard/converted/{file_name}.zarr')
    # file_segment_service = FileSegmentService(db_setup)
    # file_segment_service.insert_file_record(file_name, size=file_info.stat().st_size, converted=True)

    # assert not file_segment_service.is_file_processed(file_name)
    echodata = open_converted(file_info)
    sv_dataset = compute_sv(echodata, encode_mode='power')
    sv_dataset = sv_dataset.chunk({'ping_time': 500, 'depth': 500})

    # plot_sv_data(sv_dataset, file_name, output_path='test/processed', depth_var='depth')

    ds_MVBS = compute_MVBS(
        sv_dataset,
        range_var="depth",
        range_bin='1m',  # in meters
        ping_time_bin='5s',  # in seconds
    )
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

    print('NASC', ds_NASC.to_dict())

    plot_sv_data(ds_MVBS, f'{file_name}_MVBS', output_path='test/processed', depth_var='depth')

    ##############################################################
    # Impulse noise removal
    ##############################################################
    params_impulse = filters["mask_impulse_noise"]
    if params_impulse['enabled']:
        mask_channels = []
        for channel in sv_dataset.coords["channel"].values:
            idx = sv_dataset.channel.values.tolist().index(channel)
            impulse_noise_mask = get_impulse_noise_mask(sv_dataset, params_impulse, desired_channel=channel)
            plot_noise_mask(impulse_noise_mask, f'{file_name}_impulse_denoised_{idx}',
                            echogram_path=f"test/processed")
            mask_channels.append(impulse_noise_mask)
        multi_channel_mask = create_multichannel_mask(mask_channels, sv_dataset)
        sv_dataset_denoised = apply_mask(sv_dataset, multi_channel_mask, var_name="Sv")
        plot_sv_data(sv_dataset_denoised, f'{file_name}_impulse_denoised', output_path='test/processed', depth_var='depth')

    ##############################################################
    # Attenuated signal removal
    ##############################################################
    params_attn = filters["mask_attenuated_signal"]
    if params_attn['enabled']:
        mask_channels = []
        for channel in sv_dataset.coords["channel"].values:
            idx = sv_dataset.channel.values.tolist().index(channel)
            attn_signal_mask = get_attenuation_mask(sv_dataset, params_attn, desired_channel=channel)
            plot_noise_mask(attn_signal_mask, f'{file_name}_attn_denoised_{idx}',
                            echogram_path=f"test/processed")
            mask_channels.append(attn_signal_mask)
        multi_channel_mask = create_multichannel_mask(mask_channels, sv_dataset)
        sv_dataset_denoised = apply_mask(sv_dataset, multi_channel_mask, var_name="Sv")
        plot_sv_data(sv_dataset_denoised, f'{file_name}_attn_denoised', output_path='test/processed', depth_var='depth')

    ##############################################################
    # Transient noise removal
    ##############################################################
    params_tran = filters["mask_transient_noise"]

    if params_tran['enabled']:
        transient_mask = mask_transient_noise(
            sv_dataset,
            func="nanmean",
            depth_bin="10m",
            num_side_pings=25,
            exclude_above="250.0m",
            transient_noise_threshold="8.0dB",
            range_var="depth",
            use_index_binning=True,
            chunk_dict={'ping_time': 500, 'depth': 500}
        )

        mask_channels = []
        # for channel in sv_dataset.coords["channel"].values:
        #     idx = sv_dataset.channel.values.tolist().index(channel)
        #     transient_noise_mask = get_transient_noise_mask(sv_dataset, params_tran, desired_channel=channel)
        #
        #     # transient_noise_mask = transient_noise_mask.compute()
        #     plot_noise_mask(transient_noise_mask, f'{file_name}_transient_denoised_{idx}',
        #                     echogram_path=f"test/processed")
        #     mask_channels.append(transient_noise_mask)
        #
        # multi_channel_mask = create_multichannel_mask(mask_channels, sv_dataset)
        fill_value = np.nan
        sv_dataset['Sv'] = xr.where(
            transient_mask,
            fill_value,
            sv_dataset["Sv"]
        )
        # sv_dataset_denoised = apply_mask(sv_dataset, transient_mask, var_name="Sv")
        plot_sv_data(sv_dataset, f'{file_name}_tran_denoised', output_path='test/processed', depth_var='depth')

    ##############################################################
    # Background noise removal
    ##############################################################
    params_bg_noise = filters["remove_background_noise"]
    if params_bg_noise['enabled']:
        sv_dataset_denoised = remove_background_noise(sv_dataset,
                                                      ping_num=params_bg_noise['ping_num'],
                                                      SNR_threshold=params_bg_noise['SNR_threshold'],
                                                      range_sample_num=params_bg_noise['range_sample_num'],
                                                      background_noise_max=params_bg_noise['background_noise_max']
                                                      )
        plot_sv_data(sv_dataset_denoised, f'{file_name}_bg_denoised', output_path='test/processed', depth_var='depth')

    # Calculate processing times
    processing_time_ms = int((time.time() - start_time) * 1000)
    ping_times = sv_dataset.coords['ping_time'].values
    ping_times_index = pd.DatetimeIndex(ping_times)
    day_date = ping_times_index[0].date()
    total_recording_time = (ping_times_index[-1] - ping_times_index[0]).total_seconds()
    first_ping_time = ping_times_index[0].time()
    start_time = first_ping_time.isoformat()

    file_npings = len(sv_dataset["ping_time"].values)
    file_nsamples = len(sv_dataset["range_sample"].values)
    file_start_time = str(sv_dataset["ping_time"].values[0])
    file_end_time = str(sv_dataset["ping_time"].values[-1])
    file_freqs = ",".join(map(str, sv_dataset["frequency_nominal"].values))
    file_start_depth = str(sv_dataset["range_sample"].values[0])
    file_end_depth = str(sv_dataset["range_sample"].values[-1])

    gpsdata = extract_start_end_lat_lon(sv_dataset)
    # gps_data = query_location_points_between_timestamps(GPS_OUTPUT_FOLDER, file_start_time, file_end_time)
    # gps_result = extract_start_end_coordinates(gps_data)
    #
    payload = {
        "file_name": file_name,
        "zarr_path_converted": '',
        "zarr_path_sv": '',
        "date": day_date.isoformat(),
        "duration": total_recording_time,
        "file_npings": file_npings,
        "file_nsamples": file_nsamples,
        "file_start_time": file_start_time,
        "file_end_time": file_end_time,
        "file_freqs": file_freqs,
        "file_start_depth": file_start_depth,
        "file_end_depth": file_end_depth,
        "file_start_lat": gpsdata['file_start_lat'],
        "file_start_lon": gpsdata['file_start_lon'],
        "file_end_lat": gpsdata['file_end_lat'],
        "file_end_lon": gpsdata['file_end_lon'],
        "start_time": start_time,
        "dataset_id": None,
        "campaign_id": None,
        "processing_time_ms": processing_time_ms,
        "gps_data": None
    }

    print(payload)
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
