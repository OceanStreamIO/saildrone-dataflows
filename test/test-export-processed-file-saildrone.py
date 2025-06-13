from pathlib import Path

from dotenv import load_dotenv
from dask.distributed import Client, LocalCluster

from echopype import open_raw
from saildrone.calibrate import apply_calibration
from saildrone.process.concat import merge_location_data
from saildrone.process.sv_dataset import compute_sv
from saildrone.process import apply_denoising, process_converted_file
from saildrone.process.plot import ensure_channel_labels, plot_and_upload_echograms
from saildrone.process.seabed import mask_true_seabed
from saildrone.store import open_zarr_store, save_zarr_store

load_dotenv()

GPS_OUTPUT_FOLDER = "./test/gps-processed"


def test_file_workflow_saildrone():
    cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit='12GB')
    client = Client(cluster)
    file_name = 'SD_TPOS2023_v03-Phase0-D20230808-T015958-0'
    cruise_id = 'SD_TPOS2023_v03'

    source_container_name = 'processed-data'
    export_container_name = 'export11'
    impulse_noise_opts = dict(
        depth_bin=5,
        num_side_pings=2,
        threshold=10,
        range_var="range_sample",
        use_index_binning=True
    )
    attenuated_signal_opts = dict(
        upper_limit_sl=180,
        lower_limit_sl=300,
        num_side_pings=15,
        threshold=10,
        range_var="range_sample"
    )

    transient_noise_opts = dict(
        exclude_above=250.0,
        threshold=12.0
    )
    background_noise_opts = dict(
        ping_num=5,
        range_sample_num=30,
        background_noise_max=None,
        SNR_threshold=3.0
    )

    chunks = {'ping_time': 2000, 'depth': -1}
    zarr_path = f"{file_name}/{file_name}.zarr"

    ds = open_zarr_store(zarr_path, cruise_id=cruise_id,
                         container_name=source_container_name, chunks=chunks,
                         rechunk_after=True)

    # Merge location data
    # ds = merge_location_data(ds, location_data)
    file_path = f"{cruise_id}/{file_name}/{file_name}.zarr"
    sv_dataset_denoised = apply_denoising(ds, chunks_denoising=chunks,
                                          mask_impulse_noise=None,
                                          mask_attenuated_signal=attenuated_signal_opts,
                                          mask_transient_noise=None,
                                          remove_background_noise=None)
    file_path_denoised = f"{cruise_id}/{file_name}/{file_name}--denoised.zarr"
    print(sv_dataset_denoised)

    # save_zarr_store(sv_dataset_denoised, container_name=export_container_name, zarr_path=file_path_denoised,
    #                 chunks=chunks)

    client.close()
    cluster.close()
