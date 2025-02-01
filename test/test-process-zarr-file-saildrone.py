from pathlib import Path

from dotenv import load_dotenv
from dask.distributed import Client, LocalCluster

from saildrone.process import process_converted_file

load_dotenv()

GPS_OUTPUT_FOLDER = "./test/gps-processed"

# @pytest.fixture
# def db_setup():
#     with PostgresDB() as db:
#         db.create_tables()
#         yield db


def test_file_workflow_saildrone():
    cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit='12GB')
    client = Client(cluster)

    source_path = Path('converted/SD_TPOS2023_v03/SD_TPOS2023_v03-Phase0-D20230825-T085959-0.zarr')
    cruise_id = 'SD_TPOS2023_v03'
    output_path = 'test/processed'
    save_to_blobstorage = False
    load_from_blobstorage = True
    converted_container_name = 'converted'
    save_to_directory = True
    processed_container_name = 'processedlocal'
    reprocess = True
    plot_echograms = True
    depth_offset = 0
    echograms_container = 'echograms'
    gps_container_name = 'gpsdata'
    encode_mode = 'complex'
    waveform_mode = 'CW'
    impulse_noise_opts = dict(
        depth_bin=10,
        num_side_pings=2,
        threshold=10,
        range_var="depth"
    )
    attenuated_signal_opts = dict(
        upper_limit_sl=180,
        lower_limit_sl=300,
        num_side_pings=15,
        threshold=10,
        range_var="depth"
    )

    transient_noise_opts = dict(
        exclude_above=250.0,
        threshold=12.0
    )
    background_noise_opts = dict(
        ping_num=20,
        range_sample_num=50,
        background_noise_max=None,
        SNR_threshold=3.0
    )

    chunks = {'ping_time': 1000, 'range_sample': 1000}
    chunks_denoising = {'ping_time': 1000, 'depth': 1000}

    payload = process_converted_file(
        source_path,
        cruise_id=cruise_id,
        output_path=output_path,
        chunks=chunks,
        load_from_blobstorage=load_from_blobstorage,
        converted_container_name=converted_container_name,
        save_to_blobstorage=save_to_blobstorage,
        save_to_directory=save_to_directory,
        processed_container_name=processed_container_name,
        reprocess=reprocess,
        depth_offset=depth_offset,
        plot_echograms=plot_echograms,
        echograms_container=echograms_container,
        gps_container_name=gps_container_name,
        encode_mode=encode_mode,
        waveform_mode=waveform_mode,
        compute_nasc=False,
        compute_mvbs=False,
        colormap='ocean',
        # mask_transient_noise=transient_noise_opts,
        mask_impulse_noise=impulse_noise_opts,
        # mask_attenuated_signal=attenuated_signal_opts,
        # remove_background_noise=background_noise_opts,
        chunks_denoising=chunks_denoising
    )

    print(payload)

    client.close()
    cluster.close()
