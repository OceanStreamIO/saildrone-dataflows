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
    cluster = LocalCluster(n_workers=2, threads_per_worker=1, memory_limit='12GB')
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
    plot_echograms = False
    depth_offset = 0
    echograms_container = 'echograms'
    gps_container_name = 'gpsdata'
    encode_mode = 'complex'
    waveform_mode = 'CW'
    transient_noise_opts = None
    impulse_noise_opts = None
    attenuated_signal_opts = None
    background_noise_opts = None

    chunks = {'ping_time': 500, 'range_sample': 500}
    chunks_denoising = {'ping_time': 500, 'depth': 500}

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
        mask_transient_noise=transient_noise_opts,
        mask_impulse_noise=impulse_noise_opts,
        mask_attenuated_signal=attenuated_signal_opts,
        remove_background_noise=background_noise_opts,
        chunks_denoising=chunks_denoising
    )

    print(payload)

    client.close()
    cluster.close()
