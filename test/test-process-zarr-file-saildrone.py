from pathlib import Path

from dotenv import load_dotenv
from dask.distributed import Client, LocalCluster

from echopype import open_raw
from saildrone.process.sv_dataset import compute_sv
from saildrone.process import apply_denoising
from saildrone.process.plot import ensure_channel_labels, plot_and_upload_echograms
from saildrone.process.seabed import mask_true_seabed

load_dotenv()

GPS_OUTPUT_FOLDER = "./test/gps-processed"

# @pytest.fixture
# def db_setup():
#     with PostgresDB() as db:
#         db.create_tables()
#         yield db


def test_raw_file_workflow_hb():
    cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit='12GB')
    client = Client(cluster)

    file_name = 'D20160806-T123616'
    raw_file = Path(f'test/HB2302/{file_name}.raw')
    zarr_path = Path(f'test/HB2302/converted/{file_name}.zarr')
    echodata = open_raw(raw_file, sonar_model="EK80")
    echodata.to_zarr(zarr_path, overwrite=True)
    cruise_id = 'HB2302'
    output_path = 'test/processed'

    impulse_noise_opts = dict(
        depth_bin=5,
        num_side_pings=2,
        threshold=10,
        range_var="echo_range",
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
    colormap = 'ocean_r'
    depth_offset = 5
    encode_mode = 'power'
    waveform_mode = 'CW'
    plot_echograms = True

    sv_dataset = compute_sv(
        echodata,
        encode_mode=encode_mode,
        waveform_mode=waveform_mode,
        zarr_path=zarr_path,
        depth_offset=depth_offset
    )

    sv_dataset = ensure_channel_labels(sv_dataset, add_freq=True)
    sv_dataset_denoised = apply_denoising(sv_dataset,
                                          mask_impulse_noise=impulse_noise_opts,
                                          mask_attenuated_signal=None,
                                          mask_transient_noise=None,
                                          remove_background_noise=background_noise_opts,)
    ds_Sv = sv_dataset_denoised if sv_dataset_denoised is not None else sv_dataset
    sv_dataset_seabed = mask_true_seabed(ds_Sv)

    depth_1d = (
        sv_dataset["depth"]
        .isel(channel=0, ping_time=0)
        .data
    )

    # attach it as a coordinate on range_sample
    sv_dataset = (
        sv_dataset
        .assign_coords(depth=("range_sample", depth_1d))
        .swap_dims({"range_sample": "depth"})
    )

    if sv_dataset_denoised is not None:
        sv_dataset_denoised = (
            sv_dataset_denoised
            .assign_coords(depth=("range_sample", depth_1d))
            .swap_dims({"range_sample": "depth"})
        )

    if sv_dataset_seabed is not None:
        sv_dataset_seabed = (
            sv_dataset_seabed
            .assign_coords(depth=("range_sample", depth_1d))
            .swap_dims({"range_sample": "depth"})
        )

    if plot_echograms:
        echogram_files = plot_and_upload_echograms(sv_dataset,
                                                   cruise_id=cruise_id,
                                                   file_base_name=file_name,
                                                   output_path=output_path,
                                                   save_to_blobstorage=False,
                                                   depth_var="depth",
                                                   cmap=colormap)

    if sv_dataset_denoised is not None and plot_echograms:
        echogram_files_denoised = plot_and_upload_echograms(sv_dataset_denoised,
                                                            cruise_id=cruise_id,
                                                            file_base_name=file_name,
                                                            file_name=f"{file_name}_denoised",
                                                            output_path=output_path,
                                                            save_to_blobstorage=False,
                                                            depth_var="depth",
                                                            cmap=colormap)

    if sv_dataset_seabed is not None and plot_echograms:
        echogram_files_seabed = plot_and_upload_echograms(sv_dataset_seabed,
                                                          cruise_id=cruise_id,
                                                          file_base_name=file_name,
                                                          file_name=f"{file_name}_seabed",
                                                          output_path=output_path,
                                                          save_to_blobstorage=False,
                                                          depth_var="depth",
                                                          cmap=colormap)
    client.close()
    cluster.close()


# def test_file_workflow_saildrone():
#     cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit='12GB')
#     client = Client(cluster)
#
#     #source_path = Path('converted/SD_TPOS2023_v03/SD_TPOS2023_v03-Phase0-D20230825-T085959-0.zarr')
#     source_path = Path('converted/SD_TPOS2023_v03/SD_TPOS2023_v03-Phase0-D20231010-T095958-0.zarr')
#     #source_path = Path('converted/SD_TPOS2023_v03/SD_TPOS2023_v03-Phase0-D20231008-T005959-0.zarr')
#     source_path = Path('converted/SD_TPOS2023_v03/SD_TPOS2023_v03-Phase0-D20231008-T065958-0.zarr')
#     source_path = Path('test/reka-shipboard/converted/SE2204_-D20220706-T171126.zarr')
#     # cruise_id = 'SD_TPOS2023_v03'
#     cruise_id = 'SE2204'
#     output_path = 'test/processed'
#     save_to_blobstorage = False
#     load_from_blobstorage = False
#     converted_container_name = 'converted'
#     save_to_directory = True
#     processed_container_name = 'localnetcdftest'
#     #processed_container_name = 'processedlocal'
#     reprocess = True
#     plot_echograms = True
#     depth_offset = 6
#     echograms_container = 'echograms'
#     gps_container_name = 'gpsdata'
#     # encode_mode = 'complex'
#     encode_mode = 'power'
#     waveform_mode = 'CW'
#     impulse_noise_opts = dict(
#         depth_bin=5,
#         num_side_pings=2,
#         threshold=10,
#         range_var="echo_range",
#         use_index_binning=True
#     )
#     attenuated_signal_opts = dict(
#         upper_limit_sl=180,
#         lower_limit_sl=300,
#         num_side_pings=15,
#         threshold=10,
#         range_var="range_sample"
#     )
#
#     transient_noise_opts = dict(
#         exclude_above=250.0,
#         threshold=12.0
#     )
#     background_noise_opts = dict(
#         ping_num=5,
#         range_sample_num=30,
#         background_noise_max=None,
#         SNR_threshold=3.0
#     )
#
#     chunks = {'ping_time': 2000, 'range_sample': 1000}
#     chunks_denoising = {'ping_time': 1000, 'depth': 1000}
#
#     payload = process_converted_file(
#         source_path,
#         cruise_id=cruise_id,
#         output_path=output_path,
#         chunks=chunks,
#         load_from_blobstorage=load_from_blobstorage,
#         converted_container_name=converted_container_name,
#         save_to_blobstorage=save_to_blobstorage,
#         save_to_directory=save_to_directory,
#         processed_container_name=processed_container_name,
#         reprocess=reprocess,
#         depth_offset=depth_offset,
#         plot_echograms=plot_echograms,
#         echograms_container=echograms_container,
#         gps_container_name=gps_container_name,
#         encode_mode=encode_mode,
#         waveform_mode=waveform_mode,
#         compute_nasc=False,
#         compute_mvbs=False,
#         colormap='ocean_r',
#         mask_transient_noise=None,
#         mask_impulse_noise=impulse_noise_opts,
#         mask_attenuated_signal=None,
#         remove_background_noise=background_noise_opts,
#         chunks_denoising=chunks_denoising,
#         apply_seabed_mask=True,
#     )
#
#     print(payload)
#
#     client.close()
#     cluster.close()
