import pytest
import xarray as xr

from dotenv import load_dotenv
from dask.distributed import Client, LocalCluster

from saildrone.denoise.attenuation_signal import attenuation_mask
from saildrone.denoise.background_noise import background_noise_mask
from saildrone.denoise.mask import extract_channel_and_drop_pings
from saildrone.denoise.transient_noise import transient_noise_mask
from saildrone.denoise.impulse_noise import impulsive_noise_mask

from saildrone.process.plot import ensure_channel_labels, plot_and_upload_echograms, plot_sv_data, plot_sv_channel, \
    plot_mask_channel, plot_all_masks, plot_masks_vertical, export_interactive_echogram

from saildrone.process.workflow import apply_denoising
from saildrone.denoise import build_full_mask, apply_full_mask
from saildrone.store import open_zarr_store, save_zarr_store

from echopype.clean import remove_background_noise


load_dotenv()

GPS_OUTPUT_FOLDER = "./test/gps-processed"


def depth_to_range_sample(ds: xr.Dataset):
    ds = ds.set_coords("range_sample")
    return ds.swap_dims({"depth": "range_sample"})


# switch back so depth is the dimension
def range_sample_to_depth(ds: xr.Dataset):
    return ds.swap_dims({"range_sample": "depth"})


def test_file_workflow_saildrone_full():
    cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit='12GB')
    client = Client(cluster)
    # file_name = '2023-10-08--short_pulse'
    file_name = 'SD_TPOS2023_v03-Phase0-D20231008-T115959-0'
    cruise_id = 'SD_TPOS2023_v03'

    source_container_name = 'export44'
    export_container_name = 'export11'
    chunks = {'ping_time': 2000, 'depth': -1}
    # zarr_path = f"2023-10-08/{file_name}.zarr"
    zarr_path = f"SD_TPOS2023_v03/SD_TPOS2023_v03-Phase0-D20231008-T115959-0/{file_name}.zarr"

    ds = open_zarr_store(zarr_path, cruise_id=None,
                         container_name=source_container_name,
                         chunks=chunks,
                         rechunk_after=True)

    impulse_noise_opts = {
        38000: dict(ping_lags=(1, 2), threshold_db=10.0, range_coord="depth", vertical_bin_size="2m",
                    exclude_shallow_above=5.0),
        200000: dict(ping_lags=(1,), threshold_db=10.0, range_coord="depth", vertical_bin_size="2m",
                     exclude_shallow_above=4.0),
    }
    attenuated_signal_opts = {
        38000: dict(
            upper_limit_sl=400,
            lower_limit_sl=550,
            num_side_pings=10,
            threshold=-3.0,
            range_coord="depth"
        ),
        200000: dict(
            upper_limit_sl=20,
            lower_limit_sl=120,
            num_side_pings=15,
            threshold=-2.5,
            range_coord="depth"
        ),
    }

    transient_noise_opts = {
        38000: dict(
            range_coord="depth",
            ping_window=5,
            range_window=3,
            threshold=12.0,
            percentile=15,
            exclude_above=2.0,
        ),
        200000: dict(
            range_coord="depth",
            ping_window=5,
            range_window=3,
            threshold=12.0,
            percentile=15,
            exclude_above=2.0,
        ),
    }

    background_noise_opts = {
        "38000": dict(
            range_coord="depth",
            range_window=15,
            ping_window=50,
            background_noise_max="-130.0dB",
            SNR_threshold="6.0dB",
            sound_absorption=0.0022,  # 9 ×10⁻⁶ dB m⁻¹ (≈ 0.009 dB km⁻¹)
        ),
        "200000": dict(
            range_coord="depth",
            range_window=50,
            ping_window=50,
            background_noise_max="-120.0dB",
            SNR_threshold="6.0dB",
            sound_absorption=0.001,  # 3.8 ×10⁻⁴ dB m⁻¹ (≈ 0.38 dB km⁻¹)
        ),
    }

    plot_sv_data(ds,
                 file_base_name=file_name,
                 output_path=f'./test/processed/echograms',
                 )

    n_ch = ds.dims["channel"]
    for ch in range(n_ch):
        html_path = f"echogram_{ch}.html"
        export_interactive_echogram(ds, ch, out_html=f"./test/processed/{html_path}")

    ds_masked = apply_denoising(ds,
                                mask_impulse_noise=impulse_noise_opts,
                                mask_attenuated_signal=attenuated_signal_opts,
                                mask_transient_noise=None,
                                remove_background_noise=None
                                )

    # ds = ds.set_coords("range_sample").swap_dims({"depth": "range_sample"})
    #
    def _mask_one(ch_ds):
        freq = str(int(ch_ds["frequency_nominal"]))
        opts = background_noise_opts.get(freq)

        return remove_background_noise(
            ch_ds,
            ping_num=opts["ping_window"],
            range_sample_num=opts["range_window"],
            SNR_threshold=opts["SNR_threshold"],
            background_noise_max=opts["background_noise_max"],
        )["Sv"]

    # this returns a DataArray with dims ("channel","ping_time",...)
    sv_clean = ds_masked.groupby("channel").map(_mask_one)
    sv_clean.name = "Sv"

    # overwrite in-place
    ds_masked["Sv"] = sv_clean

    n_ch = ds_masked.dims["channel"]
    for ch in range(n_ch):
        ch_ds = extract_channel_and_drop_pings(
            ds_masked,
            channel=ch,
            drop_threshold=1.0,
            freq_coord="frequency_nominal"
        )

        plot_sv_data(ch_ds,
                     channel=ch,
                     cmap='ocean_r',
                     output_path=f'./test/processed/echograms',
                     title_template="{channel_label} / denoised and pruned",
                     file_base_name=file_name + '--denoised-pruned'
                     )

    # Run noise removal
    # n_ch = ds.dims["channel"]
    # for ch in range(n_ch):
    #     ch_ds = ds.isel(channel=ch)
    #     freq = int(ch_ds["frequency_nominal"].values.item())
    #     opts = background_noise_opts.get(freq, None)
    #     sound_absorption = opts.get("sound_absorption", 0.001)
    #     SNR_threshold = opts.get("SNR_threshold", "4dB")
    #     ping_num = opts.get("ping_window", 10)
    #     range_sample_num = opts.get("range_window", 30)
    #     background_noise_max = opts.get("background_noise_max", -120.0)
    #
    #     print('freq:', opts)
    #
    #     ch_ds = ch_ds.assign(sound_absorption=sound_absorption)
    #     ds_masked = remove_background_noise(
    #         ch_ds, ping_num=ping_num, range_sample_num=range_sample_num, SNR_threshold=SNR_threshold, \
    #         background_noise_max=background_noise_max
    #     )
    #     plot_sv_data(ds_masked,
    #                  channel=ch,
    #                  output_path=f'./test/processed/echograms',
    #                  title_template="{channel_label} / denoised and pruned",
    #                  file_base_name=file_name + '--denoised-pruned'
    #                  )

    # ds_masked = ds_masked.set_coords("depth").swap_dims({"range_sample": "depth"})

    # ds_38khz = extract_channel_and_drop_pings(ds_masked, channel=38000, drop_threshold=0.9)
    # plot_sv_data(ds_38khz,
    #              output_path=f'./test/processed/echograms',
    #              title_template="{channel_label} / denoised and pruned",
    #              file_base_name=file_name + '--denoised-pruned'
    #              )

    plot_sv_data(ds_masked,
                 output_path=f'./test/processed/echograms',
                 title_template="{channel_label} / denoised and masked",
                 file_base_name=file_name + '--denoised'
                 )

    # plot_masks_vertical(ds_masked, file_base_name=file_name + '--noise', output_path=f'./test/processed/echograms')

    try:
        client.close()
        cluster.close()
    except Exception as e:
        pass


@pytest.mark.skip(reason="Temp")
def test_file_workflow_saildrone():
    cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit='12GB')
    client = Client(cluster)
    # file_name = '2023-10-08--short_pulse'
    file_name = 'SD_TPOS2023_v03-Phase0-D20231008-T115959-0'
    cruise_id = 'SD_TPOS2023_v03'

    source_container_name = 'export44'
    export_container_name = 'export11'
    chunks = {'ping_time': 2000, 'depth': -1}
    # zarr_path = f"2023-10-08/{file_name}.zarr"
    zarr_path = f"SD_TPOS2023_v03/SD_TPOS2023_v03-Phase0-D20231008-T115959-0/{file_name}.zarr"

    ds = open_zarr_store(zarr_path, cruise_id=None,
                         container_name=source_container_name,
                         chunks=chunks,
                         rechunk_after=True)

    impulse_noise_opts = {
        38000: dict(ping_lags=(1, 2), threshold_db=10.0, range_coord="depth", vertical_bin_size="2m",
                    exclude_shallow_above=5.0),
        200000: dict(ping_lags=(1,), threshold_db=10.0, range_coord="depth", vertical_bin_size="2m",
                     exclude_shallow_above=4.0),
    }
    attenuated_signal_opts = {
        38000: dict(
            upper_limit_sl=400,
            lower_limit_sl=550,
            num_side_pings=10,
            threshold=-3.0,
            range_coord="depth"
        ),
        200000: dict(
            upper_limit_sl=20,
            lower_limit_sl=120,
            num_side_pings=15,
            threshold=-2.5,
            range_coord="depth"
        ),
    }

    transient_noise_opts = {
        38000: dict(
            range_coord="depth",
            ping_window=5,
            range_window=3,
            threshold=12.0,
            percentile=15,
            exclude_above=2.0,
        ),
        200000: dict(
            range_coord="depth",
            ping_window=5,
            range_window=3,
            threshold=12.0,
            percentile=15,
            exclude_above=2.0,
        ),
    }

    background_noise_opts = {
        38000: dict(
            range_coord="depth",
            range_window=5,
            ping_window=20,
            background_noise_max=-125.0,
            SNR_threshold=3.0,
            sound_absorption=9e-6,  # 9 ×10⁻⁶ dB m⁻¹ (≈ 0.009 dB km⁻¹)
        ),
        200000: dict(
            range_coord="depth",
            range_window=5,
            ping_window=20,
            background_noise_max=-125.0,
            SNR_threshold=3.0,
            sound_absorption=3.8e-4,  # 3.8 ×10⁻⁴ dB m⁻¹ (≈ 0.38 dB km⁻¹)
        ),
    }

    plot_sv_data(ds,
                 file_base_name=file_name,
                 output_path=f'./test/processed/echograms',
                 )

    stages = {
        "signal-attenuation": {
            "fn": attenuation_mask,
            "param_sets": attenuated_signal_opts
        },
        "impulsive": {
            "fn": impulsive_noise_mask,
            "param_sets": impulse_noise_opts,
        },
        "transient": {
            "fn": transient_noise_mask,
            "param_sets": transient_noise_opts,
        },
        "background": {
            "fn": background_noise_mask,
            "param_sets": background_noise_opts,
        }
    }

    full_mask, stage_cubes = build_full_mask(ds, stages=stages, return_stage_masks=True)
    ds_masked = apply_full_mask(ds, full_mask)
    # ds_pruned = apply_full_mask(ds, full_mask, drop_pings=True)

    plot_sv_data(ds_masked,
                 output_path=f'./test/processed/echograms',
                 title_template="{channel_label} / denoised",
                 file_base_name=file_name + '--denoised'
                 )

    # plot_sv_data(ds_pruned,
    #              output_path=f'./test/processed/echograms',
    #              title_template="{channel_label} / pruned",
    #              file_base_name=file_name + '--pruned'
    #              )

    mask_dict = {"full": full_mask, **dict(stage_cubes)}
    print(mask_dict)
    plot_masks_vertical(mask_dict, ds,
                        file_base_name=file_name + '--noise',
                        output_path=f'./test/processed/echograms')

    # 1. Give each DataArray a clear, unique name
    mask_vars = {
        f"mask_{key.replace(' ', '_').lower()}": arr.astype("bool")
        for key, arr in mask_dict.items()
    }

    # 2. Broadcast each mask to the Dataset’s shape
    mask_vars = {
        name: arr.broadcast_like(ds["Sv"]).assign_attrs(
            long_name=f"{key} quality-control mask (True = bad)"
        )
        for (name, arr), (key, _) in zip(mask_vars.items(), mask_dict.items())
    }

    # 3. Merge into a *new* Dataset to avoid modifying ds in-place accidentally
    ds_masked = ds.merge(mask_vars, compat="no_conflicts")
    print('[ ds_masked ]: \n', ds_masked)

    try:
        client.close()
        cluster.close()
    except Exception as e:
        pass
