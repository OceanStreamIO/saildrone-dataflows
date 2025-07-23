import logging
import os
import shutil
import sys
import traceback
import dask

from collections import defaultdict
from datetime import datetime, timedelta
from prefect.deployments import run_deployment
from dotenv import load_dotenv

from prefect import flow, task
from prefect.cache_policies import Inputs
from prefect.futures import as_completed, PrefectFuture

from saildrone.denoise.mask import extract_channel_and_drop_pings
from saildrone.process import plot_and_upload_echograms, apply_denoising, plot_sv_data
from saildrone.process.concat import concatenate_and_rechunk
from saildrone.process.plot import plot_and_upload_masks
from saildrone.process.workflow import compute_and_save_nasc, compute_and_save_mvbs
from saildrone.store import open_zarr_store, save_dataset_to_netcdf, save_zarr_store
from datetime import datetime
import gc

try:
    import psutil  # lightweight; ships in most distros / containers
except ImportError:
    psutil = None

input_cache_policy = Inputs()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

load_dotenv()

DASK_CLUSTER_ADDRESS = os.getenv('DASK_CLUSTER_ADDRESS')
CATEGORY_CONFIG = {
    "short_pulse": {
        "freq_key": "38000.0,200000.0",
        "zarr_name": "short_pulse{denoised}.zarr",
        "nc_name": "short_pulse{denoised}.nc",
        "file_base": "short_pulse{denoised}",
    },
    "long_pulse": {
        "freq_key": "38000.0",
        "zarr_name": "long_pulse{denoised}.zarr",
        "nc_name": "long_pulse{denoised}.nc",
        "file_base": "long_pulse{denoised}",
    },
    "exported_ds": {
        "freq_key": None,  # catch-all
        "zarr_name": "{batch_key}{denoised}.zarr",
        "nc_name": "{batch_key}{denoised}.nc",
        "file_base": "{batch_key}{denoised}"
    }
}

CHUNKS = {"ping_time": 1000, "depth": -1}
NETCDF_ROOT_DIR = os.getenv('NETCDF_ROOT_DIR', '/tmp/oceanstream/netcdfdata')
DEFAULT_TASK_TIMEOUT = 7_200  # 2 hours
MAX_RUNTIME_SECONDS = 3_300


def _ram_usage_str() -> str:
    """
    Return human-readable RAM usage, e.g. '72.3% (45.7 / 63.1 GB)'.
    Falls back gracefully if psutil is missing.
    """
    if psutil is None:  # psutil not installed → unknown
        return "unknown"
    vm = psutil.virtual_memory()
    return f"{vm.percent:.1f}% ({vm.used / 1e9:.1f} / {vm.total / 1e9:.1f} GB)"


def _log_mem(step: str) -> None:
    """
    Print a timestamped message that includes current RAM load.
    """
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {step} "
          f"| RAM: {_ram_usage_str()}")


def _nav_to_data_vars(ds):
    nav = ["latitude", "longitude", "speed_knots"]
    coords = [v for v in nav if v in ds.coords]

    if coords:
        ds = ds.reset_coords(coords)
    return ds


@task(
    log_prints=True
)
def trigger_denoising_flow(
    zarr_path_source: str,
    zarr_path_output: str,
    container_name: str,
    mask_impulse_noise=None,
    mask_attenuated_signal=None,
    mask_transient_noise=None,
    remove_background_noise=None,
    apply_seabed_mask: bool = False,
    chunks=None
):
    state = run_deployment(
        name="apply-denoising-flow/apply-denoising-flow",
        parameters={
            'zarr_path_source': zarr_path_source,
            'zarr_path_output': zarr_path_output,
            'container_name': container_name,
            'mask_impulse_noise': mask_impulse_noise,
            'mask_attenuated_signal': mask_attenuated_signal,
            'mask_transient_noise': mask_transient_noise,
            'remove_background_noise': remove_background_noise,
            'apply_seabed_mask': apply_seabed_mask,
            'chunks': chunks,
        },
        timeout=None
    )

    return state


@task(
    log_prints=True,
    task_run_name="compute_batch_nasc--{batch_key}"
)
def compute_batch_nasc(batch_results, batch_key, cruise_id, container_name, denoised, compute_nasc_options,
                       plot_echograms=False, save_to_netcdf=False, colormap='ocean_r', chunks=None):
    results = {}

    tag_for = {
        "short_pulse": "short_pulse",
        "long_pulse": "long_pulse",
        "exported_ds": batch_key,
    }

    def _run(pulse, tag):
        root = f"{batch_key}/{batch_key}--{tag}"
        suffix = "--denoised" if denoised else ""

        print('Computing NASC for pulse:', pulse, 'with tag:', tag, f'and root: {root}{suffix}.zarr')
        try:
            ds = open_zarr_store(f"{root}{suffix}.zarr",
                                 container_name=container_name,
                                 rechunk_after=True,
                                 chunks=chunks)
            ds = _nav_to_data_vars(ds)
            print(ds.data_vars)
        except Exception as e:
            logging.error(f"Failed to open Zarr store during compute_batch_nasc for {root}{suffix}.zarr: {e}")
            traceback.print_exc()
            return

        nasc = compute_and_save_nasc(
            ds,
            zarr_path=f"{root}--nasc.zarr",
            compute_nasc_opts=compute_nasc_options,
            cruise_id=cruise_id,
            container_name=container_name,
        )
        results[pulse] = nasc

        if save_to_netcdf:
            save_dataset_to_netcdf(
                nasc,
                container_name=container_name,
                ds_path=f"{root}--nasc.nc",
                base_local_temp_path=NETCDF_ROOT_DIR,
                is_temp_dir=False,
            )

    for pulse, tag in tag_for.items():
        if batch_results.get(pulse):
            _run(pulse, tag)

    return results


@task(
    log_prints=True,
    task_run_name="compute_batch_mvbs--{batch_key}"
)
def compute_batch_mvbs(batch_results, batch_key, cruise_id, container_name, denoised, compute_mvbs_options,
                       plot_echograms=False, save_to_netcdf=False, colormap='ocean_r', chunks=None):
    results = {}

    tag_for = {
        "short_pulse": "short_pulse",
        "long_pulse": "long_pulse",
        "exported_ds": batch_key
    }

    def _run(pulse, tag):
        root = f"{batch_key}/{batch_key}--{tag}"
        suffix = "--denoised" if denoised else ""
        zarr_path = f"{root}{suffix}.zarr"

        print('Computing MVBS for pulse:', pulse, 'with tag:', tag, f'and root: {zarr_path}')
        try:
            ds = open_zarr_store(zarr_path, container_name=container_name, chunks=chunks,
                                 rechunk_after=True)
            ds = _nav_to_data_vars(ds)
            print(ds.data_vars)
        except Exception as e:
            logging.error(f"Failed to open Zarr during compute_batch_mvbs for {zarr_path}: {e}")
            traceback.print_exc()
            return

        ds_mvbs = compute_and_save_mvbs(
            ds,
            cruise_id=cruise_id,
            zarr_path=f"{root}--mvbs.zarr",
            compute_mvbs_opts=compute_mvbs_options,
            container_name=container_name,
        )
        results[pulse] = ds_mvbs

        if plot_echograms:
            plot_and_upload_echograms(
                ds_mvbs,
                file_base_name=f"{tag}--mvbs",
                save_to_blobstorage=True,
                upload_path=f"{batch_key}",
                cmap=colormap,
                container_name=container_name,
                title_template=f"{batch_key} ({tag})" + " | MVBS | {channel_label}",
            )

        if save_to_netcdf:
            save_dataset_to_netcdf(
                ds_mvbs,
                container_name=container_name,
                ds_path=f"{root}--mvbs.nc",
                base_local_temp_path=NETCDF_ROOT_DIR,
                is_temp_dir=False,
            )

    for pulse, tag in tag_for.items():
        if batch_results.get(pulse):
            _run(pulse, tag)

    return results


@task(
    log_prints=True,
    task_run_name="concatenate_batch_files--{batch_key}"
)
def concatenate_batch_files(batch_key, cruise_id, files, container_name, plot_echograms, save_to_netcdf,
                            skip_concatenate_batches, colormap, **kwargs):
    """Run NASC, MVBS, … for one calendar batch."""
    # 1) bucket files by category
    batch_results = {cat: [] for cat in CATEGORY_CONFIG}
    chunks = kwargs.get('chunks', None)
    plot_channels_masked = kwargs.get('plot_channels_masked', [])

    for file_info in files:
        freqs = file_info["file_freqs"]
        for category, cfg in CATEGORY_CONFIG.items():
            if cfg["freq_key"] is None or freqs == cfg["freq_key"]:
                path = f"{cruise_id}/{file_info['file_name']}/{file_info['file_name']}.zarr"
                batch_results[category].append(path)
                break

    def _process_category(cat: str):
        paths = batch_results[cat]

        if not paths:
            return

        section = CATEGORY_CONFIG[cat]
        zarr_path = f"{batch_key}/{batch_key}--{section['zarr_name'].format(batch_key=batch_key, denoised='')}"

        if skip_concatenate_batches is not True:
            _log_mem(f"1) Concatenating files for category {cat} ({len(paths)} paths)")
            ds = concatenate_and_rechunk(paths, container_name=container_name, chunks=chunks)
            _log_mem(f"2) Finished concatenating {cat} dataset")

            # save Zarr
            save_zarr_store(ds, container_name=container_name, zarr_path=zarr_path)
            print(f"Finished saving zarr dataset to:", zarr_path)
            _log_mem("3) Zarr dataset saved")
        else:
            ds = open_zarr_store(zarr_path, container_name=container_name)

        # optional NetCDF
        if save_to_netcdf:
            nc_path = f"{batch_key}/{batch_key}--{section['nc_name'].format(batch_key=batch_key, denoised='')}"
            save_dataset_to_netcdf(
                ds,
                container_name=container_name,
                ds_path=nc_path,
                base_local_temp_path=NETCDF_ROOT_DIR,
                is_temp_dir=False,
            )
            print(f"Finished saving netcdf dataset to:", nc_path)
            _log_mem("4) NetCDF dataset saved")

        if plot_echograms:
            plot_and_upload_echograms(
                ds,
                file_base_name=f"{batch_key}--{section['file_base'].format(batch_key=batch_key, denoised='')}",
                save_to_blobstorage=True,
                upload_path=batch_key,
                cmap=colormap,
                title_template=f"{batch_key} ({cat})" + " | {channel_label}",
                container_name=container_name,
            )
            _log_mem("5) Echograms plotted & uploaded")

        ##############################################################################################################
        print('5) Applying denoising')
        _log_mem("6) Triggering denoising flow")
        zarr_path_denoised = f"{batch_key}/{batch_key}--{section['zarr_name'].format(batch_key=batch_key, denoised='--denoised')}"

        future = trigger_denoising_flow.submit(
            zarr_path_source=zarr_path,
            zarr_path_output=zarr_path_denoised,
            container_name=container_name,
            mask_impulse_noise=kwargs.get('mask_impulse_noise'),
            mask_attenuated_signal=kwargs.get('mask_attenuated_signal'),
            mask_transient_noise=kwargs.get('mask_transient_noise'),
            remove_background_noise=kwargs.get('remove_background_noise'),
            apply_seabed_mask=kwargs.get('apply_seabed_mask'),
            chunks=chunks,
        )
        state = future.result()
        future.wait()

        del ds
        gc.collect()
        _log_mem("7) Source dataset freed from memory after denoising trigger")

        sv_dataset_masked = open_zarr_store(zarr_path_denoised, container_name=container_name)
        print('sv_dataset_masked:', sv_dataset_masked)
        _log_mem("8) Denoised dataset opened")

        if save_to_netcdf:
            nc_file_path_denoised = f"{batch_key}/{batch_key}--{section['nc_name'].format(batch_key=batch_key, denoised='--denoised')}"
            print('6) Saving denoised dataset to NetCDF:', nc_file_path_denoised)
            _log_mem("9) Denoised NetCDF saved")

            save_dataset_to_netcdf(
                sv_dataset_masked,
                container_name=container_name,
                ds_path=nc_file_path_denoised,
                base_local_temp_path=NETCDF_ROOT_DIR,
                is_temp_dir=False,
            )

            print('7) Saved denoised dataset to NetCDF:', nc_file_path_denoised)

        try:
            if plot_echograms:
                plot_and_upload_echograms(
                    sv_dataset_masked,
                    file_base_name=f"{batch_key}--{section['file_base'].format(batch_key=batch_key, denoised='--denoised')}",
                    save_to_blobstorage=True,
                    upload_path=batch_key,
                    cmap=colormap,
                    container_name=container_name,
                    title_template=f"{batch_key} ({cat}, denoised)" + " | {channel_label}",
                )
                print('Plotting masked channels', plot_channels_masked)
                for channel in plot_channels_masked:
                    try:
                        ds_channel = extract_channel_and_drop_pings(
                            sv_dataset_masked, channel=channel, drop_threshold=0.9
                        )
                        print(f"Plotting pruned channel {channel} for {batch_key} ({cat})")
                        plot_and_upload_echograms(
                            ds_channel,
                            file_base_name=f"{batch_key}--{section['file_base'].format(batch_key=batch_key, denoised='--denoised-pruned')}",
                            save_to_blobstorage=True,
                            upload_path=batch_key,
                            cmap=colormap,
                            container_name=container_name,
                            title_template=f"{batch_key} ({cat}, denoised and pruned)" + " | {channel_label}",
                        )
                        _log_mem(f"10) Plotted pruned channel {channel} for {batch_key} ({cat})")
                    except Exception as e:
                        logging.error(f"Failed to plot pruned channel {channel} for {batch_key} ({cat}): {e}")
                        traceback.print_exc()

                _log_mem("10) Denoised echograms & masks plotted")
        except Exception as e:
            logging.error(f"Failed to plot echograms or masks for {cat}: {e}")
            traceback.print_exc()

    ###############################################################################################################
    # 2) run through each category
    for category in CATEGORY_CONFIG:
        _process_category(category)

    logging.info(
        f"Running batch aggregation for key: {batch_key}, with {len(files)} files."
    )
    return batch_results


@flow(log_prints=True)
def concatenate_processed_files(files_list,
                                days_to_combine=1,
                                cruise_id='',
                                container_name='',
                                plot_echograms=False,
                                save_to_netcdf=False,
                                save_nasc_to_netcdf=False,
                                save_mvbs_to_netcdf=False,
                                skip_concatenate_batches=False,
                                colormap='ocean_r',
                                compute_nasc_options=None,
                                compute_mvbs_options=None,
                                mask_impulse_noise=None,
                                mask_attenuated_signal=None,
                                mask_transient_noise=None,
                                remove_background_noise=None,
                                apply_seabed_mask=False,
                                chunks=None,
                                plot_channels_masked=None
                                ):
    by_batch = defaultdict(list)
    in_flight = []
    side_tasks = []
    denoised = (
            mask_impulse_noise
            or mask_attenuated_signal
            or mask_transient_noise
            or remove_background_noise
            or apply_seabed_mask
    )

    for source_path, file_record in files_list:
        ts = datetime.fromisoformat(file_record["file_start_time"])
        key = _batch_key(ts, days_to_combine)
        by_batch[key].append(file_record)

    batch_size = 3  # max number of concurrent tasks
    batches = by_batch.items()
    print(f"Total batches to process: {len(batches)}")

    for key, files in batches:
        print('Processing batch:', key, 'with', len(files), 'files.')
        future = concatenate_batch_files.submit(key, cruise_id, files, container_name, plot_echograms,
                                                save_to_netcdf, skip_concatenate_batches, colormap,
                                                mask_impulse_noise=mask_impulse_noise,
                                                mask_attenuated_signal=mask_attenuated_signal,
                                                mask_transient_noise=mask_transient_noise,
                                                remove_background_noise=remove_background_noise,
                                                apply_seabed_mask=apply_seabed_mask,
                                                chunks=chunks,
                                                plot_channels_masked=plot_channels_masked)

        in_flight.append(future)

        if compute_nasc_options:
            future_nasc_task = compute_batch_nasc.submit(future, key, cruise_id, container_name,
                                                         denoised=denoised,
                                                         compute_nasc_options=compute_nasc_options,
                                                         plot_echograms=plot_echograms,
                                                         save_to_netcdf=save_nasc_to_netcdf,
                                                         colormap=colormap,
                                                         chunks=chunks)
            side_tasks.append(future_nasc_task)

        if compute_mvbs_options:
            future_mvbs_task = compute_batch_mvbs.submit(future, key, cruise_id, container_name,
                                                         denoised=denoised,
                                                         compute_mvbs_options=compute_mvbs_options,
                                                         plot_echograms=plot_echograms,
                                                         save_to_netcdf=save_mvbs_to_netcdf,
                                                         colormap=colormap,
                                                         chunks=chunks)
            side_tasks.append(future_mvbs_task)

        # Throttle when max concurrent tasks reached
        if len(in_flight) >= batch_size:
            finished = next(as_completed(in_flight))
            in_flight.remove(finished)

    # all_futures = agg_in_flight + agg_side_tasks
    for remaining in in_flight:
        remaining.result()

    if os.path.exists(NETCDF_ROOT_DIR):
        shutil.rmtree(NETCDF_ROOT_DIR, ignore_errors=True)

    print("All files have been processed.")


def _batch_key(ts: datetime, width_days: int) -> str:
    """
    Anchor `ts` to the start of its `width`-day window and return a
    filename-safe key.
      width == 1  →  '2023-08-08'
      width >  1  →  '2023-08-08_to_2023-08-10'   (inclusive range)
    """
    anchor = datetime(ts.year, ts.month, ts.day)  # midnight of that day

    if width_days == 1:
        return f"{anchor:%Y-%m-%d}"

    anchor -= timedelta(days=(anchor - datetime.min).days % width_days)
    end = anchor + timedelta(days=width_days - 1)

    return f"{anchor:%Y-%m-%d}_to_{end:%Y-%m-%d}"


if __name__ == "__main__":
    try:
        concatenate_processed_files.serve(
            name='concatenate_processed_files',
            parameters={
                'files_list': [],
                'days_to_combine': 1,
                'cruise_id': '',
                'container_name': '',
                'plot_echograms': True,
                'save_to_netcdf': False,
                'save_nasc_to_netcdf': True,
                'save_mvbs_to_netcdf': True,
                'skip_concatenate_batches': False,
                'colormap': 'ocean_r',
                'compute_nasc_options': None,
                'compute_mvbs_options': None,
                'mask_impulse_noise': None,
                'mask_attenuated_signal': None,
                'mask_transient_noise': None,
                'remove_background_noise': None,
                'apply_seabed_mask': False,
                'chunks': None,
                'plot_channels_masked': []
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception occurred: {str(e)}', exc_info=True)
        sys.exit(1)
