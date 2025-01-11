import logging
import os
import sys
import time
import traceback

import numpy as np
import xarray as xr

from dotenv import load_dotenv
from prefect import flow, task
from dask.distributed import Client
from prefect_dask import DaskTaskRunner, get_dask_client
from prefect.cache_policies import Inputs
from prefect.artifacts import create_link_artifact, create_markdown_artifact
from prefect.states import Completed

from saildrone.process import apply_corrections_ds
from saildrone.process.plot import plot_noise_mask, plot_sv_data
from saildrone.process.concat import merge_location_data, optimize_zarr_store, concatenate_and_rechunk, \
    cleanup_temp_folders
from saildrone.denoise import get_impulse_noise_mask, get_attenuation_mask, create_multichannel_mask
from saildrone.store import (PostgresDB, SurveyService, FileSegmentService, open_zarr_store,
                             upload_folder_to_blob_storage, save_dataset_to_netcdf,
                             save_zarr_store, generate_container_name, ensure_container_exists,
                             generate_container_access_url)

load_dotenv()

input_cache_policy = Inputs()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

DASK_CLUSTER_ADDRESS = os.getenv('DASK_CLUSTER_ADDRESS')
CONVERTED_CONTAINER_NAME = os.getenv('CONVERTED_CONTAINER_NAME')
PROCESSED_CONTAINER_NAME = os.getenv('PROCESSED_CONTAINER_NAME')
CHUNKS = {"ping_time": 1000, "range_sample": -1}
BATCH_SIZE = os.getenv('BATCH_SIZE_FOR_EXPORT', 10)


def get_files_by_cruise_id(cruise_id, coordinates=None):
    with PostgresDB() as db_connection:
        survey_service = SurveyService(db_connection)
        survey_id = survey_service.get_survey_by_cruise_id(cruise_id)

        if survey_id is None:
            raise ValueError(f'Survey with cruise_id {cruise_id} not found.')

        polygon = f"POLYGON(({', '.join([f'{lon} {lat}' for lon, lat in coordinates])}))"
        file_service = FileSegmentService(db_connection)
        files = file_service.get_files_by_polygon_and_survey(polygon, survey_id)

        if not files:
            raise ValueError("No files found matching the given criteria.")

        return files


@task(
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    cache_policy=input_cache_policy,
    retry_jitter_factor=0.1,
    refresh_cache=True,
    result_storage=None,
    task_run_name="process-{file_name}",
)
def process_single_file(file, file_name, source_container_name, chunks, path_template, file_index):
    """
    Process a single file for Dask Futures: open it, merge location data, save the dataset, and return its path and frequency category.
    """
    location, _, file_id, location_data, file_freqs, file_start_time, file_end_time = file
    print(f"Processing file {location} with frequencies {file_freqs}")

    # Open the Zarr store lazily with Dask
    ds = open_zarr_store(location, container_name=source_container_name, chunks=chunks)

    # Merge location data
    ds = merge_location_data(ds, location_data)

    # Save the dataset to a temporary Zarr store
    category = "short_pulse" if file_freqs == "38000.0,200000.0" else "long_pulse" if file_freqs == "38000.0" else "exported_ds"
    temp_path = f"{path_template}/{category}_file_{file_index}.zarr"

    print('Writing to', temp_path)
    ds.to_zarr(temp_path, mode="w")

    optimize_zarr_store(temp_path)

    return temp_path, category


@task(
    task_run_name="process-batch-{batch_index}",
)
def process_batch(batch_files, source_container_name, chunks, batch_index, path_template):
    """
    Submit individual file processing as futures and return the results.
    """
    futures = []

    for idx, file in enumerate(batch_files):
        future = process_single_file.submit(file, file[1], source_container_name, chunks, path_template, idx)
        futures.append(future)

    results = {
        "short_pulse": [],
        "long_pulse": [],
        "exported_ds": []
    }

    # Collect results as they complete
    for future in futures:
        try:
            temp_path, category = future.result()
            results[category].append(temp_path)
        except Exception as e:
            print(f"Error processing file: {e}")

    return results


def concatenate_zarr_files(files, source_container_name, chunks=None, batch_size=10, path_template="/tmp/oceanstream"):
    temp_paths = {
        "short_pulse": [],
        "long_pulse": [],
        "exported_ds": []
    }

    futures = []
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        future = process_batch.submit(batch_files, source_container_name, chunks, i, path_template)
        futures.append(future)

    for future in futures:
        batch_results = future.result()
        print('Batch results:', batch_results)
        # Accumulate paths
        temp_paths["short_pulse"].extend(batch_results["short_pulse"])
        temp_paths["long_pulse"].extend(batch_results["long_pulse"])
        temp_paths["exported_ds"].extend(batch_results["exported_ds"])

    short_pulse_ds = concatenate_and_rechunk(temp_paths["short_pulse"], chunks=chunks) if temp_paths["short_pulse"] else None
    long_pulse_ds = concatenate_and_rechunk(temp_paths["long_pulse"], chunks=chunks) if temp_paths["long_pulse"] else None
    exported_ds = concatenate_and_rechunk(temp_paths["exported_ds"], chunks=chunks) if temp_paths["exported_ds"] else None

    # Cleanup temporary folders
    cleanup_temp_folders(temp_paths["short_pulse"] + temp_paths["long_pulse"] + temp_paths["exported_ds"])

    return short_pulse_ds, long_pulse_ds, exported_ds


def export_processed_data_task(cruise_id: str, coordinates=None, container_name=None, filters=None, export_format='netcdf'):
    files = get_files_by_cruise_id(cruise_id, coordinates)
    # ensure_container_exists(container_name, public_access='container')
    #
    # short_pulse_ds, long_pulse_ds, exported_ds = concatenate_zarr_files(files,
    #                                                                     source_container_name=PROCESSED_CONTAINER_NAME,
    #                                                                     chunks=CHUNKS)
    #
    # print('Concatenated files:', short_pulse_ds, long_pulse_ds, exported_ds)
    # # if export_format == 'netcdf':
    # sv_dataset_list = []
    # if short_pulse_ds:
    #     save_dataset_to_netcdf(short_pulse_ds, container_name=container_name, ds_path="short_pulse_data.nc")
    #     save_zarr_store(short_pulse_ds, container_name=container_name, zarr_path="short_pulse_data.zarr")
    #     sv_dataset_list.append("short_pulse_data")
    #
    # if long_pulse_ds:
    #     save_dataset_to_netcdf(long_pulse_ds, container_name=container_name, ds_path="long_pulse_data.nc")
    #     save_zarr_store(long_pulse_ds, container_name=container_name, zarr_path="long_pulse_data.zarr")
    #     sv_dataset_list.append("long_pulse_data")
    #
    # if exported_ds:
    #     save_dataset_to_netcdf(exported_ds, container_name=container_name, ds_path="exported_data.nc")
    #     save_zarr_store(exported_ds, container_name=container_name, zarr_path="exported_data.zarr")
    #     sv_dataset_list.append("exported_data")
    #
    # access_link = generate_container_access_url(container_name)
    # create_link_artifact(
    #     key=f"{container_name}-link",
    #     link=access_link,
    #     link_text="Export link",
    #     description="Link to download the exported data."
    # )
    #
    # future = plot_sv_data_task.submit(sv_dataset_list, container_name=container_name)
    # future.result()

    # if "mask_transient_noise" in filters:
    #     params = filters["mask_transient_noise"]
    #     try:
    #         short_pulse_ds_denoised = apply_mask_transient_noise(short_pulse_ds, params, chunk_dict=CHUNKS)
    #         long_pulse_ds_denoised = apply_mask_transient_noise(long_pulse_ds, params, chunk_dict=CHUNKS)
    #         exported_ds_denoised = apply_mask_transient_noise(exported_ds, params, chunk_dict=CHUNKS)
    #
    #         sv_list_denoised = [
    #             {f"{cruise_id}--short-pulse-transient": short_pulse_ds_denoised},
    #             {f"{cruise_id}--long-pulse-transient": long_pulse_ds_denoised},
    #             {f"{cruise_id}--exported-data-transient": exported_ds_denoised}
    #         ]
    #         plot_sv_data_task(sv_list_denoised, container_name=container_name)
    #     except Exception as e:
    #         print(f'Error applying mask_transient_noise: {e}')

    if "mask_impulse_noise" in filters:
        params = filters["mask_impulse_noise"]
        future = apply_mask_impulse_noise.submit(sv_dataset_list, container_name=container_name, params=params)
        future.result()

    if "mask_attenuated_signal" in filters:
        params = filters["mask_attenuated_signal"]
        try:
            short_pulse_ds_denoised = apply_mask_attenuated_signal(short_pulse_ds, params)
            long_pulse_ds_denoised = apply_mask_attenuated_signal(long_pulse_ds, params)
            exported_ds_denoised = apply_mask_attenuated_signal(exported_ds, params)

            sv_list_denoised = [
                {f"{cruise_id}--short-pulse-att-signal": short_pulse_ds_denoised},
                {f"{cruise_id}--long-pulse-att-signal": long_pulse_ds_denoised},
                {f"{cruise_id}--exported-data-att-signal": exported_ds_denoised}
            ]
            plot_sv_data_task(sv_list_denoised, container_name=container_name)
        except Exception as e:
            print(f'Error applying mask_attenuated_signal: {e}')

    if "background_noise" in filters:
        params = filters["background_noise"]

        try:
            short_pulse_ds_denoised = apply_remove_background_noise(short_pulse_ds, params)
            long_pulse_ds_denoised = apply_remove_background_noise(long_pulse_ds, params)
            exported_ds_denoised = apply_remove_background_noise(exported_ds, params)

            sv_list_denoised = [
                {f"{cruise_id}--short-pulse-background": short_pulse_ds_denoised},
                {f"{cruise_id}--long-pulse-background": long_pulse_ds_denoised},
                {f"{cruise_id}--exported-data-background": exported_ds_denoised}
            ]
            plot_sv_data_task(sv_list_denoised, container_name=container_name)
        except Exception as e:
            print(f'Error applying remove_background_noise: {e}')

    return short_pulse_ds, long_pulse_ds, exported_ds


@task(
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    retry_jitter_factor=0.1,
    refresh_cache=True,
    task_run_name="plot-echograms"
)
def plot_sv_data_task(sv_path=None, container_name=None, file_name=None):
    output_path = f'/tmp/oceanstream/echograms/{container_name}'
    os.makedirs(output_path, exist_ok=True)

    try:
        if isinstance(sv_path, list):
            for sv_item in sv_path:
                print('Plotting echogram:', sv_item)
                ds_Sv = open_zarr_store(f'{sv_item}.zarr', container_name=container_name, chunks=CHUNKS)
                plot_sv_data(ds_Sv, file_base_name=sv_item, output_path=output_path, depth_var='depth')
        elif sv_path:
            ds_Sv = open_zarr_store(sv_path, container_name=container_name, chunks=CHUNKS)
            plot_sv_data(ds_Sv, file_name, output_path=output_path, depth_var='depth')

        print(f"Uploading echograms to blob storage: {container_name}")
        upload_folder_to_blob_storage(output_path, container_name, 'echograms')
    except Exception as e:
        print(f'Error plotting echograms: {e}')
        traceback.print_exc()


@task(
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    retry_jitter_factor=0.1,
    refresh_cache=True,
    task_run_name="apply-mask-transient-noise"
)
def apply_mask_transient_noise(ds_Sv, parameters, chunk_dict):
    if not ds_Sv:
        return None

    from echopype.clean import mask_transient_noise

    print(f"Applying mask_transient_noise with parameters: {parameters}")
    ds_Sv = mask_transient_noise(
        ds_Sv=ds_Sv,
        func=parameters.get("func", "nanmean"),
        depth_bin=parameters.get("depth_bin", "10m"),
        num_side_pings=parameters.get("num_side_pings", 25),
        exclude_above=parameters.get("exclude_above", "250.0m"),
        transient_noise_threshold=parameters.get("transient_noise_threshold", "12.0dB"),
        range_var=parameters.get("range_var", "echo_range"),
        use_index_binning=parameters.get("use_index_binning", False),
        chunk_dict=chunk_dict
    )

    return ds_Sv


@task(
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    retry_jitter_factor=0.1,
    refresh_cache=True,
    task_run_name="apply-mask-impulse-noise"
)
def apply_mask_impulse_noise(sv_dataset_list, container_name, params):
    try:
        output_path = f'/tmp/oceanstream/echograms/{container_name}'
        os.makedirs(output_path, exist_ok=True)

        for sv_item in sv_dataset_list:
            ds_Sv = open_zarr_store(f'{sv_item}.zarr', container_name=container_name, chunks=CHUNKS)
            sv_item_denoised = run_impulse_noise_masking(ds_Sv, params)
            plot_sv_data(sv_item_denoised, file_base_name=sv_item, output_path=output_path, depth_var='depth')

        upload_folder_to_blob_storage(output_path, container_name, 'echograms')
    except Exception as e:
        print(f'Error applying mask_impulse_noise: {e}')
        stack_trace = traceback.format_exc()
        markdown_report = f"""# Report for apply_mask_impulse_noise"""
        markdown_report += f"\n\nError applying mask_impulse_noise: {e}"
        markdown_report += f"\n\n{stack_trace}"
        create_markdown_artifact(markdown_report)


def run_impulse_noise_masking(ds_Sv, params):
    from echopype.clean import mask_impulse_noise

    print(f"Applying mask_transient_noise with parameters: {params}")
    denoised_sv = mask_impulse_noise(
        ds_Sv=ds_Sv,
        depth_bin=params.get("depth_bin", "5m"),
        num_side_pings=params.get("num_side_pings", 2),
        impulse_noise_threshold=params.get("impulse_noise_threshold", "10.0dB"),
        range_var=params.get("range_var", "depth"),
        use_index_binning=params.get("use_index_binning", False)
    )

    ds_Sv['Sv'] = denoised_sv

    return ds_Sv


@task(
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    retry_jitter_factor=0.1,
    refresh_cache=True,
    task_run_name="apply-mask-attenuated-signal"
)
def apply_mask_attenuated_signal(ds_Sv, params):
    if not ds_Sv:
        return None

    from echopype.clean import mask_attenuated_signal

    print(f"Applying mask_transient_noise with parameters: {params}")
    denoised_sv = mask_attenuated_signal(
        ds_Sv=ds_Sv,
        upper_limit_sl=params.get("upper_limit_sl", "400.0m"),
        lower_limit_sl=params.get("lower_limit_sl", "500.0m"),
        num_side_pings=params.get("num_side_pings", 15),
        attenuation_signal_threshold=params.get("attenuation_signal_threshold", "8.0dB"),
        range_var=params.get("range_var", "depth")
    )

    ds_Sv['Sv'] = denoised_sv

    return ds_Sv


@task(
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    retry_jitter_factor=0.1,
    refresh_cache=True,
    task_run_name="remove-background-noise"
)
def apply_remove_background_noise(ds_Sv, params):
    if not ds_Sv:
        return None

    from echopype.clean import remove_background_noise

    print(f"Applying mask_transient_noise with parameters: {params}")
    ds_Sv = remove_background_noise(
        ds_Sv=ds_Sv,
        ping_num=params.get("ping_num", 10),
        range_sample_num=params.get("range_sample_num", 50),
        background_noise_max=params.get("background_noise_max", None),
        SNR_threshold=params.get("SNR_threshold", "3.0dB")
    )

    return ds_Sv


@flow(task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS))
def export_processed(cruise_id, coordinates=None, filters=None, batch_size=10, export_format='zarr', depth_offset=None, container_name=None):
    if not coordinates:
        raise ValueError("Coordinates are required for spatial queries.")

    # files = get_files_by_cruise_id(cruise_id, coordinates)
    sv_dataset_list = []
    if container_name is None:
        container_name = generate_container_name(cruise_id)

    ensure_container_exists(container_name, public_access='container')

    short_pulse_ds = open_zarr_store('short_pulse_data.zarr', container_name=container_name, chunks=CHUNKS)
    sv_data, sv_denoised = process_sv_dataset(short_pulse_ds, container_name, filters, 'short_pulse', depth_offset)
    if sv_data:
        sv_dataset_list.append("short_pulse_data")

    # short_pulse_ds, long_pulse_ds, exported_ds = concatenate_zarr_files(
    #     files,
    #     source_container_name=PROCESSED_CONTAINER_NAME,
    #     batch_size=batch_size,
    #     chunks=CHUNKS)

    # if export_format == 'netcdf':

    if short_pulse_ds:
        pass
        # sv_data, sv_denoised = process_sv_dataset(short_pulse_ds, container_name, filters, 'short_pulse', depth_offset)
        # if sv_data:
        #     sv_dataset_list.append("short_pulse_data")

        # if depth_offset is not None:
        #     short_pulse_ds = apply_corrections_ds(short_pulse_ds, depth_offset=depth_offset)
        #
        # if "mask_impulse_noise" in filters:
        #     params = filters["mask_impulse_noise"]
        #     # future = apply_mask_impulse_noise.submit(sv_dataset_list, container_name=container_name, params=params)
        #     apply_mask_impulse_noise(sv_dataset_list, container_name=container_name, params=params)
        #
        # save_dataset_to_netcdf(short_pulse_ds, container_name=container_name, ds_path="short_pulse_data.nc")
        # save_zarr_store(short_pulse_ds, container_name=container_name, zarr_path="short_pulse_data.zarr")
    # if long_pulse_ds:
    #     sv_data, sv_denoised = process_sv_dataset(long_pulse_ds, container_name, filters, 'long_pulse', depth_offset)
    #     if sv_data:
    #         sv_dataset_list.append("long_pulse_data")

    # if long_pulse_ds:
    #     if depth_offset is not None:
    #         long_pulse_ds = apply_corrections_ds(long_pulse_ds, depth_offset=depth_offset)
    #
    #     save_dataset_to_netcdf(long_pulse_ds, container_name=container_name, ds_path="long_pulse_data.nc")
    #     save_zarr_store(long_pulse_ds, container_name=container_name, zarr_path="long_pulse_data.zarr")
    #     sv_dataset_list.append("long_pulse_data")

    # if exported_ds:
    #     if depth_offset is not None:
    #         exported_ds = apply_corrections_ds(exported_ds, depth_offset=depth_offset)
    #     save_dataset_to_netcdf(exported_ds, container_name=container_name, ds_path="exported_data.nc")
    #     save_zarr_store(exported_ds, container_name=container_name, zarr_path="exported_data.zarr")
    #     sv_dataset_list.append("exported_data")

    # access_link = generate_container_access_url(container_name)
    # create_link_artifact(
    #     key=f"{container_name}-link",
    #     link=access_link,
    #     link_text="Export link",
    #     description="Link to download the exported data."
    # )

    plot_sv_data_task(sv_dataset_list, container_name=container_name)
    # future = plot_sv_data_task.submit(sv_dataset_list, container_name=container_name)
    # future.wait()


def process_sv_dataset(ds, container_name, filters, ds_name, depth_offset=None):
    from echopype.mask import apply_mask

    os.makedirs(f"/tmp/oceanstream/echograms/{container_name}", exist_ok=True)
    corrected_ds = ds
    corrected_ds_denoised = None

    # if depth_offset is not None:
    #     corrected_ds = apply_corrections_ds(ds, depth_offset=depth_offset)

    # save_dataset_to_netcdf(corrected_ds, container_name=container_name, ds_path=f"{ds_name}_data.nc")
    # save_zarr_store(corrected_ds, container_name=container_name, zarr_path=f"{ds_name}_data.zarr")

    """
    if "mask_impulse_noise" in filters:
        params = filters["mask_impulse_noise"]
        mask_channels = []

        for channel in corrected_ds.coords["channel"].values:
            idx = corrected_ds.channel.values.tolist().index(channel)
            impulse_noise_mask = get_impulse_noise_mask(corrected_ds, params, desired_channel=channel)
            plot_noise_mask(impulse_noise_mask, f'{ds_name}_impulse_denoised_{idx}',
                            echogram_path=f"/tmp/oceanstream/echograms/{container_name}")
            mask_channels.append(impulse_noise_mask)
            print("Number of valid mask points:", np.sum(impulse_noise_mask.values))
            print("Mask contains all False:", np.all(~impulse_noise_mask.values))

        multi_channel_mask = create_multichannel_mask(mask_channels, corrected_ds)
        corrected_ds_impulse_denoised = apply_mask(corrected_ds, multi_channel_mask, var_name="Sv")
        # save_dataset_to_netcdf(corrected_ds_impulse_denoised, container_name=container_name,
        #                        ds_path=f"{ds_name}_impulse_denoised.nc")
        save_zarr_store(corrected_ds_impulse_denoised, container_name=container_name,
                        zarr_path=f"{ds_name}_impulse_denoised.zarr")
        plot_sv_data_task(f"{ds_name}_impulse_denoised.zarr", container_name=container_name,
                          file_name=f"{ds_name}_impulse_denoised")

    if "mask_attenuated_signal" in filters:
        params = filters["mask_attenuated_signal"]
        mask_channels = []

        for channel in corrected_ds.coords["channel"].values:
            idx = corrected_ds.channel.values.tolist().index(channel)
            attn_signal_mask = get_attenuation_mask(corrected_ds, params, desired_channel=channel)
            plot_noise_mask(attn_signal_mask, f'{ds_name}_attn_mask_{idx}',
                            echogram_path=f"/tmp/oceanstream/echograms/{container_name}")

            # single_channel_ds = corrected_ds.sel(channel=channel)
            # masked_single_channel_ds = apply_mask(single_channel_ds, attn_signal_mask, var_name="Sv")
            mask_channels.append(attn_signal_mask)
            print("Number of valid mask points:", np.sum(attn_signal_mask.values))
            print("Mask contains all False:", np.all(~attn_signal_mask.values))
            # processed_channels.append(masked_single_channel_ds)

        multi_channel_mask = create_multichannel_mask(mask_channels, corrected_ds)
        corrected_ds_attn_denoised = apply_mask(corrected_ds, multi_channel_mask, var_name="Sv")
        # save_dataset_to_netcdf(corrected_ds_attn_denoised, container_name=container_name,
        #                        ds_path=f"{ds_name}_attn_denoised.nc")

        save_zarr_store(corrected_ds_attn_denoised, container_name=container_name,
                        zarr_path=f"{ds_name}_attn_denoised.zarr")
        plot_sv_data_task(f"{ds_name}_attn_denoised.zarr", container_name=container_name,
                          file_name=f"{ds_name}_attn_denoised")
    """
    return corrected_ds, corrected_ds_denoised


if __name__ == "__main__":
    client = Client(address=DASK_CLUSTER_ADDRESS)
    print('Dask client connected to ', DASK_CLUSTER_ADDRESS, client)

    try:
        export_processed.serve(
            name='export-processed-data',
            parameters={
                'cruise_id': '',
                'coordinates': [],
                'filters': {},
                'batch_size': BATCH_SIZE,
                'depth_offset': None,
                'export_format': 'zarr'
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception occurred: {str(e)}', exc_info=True)
        sys.exit(1)
