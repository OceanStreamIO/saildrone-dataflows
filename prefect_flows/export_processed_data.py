import logging
import os
import sys
import time
import xarray as xr

import pandas as pd

from typing import Literal
from dotenv import load_dotenv
from prefect import flow, task
from dask.distributed import Client
from prefect_dask import DaskTaskRunner, get_dask_client
from prefect.cache_policies import Inputs
from prefect.artifacts import create_link_artifact
from prefect.states import Completed

from saildrone.process import convert_file_and_save, plot_sv_data
from saildrone.store import (PostgresDB, SurveyService, FileSegmentService, open_zarr_store,
                             upload_folder_to_blob_storage, save_dataset_to_netcdf,
                             save_zarr_store, generate_container_name, ensure_container_exists,
                             generate_container_access_url, create_blob_service_client)

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
CHUNKS = {"ping_time": 500, "range_sample": -1}


@task(
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    retry_jitter_factor=0.1,
    refresh_cache=True,
    task_run_name="export-processed-data"
)
def export_processed_data_task(cruise_id: str, coordinates=None, container_name=None, export_format='netcdf'):
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

        short_pulse = []
        long_pulse = []
        exported_ds_list = []

        with get_dask_client() as client:
            for file in files:
                location, file_name, file_id, location_data, file_freqs, file_start_time, file_end_time = file
                zarr_path = location
    
                # Load the zarr store as an xarray dataset
                ds = open_zarr_store(zarr_path, container_name=PROCESSED_CONTAINER_NAME, chunks=CHUNKS)
    
                # Merge location data into the dataset
                ds = merge_location_data(ds, location_data)
    
                # Categorize datasets by file frequency
                if file_freqs == "38000.0,200000.0":
                    short_pulse.append(ds)
                elif file_freqs == "38000.0":
                    long_pulse.append(ds)
                else:
                    exported_ds_list.append(ds)
    
            ensure_container_exists(container_name)
    
            short_pulse_datasets = [
                ds.rename({"source_filenames": f"source_filenames_{i}"})
                for i, ds in enumerate(short_pulse)
            ]
            short_pulse_ds = xr.merge(short_pulse_datasets) if short_pulse_datasets else None
    
            long_pulse_datasets = [
                ds.rename({"source_filenames": f"source_filenames_{i}"})
                for i, ds in enumerate(long_pulse)
            ]
            long_pulse_ds = xr.merge(long_pulse_datasets) if long_pulse_datasets else None
    
            exported_ds_datasets = [
                ds.rename({"source_filenames": f"source_filenames_{i}"})
                for i, ds in enumerate(exported_ds_list)
            ]
            exported_ds = xr.merge(exported_ds_datasets) if exported_ds_datasets else None
    
            if export_format == 'netcdf':
                if short_pulse_ds:
                    save_dataset_to_netcdf(short_pulse_ds, container_name=container_name, ds_path="short_pulse_data.nc")
    
                if long_pulse_ds:
                    save_dataset_to_netcdf(long_pulse_ds, container_name=container_name, ds_path="long_pulse_data.nc")
    
                if exported_ds:
                    save_dataset_to_netcdf(exported_ds, container_name=container_name, ds_path="exported_data.nc")
            elif export_format == 'zarr':
                if short_pulse_ds:
                    save_zarr_store(short_pulse_ds, container_name=container_name, zarr_path="short_pulse_data.zarr")
    
                if long_pulse_ds:
                    save_zarr_store(long_pulse_ds, container_name=container_name, zarr_path="long_pulse_data.zarr")
    
                if exported_ds:
                    save_zarr_store(exported_ds, container_name=container_name, zarr_path="exported_data.zarr")
    
            access_link = generate_container_access_url(container_name)
            create_link_artifact(
                key=f"{container_name}-link",
                link=access_link,
                link_text="Export link",
                description="Link to download the exported data."
            )
    
            return short_pulse_ds, long_pulse_ds, exported_ds


@task(
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    retry_jitter_factor=0.1,
    refresh_cache=True,
    task_run_name="plot-echograms-and-upload-to-blobstorage"
)
def plot_sv_data_task(ds_Sv=None, container_name=None, file_name=None):
    output_path = f'/tmp/echograms/{container_name}'
    os.makedirs(output_path, exist_ok=True)

    if isinstance(ds_Sv, list):
        for sv_item in ds_Sv:
            for file_name, dataset in sv_item.items():
                if dataset:
                    plot_sv_data(dataset, file_name, output_path=output_path)
    elif ds_Sv:
        plot_sv_data(ds_Sv, file_name, output_path=output_path)

    print(f"Uploading echograms to blob storage: {container_name}")
    upload_folder_to_blob_storage(output_path, container_name, 'echograms')


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
def apply_mask_impulse_noise(ds_Sv, params):
    if not ds_Sv:
        return None

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


def merge_location_data(dataset: xr.Dataset, location_data) -> xr.Dataset:
    """
    Merge location data into the xarray dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        The xarray dataset to update.
    location_data : list
        A list of dictionaries containing location data.

    Returns
    -------
    xr.Dataset
        Updated dataset with location data added.
    """
    # Convert location_data to a Pandas DataFrame
    location_df = pd.DataFrame(location_data)

    # Convert timestamp strings to datetime objects
    location_df['dt'] = pd.to_datetime(location_df['dt'])

    # Create xarray variables from the location data
    dataset['latitude'] = xr.DataArray(location_df['lat'].values, dims='time',
                                       coords={'time': location_df['dt'].values})
    dataset['longitude'] = xr.DataArray(location_df['lon'].values, dims='time',
                                        coords={'time': location_df['dt'].values})
    dataset['speed_knots'] = xr.DataArray(location_df['knt'].values, dims='time',
                                          coords={'time': location_df['dt'].values})

    return dataset


@flow(task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS))
def export_processed(cruise_id='', coordinates=None, filters=None, export_format='zarr'):
    if not coordinates:
        raise ValueError("Coordinates are required for spatial queries.")

    container_name = generate_container_name(cruise_id)

    short_pulse_ds, long_pulse_ds, exported_ds = export_processed_data_task(cruise_id, coordinates, container_name,
                                                                            export_format=export_format)
    sv_list = [
        {f"{cruise_id}--short-pulse": short_pulse_ds},
        {f"{cruise_id}--long-pulse": long_pulse_ds}
    ]

    if exported_ds:
        sv_list.append({f"{cruise_id}--exported-data": exported_ds})

    plot_sv_data_task(sv_list, container_name=container_name)

    if "mask_transient_noise" in filters:
        params = filters["mask_transient_noise"]
        try:
            short_pulse_ds_denoised = apply_mask_transient_noise(short_pulse_ds, params, chunk_dict=CHUNKS)
            long_pulse_ds_denoised = apply_mask_transient_noise(long_pulse_ds, params, chunk_dict=CHUNKS)
            exported_ds_denoised = apply_mask_transient_noise(exported_ds, params, chunk_dict=CHUNKS)

            sv_list_denoised = [
                {f"{cruise_id}--short-pulse-transient": short_pulse_ds_denoised},
                {f"{cruise_id}--long-pulse-transient": long_pulse_ds_denoised},
                {f"{cruise_id}--exported-data-transient": exported_ds_denoised}
            ]
            plot_sv_data_task(sv_list_denoised, container_name=container_name)
        except Exception as e:
            print(f'Error applying mask_transient_noise: {e}')

    if "mask_impulse_noise" in filters:
        params = filters["mask_impulse_noise"]
        try:
            short_pulse_ds_denoised = apply_mask_impulse_noise(short_pulse_ds, params)
            long_pulse_ds_denoised = apply_mask_impulse_noise(long_pulse_ds, params)
            exported_ds_denoised = apply_mask_impulse_noise(exported_ds, params)

            sv_list_denoised = [
                {f"{cruise_id}--short-pulse-impulse": short_pulse_ds_denoised},
                {f"{cruise_id}--long-pulse-impulse": long_pulse_ds_denoised},
                {f"{cruise_id}--exported-data-impulse": exported_ds_denoised}
            ]
            plot_sv_data_task(sv_list_denoised, container_name=container_name)
        except Exception as e:
            print(f'Error applying mask_impulse_noise: {e}')

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


# def process_files(files, container_name):
#     short_pulse = []
#     long_pulse = []
#
#     for file in files:
#         file_freqs = file.get("file_freqs")
#         location = file.get("location")
#
#         # Load the zarr store as an xarray dataset
#         # ds = open_converted(zarr_path, chunks=CHUNKS)
#         ds = xr.open_zarr(location, chunks=CHUNKS)
#
#         if file_freqs == "38000.0,200000.0":
#             short_pulse.append(ds)
#         elif file_freqs == "38000.0":
#             long_pulse.append(ds)
#
#     ensure_container_exists(container_name)
#
#     short_pulse_datasets = [
#         ds.rename({"source_filenames": f"source_filenames_{i}"})
#         for i, ds in enumerate(short_pulse)
#     ]
#     short_pulse_ds = xr.merge(short_pulse_datasets) if short_pulse_datasets else xr.Dataset()
#
#     long_pulse_datasets = [
#         ds.rename({"source_filenames": f"source_filenames_{i}"})
#         for i, ds in enumerate(long_pulse)
#     ]
#
#     long_pulse_ds = xr.merge(long_pulse_datasets) if long_pulse_datasets else xr.Dataset()
#     save_datasets_to_netcdf(short_pulse_ds, long_pulse_ds, container_name)
#
#     access_link = generate_container_access_url(container_name)
#
#     return short_pulse_ds, long_pulse_ds, access_link


if __name__ == "__main__":
    try:
        client = Client(address=DASK_CLUSTER_ADDRESS)
        export_processed.serve(
            name='export-processed-data',
            parameters={
                'cruise_id': '',
                'coordinates': [],
                'filters': {},
                'export_format': 'zarr'
            }
        )

        # cruise_id = "SD_TPOS2023_v03"
        # container_name = generate_container_name(cruise_id)
        #
        # files = [
        #     {
        #         'location': './test/sd-zarr/SD_TPOS2023_v03-Phase0-D20230530-T011150-0.zarr',
        #         'file_name': None,
        #         'file_id': None,
        #         'location_data': None,
        #         'file_freqs': "38000.0,200000.0",
        #         'file_start_time': None,
        #         'file_end_time': None
        #     },
        #     {
        #         'location': './test/sd-zarr/SD_TPOS2023_v03-Phase0-D20230530-T012350-0.zarr',
        #         'file_name': None,
        #         'file_id': None,
        #         'location_data': None,
        #         'file_freqs': "38000.0,200000.0",
        #         'file_start_time': None,
        #         'file_end_time': None
        #     },
        #     {
        #         'location': './test/sd-zarr/SD_TPOS2023_v03-Phase0-D20230530-T013549-0.zarr',
        #         'file_name': None,
        #         'file_id': None,
        #         'location_data': None,
        #         'file_freqs': "38000.0,200000.0",
        #         'file_start_time': None,
        #         'file_end_time': None
        #     },
        #     {
        #         'location': './test/sd-zarr/SD_TPOS2023_v03-Phase0-D20230530-T014750-0.zarr',
        #         'file_name': None,
        #         'file_id': None,
        #         'location_data': None,
        #         'file_freqs': "38000.0,200000.0",
        #         'file_start_time': None,
        #         'file_end_time': None
        #     },
        #     {
        #         'location': './test/sd-zarr/SD_TPOS2023_v03-Phase0-D20230530-T015950-0.zarr',
        #         'file_name': None,
        #         'file_id': None,
        #         'location_data': None,
        #         'file_freqs': "38000.0,200000.0",
        #         'file_start_time': None,
        #         'file_end_time': None
        #     },
        #     {
        #         'location': './test/sd-zarr/SD_TPOS2023_v03-Phase0-D20230530-T021150-0.zarr',
        #         'file_name': None,
        #         'file_id': None,
        #         'location_data': None,
        #         'file_freqs': "38000.0,200000.0",
        #         'file_start_time': None,
        #         'file_end_time': None
        #     }
        # ]
        #
        # short_pulse_ds, long_pulse_ds, access_link = process_files(files, container_name=container_name)
        # print(f"Access link: {access_link}")
        #
        # sv_list = [
        #     {f"{cruise_id}--short-pulse": short_pulse_ds},
        #     {f"{cruise_id}--long-pulse": long_pulse_ds}
        # ]

        # plot_sv_data_task(sv_list, container_name=container_name)
        #
        # short_pulse_ds_denoised = apply_mask_impulse_noise(short_pulse_ds, params={
        #     'range_var': 'echo_range',
        #     'use_index_binning': True
        # })
        # long_pulse_ds_denoised = apply_mask_impulse_noise(long_pulse_ds, params={
        #     'range_var': 'echo_range',
        #     'use_index_binning': True
        # })
        # print('Applied mask_transient_noise. Uploading denoised echograms...')
        #
        # sv_list_denoised = [
        #     {f"{cruise_id}--short-pulse-transient": short_pulse_ds_denoised},
        #     {f"{cruise_id}--long-pulse-transient": long_pulse_ds_denoised}
        # ]
        # plot_sv_data_task(sv_list_denoised, container_name=container_name)

    except Exception as e:
        logging.critical(f'Unhandled exception occurred: {str(e)}', exc_info=True)
        sys.exit(1)
