import os
import random
import re
import shutil
import tempfile
import zipfile

import pandas as pd
import xarray as xr
import geopandas as gpd
import logging
import uuid

from prefect_dask import get_dask_client
from pathlib import Path
from datetime import timedelta, datetime
from typing import List, Union, TypedDict
from adlfs import AzureBlobFileSystem
from azure.storage.blob import BlobServiceClient, generate_container_sas, ContainerSasPermissions


# Initialize the logger
logger = logging.getLogger(__name__)

CONVERTED_CONTAINER_NAME = os.getenv('CONVERTED_CONTAINER_NAME', 'converted')
PROCESSED_CONTAINER_NAME = os.getenv('PROCESSED_CONTAINER_NAME', 'processed')


def zip_and_save_netcdf_files(file_paths, zip_name, container_name, tmp_dir=None):
    if tmp_dir is not None:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
    else:
        tmp_dir = tempfile.mkdtemp()

    with tempfile.NamedTemporaryFile(suffix=".zip", dir=tmp_dir, delete=False) as tmpfile:
        zip_path = Path(tmpfile.name)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in file_paths:
            archive.write(path, arcname=Path(path).name)
            logger.info(f"Added to archive: {path}")

    upload_file_to_blob(str(zip_path), zip_name, container_name)


def create_blob_service_client(connect_str=None) -> BlobServiceClient:
    """
    Create an Azure Blob Storage client.

    Returns:
    - BlobServiceClient: The Azure Blob Storage client.
    """
    if connect_str is None:
        connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

    if not connect_str:
        raise EnvironmentError("AZURE_STORAGE_CONNECTION_STRING environment variable not set.")

    return BlobServiceClient.from_connection_string(connect_str)


def get_azure_blob_filesystem(storage_config=None) -> AzureBlobFileSystem:
    """
    Get Azure Blob FileSystem Mapper for Dask.

    Returns:
    - AzureBlobFileSystem: The Azure Blob FileSystem Mapper.
    """

    if storage_config and storage_config['storage_type'] == 'azure':
        azfs = AzureBlobFileSystem(**storage_config['storage_options'])
        return azfs

    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

    if not connect_str:
        raise EnvironmentError("AZURE_STORAGE_CONNECTION_STRING environment variable not set.")

    return AzureBlobFileSystem(connection_string=connect_str)


def ensure_container_exists(container_name: str, blob_service_client: BlobServiceClient = None, public_access=None):
    """
    Ensure that the specified container exists in Azure Blob Storage.

    Parameters:
    - blob_service_client: BlobServiceClient
        The Azure Blob Storage client.
    - container_name: str
        The name of the container to check or create.
    """
    try:
        if blob_service_client is None:
            blob_service_client = create_blob_service_client()

        container_client = blob_service_client.get_container_client(container_name)

        # Check if the container exists
        if not container_client.exists():
            logger.info(f"Container '{container_name}' does not exist. Creating container...")
            container_client.create_container(public_access=public_access)
            logger.info(f"Container '{container_name}' created successfully.")
        else:
            logger.info(f"Container '{container_name}' already exists.")

    except Exception as e:
        logger.error(f"Error ensuring container '{container_name}' exists: {e}", exc_info=True)
        raise


def save_zarr_store(echodata_or_sv_ds, zarr_path, survey_id=None, container_name=None, mode="w", append_dim=None):
    if survey_id is not None:
        zarr_path = f"{survey_id}/{zarr_path}"

    if container_name is not None:
        zarr_path_full = f"{container_name}/{zarr_path}"
    else:
        zarr_path_full = zarr_path

    azfs = get_azure_blob_filesystem()
    zarr_store = azfs.get_mapper(zarr_path_full)

    logger.info(f"Saving converted data to Zarr format at: {zarr_path_full}")

    if isinstance(echodata_or_sv_ds, xr.Dataset):
        # echodata_or_sv_ds.to_zarr(store=zarr_store, mode='w')
        rechunked_ds = echodata_or_sv_ds.chunk({dim: -1 for dim in echodata_or_sv_ds.dims})
        if mode == "w":
            rechunked_ds.to_zarr(store=zarr_store, mode='w')
        elif mode == "a" and append_dim is not None:
            rechunked_ds.to_zarr(store=zarr_store, mode='a', append_dim=append_dim)
    else:
        echodata_or_sv_ds.to_zarr(save_path=zarr_store, overwrite=True)

    return zarr_path


def open_zarr_store(zarr_path, cruise_id=None, container_name=PROCESSED_CONTAINER_NAME, chunks=None, rechunk_after=False):
    """Open a Zarr store from Azure Blob Storage."""
    azfs = get_azure_blob_filesystem()

    if cruise_id is not None:
        zarr_path_full = f"{container_name}/{cruise_id}/{zarr_path}"
    else:
        zarr_path_full = f"{container_name}/{zarr_path}"

    logger.info(f"Opening Zarr store: {zarr_path_full}")
    chunk_store = azfs.get_mapper(zarr_path_full)

    if rechunk_after and chunks is not None:
        ds = xr.open_dataset(chunk_store, engine='zarr', chunks=None)
        ds = ds.chunk(chunks)
        return ds

    return xr.open_dataset(chunk_store, engine='zarr', chunks=chunks)


def list_zarr_files(path, azfs=None, cruise_id=None, file_names=None) -> List[Path]:
    """List all Zarr files in the Azure Blob Storage container along with their metadata."""

    if azfs is None:
        azfs = get_azure_blob_filesystem()

    if azfs is None:
        raise ValueError("Azure Blob Storage connection string not found and no azfs instance was specified.")

    zarr_files = []

    if cruise_id is not None:
        path = f"{path}/{cruise_id}"

    if file_names is not None and cruise_id is not None:
        for file_name in file_names:
            file_name = f"{path}/{file_name}.zarr"
            zarr_files.append(Path(file_name))
    else:
        for blob in azfs.ls(path, detail=True):
            if blob['type'] == 'directory' and not blob['name'].endswith('.zarr'):
                subdir_files = list_zarr_files(blob['name'], azfs)
                zarr_files.extend(subdir_files)
            elif blob['name'].endswith('.zarr'):
                zarr_files.append(Path(blob['name']))

    return zarr_files


def open_converted(zarr_path, survey_id=None, container_name=None, chunks=None):
    """Open a Zarr store from Azure Blob Storage."""
    from echopype.echodata.api import open_converted

    azfs = get_azure_blob_filesystem()

    if survey_id is not None:
        zarr_path_full = f"{container_name}/{survey_id}/{zarr_path}"
    elif container_name is not None:
        zarr_path_full = f"{container_name}/{zarr_path}"
    else:
        zarr_path_full = zarr_path

    logger.info(f"Opening Zarr store: {zarr_path_full}")
    chunk_store = azfs.get_mapper(zarr_path_full)

    return open_converted(chunk_store, chunks=chunks)


def open_geo_parquet(pq_path, survey_id=None, container_name=None, has_geometry=True):
    """Open a geo parquet file from Azure Blob Storage."""
    azfs = get_azure_blob_filesystem()

    if survey_id is not None and container_name is not None:
        pq_path_full = f"{container_name}/{survey_id}/{pq_path}"
    elif container_name is not None:
        pq_path_full = f"{container_name}/{pq_path}"
    else:
        pq_path_full = pq_path

    logger.info(f"Opening parquet file: {pq_path_full}")
    with azfs.open(pq_path_full, 'rb') as f:
        if has_geometry:
            gdf = gpd.read_parquet(f)
        else:
            gdf = pd.read_parquet(f)

    return gdf


def get_variable_encoding(ds: xr.Dataset, compression_level):
    """Generate encoding dictionary for dataset variables."""
    encoding = {}
    for var in ds.data_vars:
        if ds[var].dtype.kind in {"U", "S", "O"}:  # String or object types
            # No compression or chunking for unsupported types
            encoding[var] = {}
        else:
            # Apply compression for numeric types
            encoding[var] = {
                "zlib": True,
                "complevel": compression_level,
            }
    return encoding


def save_dataset_to_netcdf(
    ds: xr.Dataset,
    container_name: str = None,
    base_local_temp_path: str = '/tmp/oceanstream/netcdfdata',
    ds_path: str = "short_pulse_data.nc",
    compression_level: int = 5,
    is_temp_dir: bool = True
):
    # Construct full local path
    full_dataset_path = Path(base_local_temp_path) / container_name / ds_path

    enc = get_variable_encoding(ds, compression_level)

    def _write_and_upload(d, path_str, encoding, filename, container):
        p = Path(path_str)
        if is_temp_dir:
            p.parent.mkdir(parents=True, exist_ok=True, mode=0o775)
            os.chmod(p.parent, 0o775)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)

        d.load()
        d.to_netcdf(str(p), engine="netcdf4", format="NETCDF4", encoding=encoding)
        upload_file_to_blob(str(p), filename, container_name=container)
        return str(p)

    # Save the dataset to the full path
    with get_dask_client() as client:
        future = client.submit(
            _write_and_upload,
            ds,
            str(full_dataset_path),
            enc,
            ds_path,
            container_name,
            pure=False
        )
        output_path = future.result()
        print('Saved dataset to:', output_path)

    return full_dataset_path


def save_datasets_to_netcdf(
    short_pulse_ds: xr.Dataset,
    long_pulse_ds: xr.Dataset,
    container_name: str = None,
    base_local_temp_path: str = '/tmp/osnetcdf',
    short_pulse_path: str = "short_pulse_data.nc",
    long_pulse_path: str = "long_pulse_data.nc",
    compression_level: int = 5
):
    container_local_path = os.path.join(base_local_temp_path, container_name)
    os.makedirs(container_local_path, exist_ok=True)

    local_short_pulse_path = os.path.join(container_local_path, short_pulse_path)
    local_long_pulse_path = os.path.join(container_local_path, long_pulse_path)

    # Save the datasets locally
    short_pulse_ds.to_netcdf(
        path=local_short_pulse_path,
        format='NETCDF4',
        engine='netcdf4',
        encoding=get_variable_encoding(short_pulse_ds, compression_level)
    )
    long_pulse_ds.to_netcdf(
        path=local_long_pulse_path,
        format='NETCDF4',
        engine='netcdf4',
        encoding=get_variable_encoding(long_pulse_ds, compression_level)
    )

    upload_file_to_blob(local_short_pulse_path, short_pulse_path, container_name=container_name)
    upload_file_to_blob(local_long_pulse_path, long_pulse_path, container_name=container_name)


def upload_file_to_blob(local_path, blob_path, container_name=None):
    """
    Upload a file from local path to Azure Blob Storage.

    Parameters:
        local_path: Local path to the file.
        blob_path: Blob path in the container.
    """
    blob_service_client = create_blob_service_client()
    container_client = blob_service_client.get_container_client(container_name)

    with open(local_path, "rb") as data:
        container_client.upload_blob(name=blob_path, data=data, overwrite=True)


def generate_container_access_url(container_name, duration_days=90):
    """
    Generate a SAS token for container access and return the URL.
    """
    AZURE_STORAGE_ACCOUNT_NAME = os.getenv('AZURE_STORAGE_ACCOUNT_NAME')
    AZURE_STORAGE_ACCOUNT_KEY = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')

    sas_token = generate_container_sas(
        account_name=AZURE_STORAGE_ACCOUNT_NAME,
        container_name=container_name,
        account_key=AZURE_STORAGE_ACCOUNT_KEY,
        permission=ContainerSasPermissions(read=True, list=True),
        expiry=datetime.utcnow() + timedelta(days=duration_days)
    )

    url = f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{container_name}?{sas_token}"
    return url


def generate_container_name(cruise_id: str):
    """Generate a unique container name based on the date, cruise_id, and a UUID."""
    date_str = datetime.now().strftime("%Y%m%d")
    unique_id = uuid.uuid4().hex[:8]

    raw_name = f"{cruise_id}{date_str}{unique_id}".lower()
    sanitized_name = re.sub(r'[^a-z0-9-]', '', raw_name)

    return sanitized_name[:63]  # Limit to 63 characters for Azure Blob Storage


def upload_folder_to_blob_storage(folder_path, container_name, target_path):
    blob_service_client = create_blob_service_client()
    container_client = blob_service_client.get_container_client(container_name)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, start=folder_path)
            blob_path = os.path.join(target_path, relative_path).replace(os.sep, '/')

            blob_client = container_client.get_blob_client(blob_path)

            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
