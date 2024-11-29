import os
from pathlib import Path

import pandas as pd
import xarray as xr
import geopandas as gpd
import logging

from typing import List, Union, TypedDict
from adlfs import AzureBlobFileSystem
from azure.storage.blob import BlobServiceClient

# Initialize the logger
logger = logging.getLogger(__name__)

CONVERTED_CONTAINER_NAME = os.getenv('CONVERTED_CONTAINER_NAME', 'converted')
PROCESSED_CONTAINER_NAME = os.getenv('PROCESSED_CONTAINER_NAME', 'processed')


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


def ensure_container_exists(container_name: str, blob_service_client: BlobServiceClient = None):
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
            container_client.create_container()
            logger.info(f"Container '{container_name}' created successfully.")
        else:
            logger.info(f"Container '{container_name}' already exists.")

    except Exception as e:
        logger.error(f"Error ensuring container '{container_name}' exists: {e}", exc_info=True)
        raise


def save_zarr_store(echodata_or_sv_ds, zarr_path, survey_id=None, container_name=None):
    if container_name is not None and survey_id is not None:
        zarr_path_full = f"{container_name}/{survey_id}/{zarr_path}"
    elif container_name is not None:
        zarr_path_full = f"{container_name}/{zarr_path}"
    else:
        zarr_path_full = zarr_path

    azfs = get_azure_blob_filesystem()
    zarr_store = azfs.get_mapper(zarr_path_full)

    logger.info(f"Saving converted data to Zarr format at: {zarr_path_full}")

    if isinstance(echodata_or_sv_ds, xr.Dataset):
        echodata_or_sv_ds.to_zarr(store=zarr_store, mode='w')
    else:
        echodata_or_sv_ds.to_zarr(save_path=zarr_store, overwrite=True)

    return zarr_store


def open_zarr_store(zarr_path, survey_id=None, container_name=PROCESSED_CONTAINER_NAME, chunks=None):
    """Open a Zarr store from Azure Blob Storage."""
    azfs = get_azure_blob_filesystem()

    if azfs is None:
        raise ValueError("Azure Blob Storage connection string not found and no azfs instance was specified.")

    if survey_id is not None:
        zarr_path_full = f"{container_name}/{survey_id}/{zarr_path}"
    else:
        zarr_path_full = f"{container_name}/{zarr_path}"

    logger.info(f"Opening Zarr store: {zarr_path_full}")
    chunk_store = azfs.get_mapper(zarr_path_full)

    return xr.open_dataset(chunk_store, engine='zarr', chunks=chunks)


def list_zarr_files(path, azfs=None, cruise_id=None) -> List[Path]:
    """List all Zarr files in the Azure Blob Storage container along with their metadata."""

    if azfs is None:
        azfs = get_azure_blob_filesystem()

    if azfs is None:
        raise ValueError("Azure Blob Storage connection string not found and no azfs instance was specified.")

    zarr_files = []

    if cruise_id is not None:
        path = f"{path}/{cruise_id}"

    print('Listing files in path:', path)
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

    if azfs is None:
        raise ValueError("Azure Blob Storage connection string not found and no azfs instance was specified.")

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

    if azfs is None:
        raise ValueError("Azure Blob Storage connection string not found and no azfs instance was specified.")

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
