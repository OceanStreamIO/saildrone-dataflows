import os
from typing import Optional, List
from adlfs import AzureBlobFileSystem
import xarray as xr
import geopandas as gpd
import logging
from azure.storage.blob import BlobServiceClient, ContentSettings

# Initialize the logger
logger = logging.getLogger('oceanstream')

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


def get_azure_blob_filesystem() -> AzureBlobFileSystem:
    """
    Get Azure Blob FileSystem Mapper for Dask.

    Returns:
    - AzureBlobFileSystem: The Azure Blob FileSystem Mapper.
    """
    connect_str: Optional[str] = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
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
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    azfs = AzureBlobFileSystem(connection_string=connection_string)

    if azfs is None:
        raise ValueError("Azure Blob Storage connection string not found and no azfs instance was specified.")

    if survey_id is not None:
        zarr_path_full = f"{container_name}/{survey_id}/{zarr_path}"
    else:
        zarr_path_full = f"{container_name}/{zarr_path}"

    logger.info(f"Opening Zarr store: {zarr_path_full}")
    chunk_store = azfs.get_mapper(zarr_path_full)

    return xr.open_dataset(chunk_store, engine='zarr', chunks=chunks)


def open_converted(zarr_path, survey_id=None, container_name=CONVERTED_CONTAINER_NAME, chunks=None):
    """Open a Zarr store from Azure Blob Storage."""
    from echopype.echodata.api import open_converted

    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    azfs = AzureBlobFileSystem(connection_string=connection_string)

    if azfs is None:
        raise ValueError("Azure Blob Storage connection string not found and no azfs instance was specified.")

    if survey_id is not None:
        zarr_path_full = f"{container_name}/{survey_id}/{zarr_path}"
    else:
        zarr_path_full = f"{container_name}/{zarr_path}"

    logger.info(f"Opening Zarr store: {zarr_path_full}")
    chunk_store = azfs.get_mapper(zarr_path_full)

    return open_converted(chunk_store, chunks=chunks)


def open_geo_parquet(pq_path, survey_id=None, container_name=None):
    """Open a Zarr store from Azure Blob Storage."""
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    azfs = AzureBlobFileSystem(connection_string=connection_string)

    if azfs is None:
        raise ValueError("Azure Blob Storage connection string not found and no azfs instance was specified.")

    if survey_id is not None and container_name is not None:
        pq_path_full = f"{container_name}/{survey_id}/{pq_path}"
    elif container_name is not None:
        pq_path_full = f"{container_name}/{pq_path}"
    else:
        pq_path_full = pq_path

    logger.info(f"Opening Zarr store: {pq_path_full}")
    with azfs.open(pq_path_full, 'rb') as f:
        gdf = gpd.read_parquet(f)

    return gdf
