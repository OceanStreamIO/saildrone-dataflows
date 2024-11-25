import logging
import os
import io
import sys
import time
from pathlib import Path
from dask.distributed import Client
from typing import List
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from prefect import flow, task
from prefect_dask import DaskTaskRunner
from functools import partial

load_dotenv()

# Constants and environment variables
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = './service_account.json'
DOWNLOAD_DIR = os.getenv('RAW_DATA_LOCAL', './downloaded_files')
FOLDER_ID = os.getenv('GDRIVE_FOLDER_ID')
DASK_CLUSTER_ADDRESS = os.getenv('DASK_CLUSTER_ADDRESS')
MAX_RETRIES = 5
RETRY_DELAY = 30  # seconds

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Authenticate and initialize Google Drive service
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=credentials)


@task(
    retries=MAX_RETRIES,
    retry_delay_seconds=RETRY_DELAY,
    task_run_name="download-file-{file_name}",
)
def download_file(file_id: str, file_name: str, download_dir: str) -> None:
    """
    Download a single file from Google Drive.

    Args:
        file_id (str): The Google Drive file ID.
        file_name (str): The name of the file to download.
        download_dir (str): The directory to save the downloaded file.
    """
    location = os.path.join(download_dir, file_name)
    os.makedirs(download_dir, exist_ok=True)

    if os.path.exists(location):
        logging.info(f'{file_name} already exists. Skipping download.')
        return

    try:
        logging.info(f'Starting download for {file_name}')
        request = service.files().get_media(fileId=file_id)
        with io.FileIO(location, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                logging.info(f'Download progress for {file_name}: {int(status.progress() * 100)}%')

        logging.info(f'{file_name} downloaded successfully to {location}.')
    except Exception as e:
        logging.error(f'Error downloading {file_name}: {e}')
        raise


def list_files_in_folder(folder_id: str) -> List[dict]:
    """
    List all files in a Google Drive folder.

    Args:
        folder_id (str): The Google Drive folder ID.

    Returns:
        List[dict]: List of file metadata with `id` and `name`.
    """
    query = f"'{folder_id}' in parents and trashed = false"
    items = []
    page_token = None

    try:
        while True:
            response = service.files().list(q=query, pageToken=page_token).execute()
            items.extend(response.get('files', []))
            page_token = response.get('nextPageToken', None)
            if not page_token:
                break

        logging.info(f'Total files found in folder {folder_id}: {len(items)}')
    except Exception as e:
        logging.error(f'Error listing files in folder {folder_id}: {e}')
        raise

    return [{'id': item['id'], 'name': item['name']} for item in items]


@flow(task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS))
def download_folder_from_drive(folder_id: str, download_dir: str, batch_size: int = 10) -> None:
    """
    Flow to download files from a Google Drive folder in parallel.

    Args:
        folder_id (str): The Google Drive folder ID.
        download_dir (str): The local directory to save files.
        batch_size (int): Number of files to process in each batch.
    """
    os.makedirs(download_dir, exist_ok=True)

    # List files in the folder
    raw_files = list_files_in_folder(folder_id)

    if not raw_files:
        logging.info('No files to download.')
        return

    total_files = len(raw_files)
    logging.info(f'Starting download of {total_files} files.')

    # Process files in batches
    for i in range(0, total_files, batch_size):
        batch_files = raw_files[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}")
        download_raw_data(batch_files, download_dir)

    logging.info('All files have been downloaded.')


def download_raw_data(files, download_dir) -> None:
    task_futures = []
    for file in files:
        future = download_file.submit(file_id=file['id'], file_name=file['name'], download_dir=download_dir)
        task_futures.append(future)

    # Wait for all tasks in the batch to complete
    for future in task_futures:
        future.result()


if __name__ == "__main__":
    try:
        print(f'Starting flow... {DASK_CLUSTER_ADDRESS}')
        client = Client(address=DASK_CLUSTER_ADDRESS)

        # Start the flow with the specified folder ID and download directory
        download_folder_from_drive.serve(
            name='download-folder-from-drive',
            parameters={
                'folder_id': FOLDER_ID,
                'download_dir': DOWNLOAD_DIR,
                'batch_size': 10
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception: {e}', exc_info=True)
        sys.exit(1)
