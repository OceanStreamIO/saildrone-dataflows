import logging
import os
import io
import sys
import time
import requests

from pathlib import Path
from dask.distributed import Client
from typing import List
from dotenv import load_dotenv


from prefect import flow, task
from prefect_dask import DaskTaskRunner
from saildrone.store import PostgresDB, SurveyService, FileSegmentService

load_dotenv()

# Constants and environment variables
DOWNLOAD_DIR = os.getenv('RAW_DATA_LOCAL', './downloaded_files')
DASK_CLUSTER_ADDRESS = os.getenv('DASK_CLUSTER_ADDRESS')
SURVEY_ID = 'AKBM-SagaSea-2023'
SURVEY_SEARCH_STRING = 'AKBM-SagaSea-2023'
BEARER_TOKEN = os.getenv('BEARER_TOKEN')
BASE_URL = \
    'https://api.hubocean.earth/data/catalog.hubocean.io/dataset/1e3401d4-9630-40cd-a9cf-d875cb310449-akbm-raw-ds'
MAX_RETRIES = 5
RETRY_DELAY = 30  # seconds

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@task(
    retries=MAX_RETRIES,
    retry_delay_seconds=RETRY_DELAY,
    task_run_name="download-file-{file_name}",
)
def download_file(file_name: str, file_url, download_dir: str, survey_id: int, redownload: bool, bearer_token: str,
                  metadata: dict):
    location = os.path.join(download_dir, file_name)
    bearer_token = bearer_token or BEARER_TOKEN
    os.makedirs(download_dir, exist_ok=True)
    headers = {"Authorization": f"Bearer {bearer_token}"}

    fpath = Path(location)
    file_name = fpath.stem

    try:
        logging.info(f'Starting download for {file_name}')

        with PostgresDB() as db_connection:
            file_service = FileSegmentService(db_connection)
            is_downloaded = file_service.is_file_downloaded(file_name, survey_id)

            if is_downloaded:
                logging.info(f'{file_name} has already been downloaded.')
                return

            response = requests.get(file_url, headers=headers, stream=True)
            response.raise_for_status()

            with open(location, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)

            file_service.insert_file_record(
                file_name=file_name,
                size=metadata.get("size_bytes"),
                downloaded=True,
                survey_db_id=survey_id
            )
        logging.info(f'{file_name} downloaded successfully to {location}.')
    except Exception as e:
        logging.error(f'Error downloading {file_name}: {e}')
        raise


def list_raw_files(api_url: str, bearer_token: str) -> List[dict]:
    bearer_token = bearer_token or BEARER_TOKEN
    headers = {"Authorization": f"Bearer {bearer_token}"}
    payload = {"files": "*"}
    files = []
    next_page = None

    try:
        while True:
            url = f"{api_url}&page={next_page}" if next_page else api_url

            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            # Filter files based on the prefix
            for file_info in data.get("results", []):
                if file_info["name"].startswith(SURVEY_SEARCH_STRING):
                    file_url = f"{BASE_URL}/{file_info['name']}"
                    files.append({
                        "name": file_info["name"],
                        "url": file_url,
                        "metadata": {
                            "size_bytes": file_info.get("size_bytes"),
                            "created_time": file_info.get("created_time"),
                        }
                    })
            print(f"Found {len(files)} matching .raw files.")
            next_page = data.get("next")

            if not next_page:
                break

        logging.info(f"Found {len(files)} matching .raw files.")
    except Exception as e:
        logging.error(f'Error listing files: {e}')
        raise

    return files


@flow(task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS))
def download_raw_files_from_api(download_dir: str, cruise_id: str, page_size: int = 50, batch_size: int = 10,
                                redownload: bool = False, bearer_token: str = '') -> None:
    with PostgresDB() as db_connection:
        survey_service = SurveyService(db_connection)

        # Check if a survey with the given cruise_id exists
        survey_id = survey_service.get_survey_by_cruise_id(cruise_id)

        if survey_id is None:
            # Insert a new survey record
            survey_id = survey_service.insert_survey(cruise_id, start_date='2024-05-01', end_date='2024-06-30')
            logging.info(f"Inserted new survey with cruise_id: {cruise_id}")

    logging.info(f"Survey ID: {survey_id}")
    api_url = f"{BASE_URL}/list?page_size={page_size}"

    logging.info(f"API URL: {api_url}")
    os.makedirs(download_dir, exist_ok=True)

    raw_files = list_raw_files(api_url, bearer_token)

    if not raw_files:
        logging.info('No files to download.')
        return

    total_files = len(raw_files)
    logging.info(f'Starting download of {total_files} files.')

    # Process files in batches
    for i in range(0, total_files, batch_size):
        batch_files = raw_files[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}")
        download_raw_data(batch_files, download_dir, survey_id, redownload, bearer_token)

    logging.info('All files have been downloaded.')


def download_raw_data(files, download_dir, survey_id, redownload, bearer_token) -> None:
    task_futures = []
    for file in files:
        future = download_file.submit(file['name'], file['url'], download_dir, survey_id, redownload, bearer_token,
                                      metadata=file["metadata"])
        task_futures.append(future)

    # Wait for all tasks in the batch to complete
    for future in task_futures:
        future.result()


if __name__ == "__main__":
    try:
        with PostgresDB() as db:
            db.create_tables()

        logging.info(f'Starting flow... {DASK_CLUSTER_ADDRESS}')
        client = Client(address=DASK_CLUSTER_ADDRESS)

        # Start the flow with the specified folder ID and download directory
        download_raw_files_from_api.serve(
            name='download-survey-from-hubocean',
            parameters={
                'download_dir': './downloaded_files',
                'cruise_id': 'AKBM-SagaSea-2023',
                'batch_size': 10,
                'page_size': 100,
                'redownload': False,
                'bearer_token': ''
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception: {e}', exc_info=True)
        sys.exit(1)
