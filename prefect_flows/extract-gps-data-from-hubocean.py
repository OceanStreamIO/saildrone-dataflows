import logging
import os
import sys
import traceback

import requests

from pathlib import Path
from dask.distributed import Client
from typing import List
from dotenv import load_dotenv

from prefect.artifacts import create_markdown_artifact
from prefect.states import Completed
from prefect import flow, task
from prefect_dask import DaskTaskRunner

from saildrone.azure_iot import serialize_location_data
from saildrone.store import PostgresDB, SurveyService, FileSegmentService
from saildrone.process import (process_geo_location, save_to_partitioned_geoparquet,
                               create_geodataframe_from_location_data)

load_dotenv()

# Constants and environment variables
DOWNLOAD_DIR = os.getenv('RAW_DATA_LOCAL', './downloaded_files')
DASK_CLUSTER_ADDRESS = os.getenv('DASK_CLUSTER_ADDRESS')
GPSDATA_CONTAINER_NAME = os.getenv('GPSDATA_CONTAINER_NAME')
SURVEY_ID = 'AKBM_SagaSea_2023'
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
    retries=5,
    retry_delay_seconds=[10, 30, 60],
    retry_jitter_factor=0.1,
    refresh_cache=True,
    task_run_name="process-file-{file_path}",
)
def process_file(file_path: str, geolocation: dict, metadata):
    markdown_report = f"""# Task Report for {file_path}"""

    logging.info(f'Starting download for {file_path}')

    with PostgresDB() as db_connection:
        file_service = FileSegmentService(db_connection)
        file_obj = Path(file_path)
        file_name = file_obj.stem
        file_info = file_service.get_file_info(file_name)

        if not file_info:
            logging.info(f'{file_name} has not been downloaded.')
            return

        print(f'Processing file: {file_name}')

        if geolocation is None:
            return Completed(message=f"No geolocation data found for file: {file_name}")

        location_summary = process_geo_location(file_name, geolocation, metadata)
        markdown_report += f"\n\nLocation summary: {location_summary}"

        if location_summary is None:
            return Completed(message=f"No valid geo_location found for file: {file_name}")

        location_data_str = None

        if location_summary["location_data"]:
            location_data_str = serialize_location_data(location_summary["location_data"])

        try:
            file_service.update_file_record(
                file_id=file_info['id'],
                file_start_lat=location_summary["file_start_lat"],
                file_start_lon=location_summary["file_start_lon"],
                file_end_lat=location_summary["file_end_lat"],
                file_end_lon=location_summary["file_end_lon"],
                location_data=location_data_str
            )

            file_service.update_geospatial_data(
                file_id=file_info['id'],
                file_start_lat=location_summary["file_start_lat"],
                file_start_lon=location_summary["file_start_lon"],
                file_end_lat=location_summary["file_end_lat"],
                file_end_lon=location_summary["file_end_lon"],
                track_geom=geolocation["geometry"]
            )

            markdown_report += f"\n\nProcessed file {file_name} successfully."
            create_markdown_artifact(markdown_report)

        except Exception as e:
            logging.error(f'Error processing file {file_name}: {e}')
            stack_trace = traceback.format_exc()
            markdown_report += f"\n\nError processing file {file_name}: {e}"
            markdown_report += f"\n\n{stack_trace}"
            create_markdown_artifact(markdown_report)

            return Completed(message=f"Error processing file {file_name}: {e}")


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
                        "metadata": file_info.get("metadata"),
                        "geolocation": file_info.get("geo_location")
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


@task(
    task_run_name="create-geoparquet-file-{cruise_id}"
)
def create_geoparquet_file(cruise_id, survey_id, output_path, storage_type):
    markdown_report = f"""# Report for create_geoparquet_file"""
    try:
        with PostgresDB() as db_connection:
            file_service = FileSegmentService(db_connection)

            location_data_list = file_service.fetch_location_data_by_survey_id(survey_id)
            gdf = create_geodataframe_from_location_data(location_data_list)

            if storage_type == 'azure':
                output_path = f'{GPSDATA_CONTAINER_NAME}/{cruise_id}'
                output_path = output_path.replace('-', '_')

            print(f'Saving GeoParquet file...{output_path}')

            save_to_partitioned_geoparquet(gdf, output_path, storage_type)
            markdown_report += f"\n\nSaved GeoParquet file to {output_path}."
            create_markdown_artifact(markdown_report)

        return output_path

    except Exception as e:
        logging.error(f'Error creating GeoParquet file: {e}')
        markdown_report += f"\n\nError creating GeoParquet file: {e}"
        create_markdown_artifact(markdown_report)
        raise


@flow(task_runner=DaskTaskRunner(address=DASK_CLUSTER_ADDRESS))
def extract_geolocation_from_api(cruise_id: str, batch_size: int = 10, geoparquet_storage_type: str = None,
                                 bearer_token: str = '') -> None:
    with PostgresDB() as db_connection:
        survey_service = SurveyService(db_connection)

        # Check if a survey with the given cruise_id exists
        survey_id = survey_service.get_survey_by_cruise_id(cruise_id)

        if survey_id is None:
            # Insert a new survey record
            survey_id = survey_service.insert_survey(cruise_id, start_date='2024-05-01', end_date='2024-06-30')
            logging.info(f"Inserted new survey with cruise_id: {cruise_id}")

    logging.info(f"Survey ID: {survey_id}")
    api_url = f"{BASE_URL}/list?page_size={batch_size}"

    logging.info(f"API URL: {api_url}")

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
        process_raw_data(batch_files)

    logging.info('All files have been downloaded.')
    # create_geoparquet_file(cruise_id, survey_id, './gps_data', geoparquet_storage_type)

    return Completed(message="All files have been downloaded")


def process_raw_data(files) -> None:
    task_futures = []
    for file in files:
        future = process_file.submit(file['name'], file["geolocation"], file["metadata"])
        task_futures.append(future)

    # Wait for all tasks in the batch to complete
    for future in task_futures:
        result = future.result()


if __name__ == "__main__":
    try:
        with PostgresDB() as db:
            db.create_tables()

        logging.info(f'Starting flow... {DASK_CLUSTER_ADDRESS}')
        client = Client(address=DASK_CLUSTER_ADDRESS)

        # Start the flow with the specified folder ID and download directory
        extract_geolocation_from_api.serve(
            name='extract-geolocation-from-hubocean',
            parameters={
                'cruise_id': 'AKBM_SagaSea_2023',
                'batch_size': 10,
                'geoparquet_storage_type': 'local',
                'bearer_token': ''
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception: {e}', exc_info=True)
        sys.exit(1)
