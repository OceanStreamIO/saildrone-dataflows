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
from saildrone.process import save_to_partitioned_geoparquet, create_geodataframe_from_location_data

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
def create_geoparquet_file_flow(cruise_id: str, output_path: str, storage_type: str) -> None:
    with PostgresDB() as db_connection:
        survey_service = SurveyService(db_connection)

        # Check if a survey with the given cruise_id exists
        survey_id = survey_service.get_survey_by_cruise_id(cruise_id)

        if survey_id is None:
            survey_id = survey_service.insert_survey(cruise_id, start_date='2024-05-01', end_date='2024-06-30')

    logging.info(f"Survey ID: {survey_id}")
    create_geoparquet_file(cruise_id, survey_id, output_path, storage_type)

    return Completed(message="All files have been downloaded")


if __name__ == "__main__":
    try:
        with PostgresDB() as db:
            db.create_tables()

        logging.info(f'Starting flow... {DASK_CLUSTER_ADDRESS}')
        client = Client(address=DASK_CLUSTER_ADDRESS)

        # Start the flow with the specified folder ID and download directory
        create_geoparquet_file_flow.serve(
            name='create_geoparquet_file_from_db',
            parameters={
                'cruise_id': 'AKBM_SagaSea_2023',
                'output_path': './gps_data',
                'storage_type': 'local'
            }
        )
    except Exception as e:
        logging.critical(f'Unhandled exception: {e}', exc_info=True)
        sys.exit(1)
