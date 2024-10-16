import json
import os
import sys
import time
import traceback
import logging
import warnings
from pathlib import Path
from echopype.calibrate import compute_Sv
from echopype.convert.api import open_raw

from dotenv import load_dotenv
from saildrone.process_data import apply_calibration, plot_sv_data
from saildrone.store import ensure_container_exists, save_zarr_store

# from prefect_flows.convert import process_file

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
warnings.filterwarnings("ignore", module="echopype")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CHUNKS = {
    "ping_time": 100,
    "range_sample": 100
}

RAW_SOURCE_DIR = './test/data'
SONAR_MODEL = 'EK80'
CONVERTED_CONTAINER_NAME = os.getenv('CONVERTED_CONTAINER_NAME', 'converted')
PROCESSED_CONTAINER_NAME = os.getenv('PROCESSED_CONTAINER_NAME', 'processed')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')
MAX_RETRIES = 3  # Maximum number of retries for each processing
RETRY_DELAY = 5  # Delay between retries in seconds

failed_files = []


def list_files_in_directory(directory):
    raw_files = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith('.raw'):
                file_path = os.path.join(root, file_name)
                size = os.path.getsize(file_path)
                path_obj = Path(file_path)
                last_modified = time.ctime(os.path.getmtime(file_path))
                folder = os.path.dirname(file_path)

                raw_files.append({
                    'name': file_name,
                    'folder': folder,
                    'stem': path_obj.stem,
                    'size': size,
                    'path': file_path,
                    'last_modified': last_modified
                })

    logging.info(f'Total number of raw files: {len(raw_files)}')

    return raw_files


def process_raw_file(file_info, sonar_model=None):

    try:
        logging.info(f'Starting processing of {file_info["name"]}')
        # echodata = process_file(file_info['path'], sonar_model=sonar_model)
        file_path = file_info['path']
        echodata = open_raw(file_path, sonar_model=sonar_model)
        echodata = apply_calibration(echodata)

        # echodata["Environment"] = echodata["Environment"].assign_coords(
        #     sound_velocity_profile_depth=[0]
        # )
        zarr_path = f"{file_info['stem']}.zarr"
        ensure_container_exists(CONVERTED_CONTAINER_NAME)
        save_zarr_store(echodata, container_name=CONVERTED_CONTAINER_NAME, zarr_path=zarr_path)

        # echodata["Vendor_specific"] = echodata["Vendor_specific"].isel(channel=slice(1))
        # sv_dataset = compute_Sv(echodata, waveform_mode='CW', encode_mode='complex').compute()

        # sv_path = f"test/processed/{file_info['stem']}_Sv.zarr"
        # plot_sv_data(sv_dataset, file_info['stem'], 'test/processed')
        # sv_dataset.to_zarr(sv_path, mode='w')

    except Exception as e:
        logging.error(f'Failed to process {file_info["name"]}: {e}')
        traceback.print_exc()
        return None


def main():
    files = list_files_in_directory(RAW_SOURCE_DIR)
    if not files:
        return

    process_raw_file('./test/data/SD_TPOS2023_v03-Phase0-D20230826-T015958-0.raw', SONAR_MODEL)


if __name__ == "__main__":
    main()
