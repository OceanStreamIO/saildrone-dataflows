from pathlib import Path
import pytest
from echopype.convert.api import open_raw
from saildrone.process import apply_calibration
from saildrone.store import ensure_container_exists, open_converted, save_zarr_store

TEST_DATA_FOLDER = "./test/data"
OUTPUT_FOLDER = "./test/processed"
SONAR_MODEL = "EK80"
CONVERTED_CONTAINER_NAME = "converted"


def test_convert_and_write_to_azure():
    ensure_container_exists('processed')

    file_path = './test/data/SD_TPOS2023_v03-Phase0-D20230826-T015958-0.raw'
    echodata = open_raw(file_path, sonar_model=SONAR_MODEL)
    echodata = apply_calibration(echodata, './saildrone/utils/calibration_values.xlsx')

    file_info = Path(file_path)
    survey_id = "testsurvey"
    zarr_path = f"{file_info.stem}.zarr"

    try:
        ensure_container_exists(CONVERTED_CONTAINER_NAME)
        save_zarr_store(echodata, zarr_path, container_name=CONVERTED_CONTAINER_NAME, survey_id=survey_id)
    except Exception as e:
        pytest.fail(f"Failed to save zarr file to Azure Blob Storage: {e}")

    echodata = open_converted(zarr_path, container_name=CONVERTED_CONTAINER_NAME, survey_id=survey_id)
    assert echodata is not None

    # <EchoData: standardized raw data from converted/testsurvey/SD_TPOS2023_v03-Phase0-D20230826-T015958-0.zarr>
    # Top-level: contains metadata about the SONAR-netCDF4 file format.
    # ├── Environment: contains information relevant to acoustic propagation through water.
    # ├── Platform: contains information about the platform on which the sonar is installed.
    # ├── Provenance: contains metadata about how the SONAR-netCDF4 version of the data were obtained.
    # ├── Sonar: contains sonar system metadata and sonar beam groups.
    # └── Vendor_specific: contains vendor-specific information about the sonar and the data.

    assert echodata['Environment'] is not None
    print("Zarr file loaded and passed basic integrity checks.")