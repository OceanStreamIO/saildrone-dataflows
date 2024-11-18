import pandas as pd
import pytest
from saildrone.process import consolidate_csv_to_geoparquet_partitioned
from saildrone.store import ensure_container_exists, open_geo_parquet


TEST_DATA_FOLDER = "./test/gps-data"
OUTPUT_FOLDER = "./test/gps-processed"
PARTITIONED_DATA_PATH = OUTPUT_FOLDER


def test_geoparquet_data_writetoazure():
    container_name = 'gpsdata'
    survey_name = 'SD_TPOS2023_v03'

    ensure_container_exists(container_name)
    storage_path = f"{container_name}/{survey_name}"

    consolidate_csv_to_geoparquet_partitioned(TEST_DATA_FOLDER, storage_path, storage_type='azure')

    container_name = 'gpsdata'
    partition_path = f'{survey_name}/lon_grid=-158/lat_grid=21/data.parquet'

    try:
        gdf = open_geo_parquet(partition_path, container_name=container_name)

        # Basic integrity checks
        required_columns = ['time', 'latitude', 'longitude', 'geometry']
        missing_columns = [col for col in required_columns if col not in gdf.columns]
        assert not missing_columns, f"Missing required columns: {missing_columns}"

        # Check that 'time' is in datetime format
        assert pd.api.types.is_datetime64_any_dtype(gdf['time']), "'time' column is not in datetime format."

        # Check the CRS (Coordinate Reference System)
        assert gdf.crs == "EPSG:4326", f"CRS is {gdf.crs}, expected 'EPSG:4326'."

        # Check that geometry is valid
        invalid_geometries = gdf[~gdf.is_valid]
        assert invalid_geometries.empty, f"Found {len(invalid_geometries)} invalid geometries."

        print("GeoParquet file loaded and passed basic integrity checks.")
    except Exception as e:
        pytest.fail(f"Failed to load GeoParquet file from Azure Blob Storage: {e}")