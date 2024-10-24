import os
import shutil
import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from saildrone.process import consolidate_csv_to_geoparquet_partitioned

TEST_DATA_FOLDER = "./test/gps-data"
OUTPUT_FOLDER = "./test/gps-processed"
PARTITIONED_DATA_PATH = OUTPUT_FOLDER


@pytest.fixture(scope="function")
def setup_and_cleanup():
    """
    Fixture to set up the test environment: clear the output folder and process the CSV files.
    """
    # Clear the output folder
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Process CSV files to create a single GeoParquet file
    consolidate_csv_to_geoparquet_partitioned(TEST_DATA_FOLDER, PARTITIONED_DATA_PATH)

    yield


@pytest.mark.usefixtures("setup_and_cleanup")
def test_geoparquet_data_integrity():
    """
    Test the data integrity of the GeoParquet file and perform example geospatial queries.
    """
    parquet_files = []
    for root, dirs, files in os.walk(PARTITIONED_DATA_PATH):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))

    assert parquet_files, "No partitioned Parquet files found."

    data_frames = []
    for file_path in parquet_files:
        try:
            gdf = gpd.read_parquet(file_path)
            data_frames.append(gdf)
            print(f"Loaded partitioned file: {file_path}")
        except Exception as e:
            pytest.fail(f"Failed to load partitioned Parquet file '{file_path}': {e}")

    combined_gdf = gpd.GeoDataFrame(pd.concat(data_frames, ignore_index=True))
    assert not combined_gdf.empty, "Combined GeoDataFrame is empty."

    required_columns = ['time', 'latitude', 'longitude', 'geometry', 'lon_grid', 'lat_grid']
    missing_columns = [col for col in required_columns if col not in combined_gdf.columns]
    assert not missing_columns, f"Missing required columns: {missing_columns}"

    assert pd.api.types.is_datetime64_any_dtype(combined_gdf['time']), "'time' column is not in datetime format."

    assert combined_gdf.crs == "EPSG:4326", f"CRS is {combined_gdf.crs}, expected 'EPSG:4326'."

    invalid_geometries = combined_gdf[~combined_gdf.is_valid]
    assert invalid_geometries.empty, f"Found {len(invalid_geometries)} invalid geometries."

    # **Example Geospatial Queries**

    # Query 1: Filter data within a bounding box (e.g., a specific region)
    bbox_geom = box(-158, 21, -157, 22)
    filtered_data = combined_gdf[combined_gdf.geometry.intersects(bbox_geom)]
    assert not filtered_data.empty, "No data points found within bounding box."

    # Query 2: Find the nearest point to a specific location (e.g., research station)
    target_point = Point(-157.7349, 21.9525)
    combined_gdf['distance_to_target'] = combined_gdf.geometry.distance(target_point)
    nearest_point = combined_gdf.loc[combined_gdf['distance_to_target'].idxmin()]
    assert nearest_point is not None, "Failed to find nearest point to the target location."

    # Query 3: Check if points are within a defined polygon (e.g., custom area)
    polygon = Polygon([(-158, 21), (-157, 21), (-157, 22), (-158, 22), (-158, 21)])
    points_within_polygon = combined_gdf[combined_gdf.geometry.within(polygon)]
    assert not points_within_polygon.empty, "No data points found within the custom polygon."

    print("\nData Integrity and Geospatial Queries completed successfully.")
