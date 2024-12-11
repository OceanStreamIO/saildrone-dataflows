import os
import fsspec
import geopandas as gpd
import pandas as pd

from .location import extract_location_data
from saildrone.store import get_azure_blob_filesystem, open_geo_parquet
from shapely.geometry import Point
from datetime import datetime, timedelta


def process_geo_location(file_name, geo_data, metadata):
    geometry = geo_data.get("geometry", {})
    coordinates = geometry.get("coordinates", [])

    # Ensure coordinates exist
    if not coordinates or geometry.get("type") != "LineString":
        print(f"No valid geo_location found for file: {file_name}")
        return

    # Extract start and end coordinates
    start_lat, start_lon = coordinates[0][1], coordinates[0][0]
    end_lat, end_lon = coordinates[-1][1], coordinates[-1][0]

    # Extract timestamps
    time_start = datetime.fromisoformat(metadata.get("timeStart").replace("Z", "+00:00"))
    time_end = datetime.fromisoformat(metadata.get("timeEnd").replace("Z", "+00:00"))
    num_points = len(coordinates)

    # Interpolate timestamps for each coordinate
    time_delta = (time_end - time_start) / (num_points - 1)
    timestamps = [time_start + i * time_delta for i in range(num_points)]

    # Create a GeoDataFrame
    df = pd.DataFrame({
        "latitude": [coord[1] for coord in coordinates],
        "longitude": [coord[0] for coord in coordinates],
        "time": timestamps
    })
    df['geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    processed_df = extract_location_data(gdf)
    # Partition and save to GeoParquet
    # save_to_partitioned_geoparquet(gdf, output_path, storage_type)

    # Return metadata summary for database storage
    return {
        "file_start_lat": start_lat,
        "file_start_lon": start_lon,
        "file_end_lat": end_lat,
        "file_end_lon": end_lon,
        "location_data": processed_df.to_dict(orient="records")
    }


def save_to_partitioned_geoparquet(gdf: gpd.GeoDataFrame, output_path: str, storage_type='local', grid_size=1.0):
    """
    Saves a GeoDataFrame to partitioned GeoParquet files.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing GPS data.
        output_path (str): Path to save the partitioned GeoParquet files.
        storage_type (str): Storage type ('local' or 'azure').
        grid_size (float): Size of the grid for partitioning in degrees.

    Returns:
        None
    """
    if storage_type == 'azure':
        fs = get_azure_blob_filesystem()
    else:
        fs = fsspec.filesystem('file')

    # Calculate partitioning columns
    gdf['lon_grid'] = (gdf['longitude'] // grid_size).astype('int32')
    gdf['lat_grid'] = (gdf['latitude'] // grid_size).astype('int32')

    grouped = gdf.groupby(['lon_grid', 'lat_grid'])

    metadata_records = []

    for (lon_grid, lat_grid), group in grouped:
        if storage_type == 'azure':
            partition_path = f"{output_path}/lon_grid={lon_grid}/lat_grid={lat_grid}/data.parquet"
        else:
            partition_path = os.path.join(output_path, f'lon_grid={lon_grid}', f'lat_grid={lat_grid}', 'data.parquet')
            local_directory = os.path.dirname(partition_path)
            os.makedirs(local_directory, exist_ok=True)

        # Drop partitioning columns before saving to Parquet
        group_to_save = group.drop(columns=['lon_grid', 'lat_grid'])
        with fs.open(partition_path, 'wb') as f:
            group_to_save.to_parquet(f, index=False)

        # Save metadata for partition
        start_time = group['time'].min()
        end_time = group['time'].max()
        min_lat = group['latitude'].min()
        max_lat = group['latitude'].max()
        min_lon = group['longitude'].min()
        max_lon = group['longitude'].max()

        metadata_records.append({
            'partition_path': partition_path,
            'start_time': start_time,
            'end_time': end_time,
            'min_lat': min_lat,
            'max_lat': max_lat,
            'min_lon': min_lon,
            'max_lon': max_lon,
            'num_records': len(group)
        })

    # Save metadata to a Parquet file
    metadata_df = pd.DataFrame(metadata_records)
    metadata_path = os.path.join(output_path, 'metadata.parquet')
    with fs.open(metadata_path, 'wb') as f:
        metadata_df.to_parquet(f, index=False)
