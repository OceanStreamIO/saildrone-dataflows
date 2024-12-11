import os
import fsspec
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point
from saildrone.store import get_azure_blob_filesystem, open_geo_parquet


def consolidate_csv_to_geoparquet_partitioned(folder_path, output_path, storage_type='local'):
    """
    Consolidates multiple CSV files in a folder into partitioned GeoParquet files.

    Parameters:
    folder_path (str): Path to the folder containing CSV files.
    output_directory (str): Path where the partitioned GeoParquet files will be saved.
    """
    combined_data = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            df['time'] = pd.to_datetime(df.iloc[:, 0])  # Convert the first column to datetime
            df = df.rename(columns={df.columns[1]: 'latitude', df.columns[2]: 'longitude'})  # Rename columns
            df = df.drop(columns=[df.columns[0]])  # Drop the original time column

            df = df.dropna(subset=['latitude', 'longitude'])

            # Ensure latitude and longitude are numeric
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

            # Create GeoDataFrame with geometry based on latitude and longitude
            df['geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
            gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
            combined_data.append(gdf)

    combined_gdf = pd.concat(combined_data, ignore_index=True)
    combined_gdf = combined_gdf.dropna(subset=['longitude', 'latitude'])

    save_to_partitioned_geoparquet(combined_gdf, output_path, storage_type)


def save_to_partitioned_geoparquet(gdf: gpd.GeoDataFrame, output_path: str, storage_type='local', grid_size=1.0):
    # Calculate grid indices for partitioning
    if storage_type == 'azure':
        fs = get_azure_blob_filesystem()
    else:
        fs = fsspec.filesystem('file')

    gdf['lon_grid'] = (gdf['longitude'] // grid_size).astype('int32')
    gdf['lat_grid'] = (gdf['latitude'] // grid_size).astype('int32')

    # Group by partitioning columns and write each group to a separate Parquet file
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

        # Save the partitioned data with fastparquet
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

    metadata_df = pd.DataFrame(metadata_records)
    metadata_path = os.path.join(output_path, 'metadata.parquet')
    with fs.open(metadata_path, 'wb') as f:
        metadata_df.to_parquet(f, index=False)


def query_location_points_between_timestamps(file_start_time, file_end_time, geoparquet_path=None,
                                             container_name=None, survey_id=None):
    """
    Query location points between two timestamps by loading only relevant partitions based on metadata.

    Parameters:
    geoparquet_path (str): Path to the directory containing partitioned GeoParquet files and metadata.
    file_start_time (str or pd.Timestamp): Start timestamp for the query range.
    file_end_time (str or pd.Timestamp): End timestamp for the query range.
    survey_id (str, optional): Identifier for the survey in Azure Blob Storage.
    container_name (str, optional): Container name in Azure Blob Storage.

    Returns:
    GeoDataFrame: Combined GeoDataFrame with location points within the specified timestamp range.
    """
    # Convert input timestamps to pd.Timestamp if they are strings
    file_start_time = pd.to_datetime(file_start_time)
    file_end_time = pd.to_datetime(file_end_time)

    # Load the metadata file to identify relevant partitions
    if geoparquet_path is not None:
        metadata_path = f"{geoparquet_path}/metadata.parquet"
        metadata_df = pd.read_parquet(metadata_path)
    elif container_name is not None:
        metadata_df = open_geo_parquet('metadata.parquet', container_name=container_name, survey_id=survey_id,
                                       has_geometry=False)
    else:
        raise ValueError("Either 'geoparquet_path' or 'container_name' must be provided.")

    # Filter metadata to find partitions overlapping with the query timestamp range
    relevant_partitions = metadata_df[
        (metadata_df['end_time'] >= file_start_time) &
        (metadata_df['start_time'] <= file_end_time)
        ]

    # Check if any partitions match the time range
    if relevant_partitions.empty:
        return gpd.GeoDataFrame()

    # Load and filter relevant partitions
    combined_data = []

    for _, row in relevant_partitions.iterrows():
        partition_path = row['partition_path']

        try:
            if geoparquet_path is not None:
                partition_gdf = gpd.read_parquet(partition_path)
            else:
                partition_gdf = open_geo_parquet(partition_path)

            # Filter the partition to only include records within the specified timestamp range
            filtered_gdf = partition_gdf[
                (partition_gdf['time'] >= file_start_time) &
                (partition_gdf['time'] <= file_end_time)
                ]

            combined_data.append(filtered_gdf)
        except Exception as e:
            print(f"Error loading partition {partition_path}: {e}")
            continue

    # Combine all filtered GeoDataFrames into a single GeoDataFrame
    result_gdf = gpd.GeoDataFrame(pd.concat(combined_data, ignore_index=True))

    return result_gdf


def extract_start_end_coordinates(result_gdf):
    """
    Extracts the starting and ending latitude and longitude from a GeoDataFrame
    based on the earliest and latest timestamps.

    Parameters:
    result_gdf (GeoDataFrame): The GeoDataFrame returned by query_location_points_between_timestamps.

    Returns:
    dict: A dictionary containing start and end lat/lon coordinates.
    """
    # Ensure the GeoDataFrame is not empty
    if result_gdf.empty:
        raise ValueError("The provided GeoDataFrame is empty.")

    # Sort by 'time' to get the start and end coordinates
    sorted_gdf = result_gdf.sort_values(by='time')

    # Extract coordinates for the start (first row) and end (last row)
    file_start_lat = sorted_gdf.iloc[0]['latitude']
    file_start_lon = sorted_gdf.iloc[0]['longitude']
    file_end_lat = sorted_gdf.iloc[-1]['latitude']
    file_end_lon = sorted_gdf.iloc[-1]['longitude']

    # Return the coordinates in a dictionary
    return {
        'file_start_lat': file_start_lat,
        'file_start_lon': file_start_lon,
        'file_end_lat': file_end_lat,
        'file_end_lon': file_end_lon
    }

