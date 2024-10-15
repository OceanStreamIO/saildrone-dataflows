import os
import fsspec
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from store import get_azure_blob_filesystem


def consolidate_csv_to_geoparquet_partitioned(folder_path, output_path, storage_type='local'):
    """
    Consolidates multiple CSV files in a folder into partitioned GeoParquet files.

    Parameters:
    folder_path (str): Path to the folder containing CSV files.
    output_directory (str): Path where the partitioned GeoParquet files will be saved.
    """
    combined_data = []

    if storage_type == 'azure':
        fs = get_azure_blob_filesystem()
    else:
        fs = fsspec.filesystem('file')

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

    # Combine all GeoDataFrames into a single GeoDataFrame
    combined_gdf = pd.concat(combined_data, ignore_index=True)

    combined_gdf = combined_gdf.dropna(subset=['longitude', 'latitude'])

    # Define grid size (e.g., 1 degree)
    grid_size = 1.0  # degrees

    # Calculate grid indices for partitioning
    combined_gdf['lon_grid'] = (combined_gdf['longitude'] // grid_size).astype('int32')
    combined_gdf['lat_grid'] = (combined_gdf['latitude'] // grid_size).astype('int32')

    # Group by partitioning columns and write each group to a separate Parquet file
    grouped = combined_gdf.groupby(['lon_grid', 'lat_grid'])

    for (lon_grid, lat_grid), group in grouped:
        if storage_type == 'azure':
            partition_path = f"{output_path}/lon_grid={lon_grid}/lat_grid={lat_grid}/data.parquet"
        else:
            partition_path = os.path.join(output_path, f'lon_grid={lon_grid}', f'lat_grid={lat_grid}', 'data.parquet')
            local_directory = os.path.dirname(partition_path)
            os.makedirs(local_directory, exist_ok=True)

        group_to_save = group.drop(columns=['lon_grid', 'lat_grid'])
        with fs.open(partition_path, 'wb') as f:
            group_to_save.to_parquet(f, index=False)

