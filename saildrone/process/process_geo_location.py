import geopandas as gpd
import pandas as pd

from .location import extract_location_data
from shapely.geometry import Point
from datetime import datetime, timedelta


def process_geo_location(file_name, geo_data, metadata):
    geometry = geo_data.get("geometry", {})
    coordinates = geometry.get("coordinates", [])

    # Ensure coordinates exist
    if not coordinates or geometry.get("type") != "LineString":
        print(f"No valid geo_location found for file: {file_name}")
        return None

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


def create_geodataframe_from_location_data(location_data_list):
    """
    Creates a GeoDataFrame from location data exported from the database.

    Parameters
    ----------
    location_data_list : list
        A list of location data dictionaries fetched from the database.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the merged location data.
    """
    all_data = []

    # Flatten and process location data
    for location_data in location_data_list:
        for record in location_data:
            all_data.append({
                'latitude': record['lat'],
                'longitude': record['lon'],
                'time': pd.to_datetime(record['dt']),
                'speed_knots': record.get('knt')
            })

    # Create a DataFrame
    df = pd.DataFrame(all_data)

    # Convert to GeoDataFrame
    df['geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

    return gdf
