import logging
import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr

from haversine import haversine
from scipy.signal import savgol_filter


logger = logging.getLogger(__name__)


def extract_location_data(gdf: gpd.GeoDataFrame, epsilon=0.00001, min_distance=0.01) -> pd.DataFrame:
    """
    Extract location data (GPS coordinates) from a GeoDataFrame.

    Parameters:
    - gdf: gpd.GeoDataFrame
        The GeoDataFrame containing GPS data with 'geometry', 'latitude', 'longitude', and 'time' columns.
    - epsilon: float
        Epsilon parameter for the Ramer-Douglas-Peucker algorithm.
    - min_distance: float
        Minimum distance between points (in nautical miles).

    Returns:
    - pd.DataFrame: A DataFrame containing the extracted GPS data.
    """

    required_columns = ["geometry", "latitude", "longitude", "time"]
    for col in required_columns:
        if col not in gdf.columns:
            raise ValueError(f"GeoDataFrame is missing required column: {col}")

    df = gdf.copy()
    df = df.rename(columns={"latitude": "lat", "longitude": "lon", "time": "dt"})

    df = df.dropna(subset=["lat", "lon", "dt"])
    df = df[(df["lat"] >= -90) & (df["lat"] <= 90) & (df["lon"] >= -180) & (df["lon"] <= 180)]

    if df.empty:
        return pd.DataFrame(columns=["lat", "lon", "dt", "knt"])

    # Apply smoothing filters
    window_size = min(11, len(df))  # Window size for smoothing filter
    poly_order = 2  # Polynomial order for smoothing filter

    if len(df) > window_size:
        df["lat"] = savgol_filter(df["lat"], window_size, poly_order)
        df["lon"] = savgol_filter(df["lon"], window_size, poly_order)

    # Calculate distance and speed
    df["distance"] = [
        haversine(
            (df["lat"].iloc[i], df["lon"].iloc[i - 1]),
            (df["lat"].iloc[i - 1], df["lon"].iloc[i]),
            unit="nmi",
        )
        if i > 0 else 0 for i in range(len(df))
    ]
    df["time_interval"] = df["dt"] - df["dt"].shift()
    df["knt"] = (df["distance"] / df["time_interval"].dt.total_seconds()) * 3600
    df = df[["lat", "lon", "dt", "knt"]]

    # Remove unrealistic speed values
    df = df[df["knt"] < 100]

    # Apply Ramer-Douglas-Peucker algorithm for thinning coordinates
    points = df[["lat", "lon"]].values
    thinned_points = ramer_douglas_peucker(points, epsilon)
    thinned_df = pd.DataFrame(thinned_points, columns=["lat", "lon"])

    try:
        thinned_df["dt"] = thinned_df.apply(
            lambda row: df.loc[(df["lat"] == row["lat"]) & (df["lon"] == row["lon"]), "dt"].values[0], axis=1)
        thinned_df["knt"] = thinned_df.apply(
            lambda row: df.loc[(df["lat"] == row["lat"]) & (df["lon"] == row["lon"]), "knt"].values[0], axis=1)
    except Exception:
        logger.warning("Error in extracting time and speed data for thinned points.")

    # Further thin by minimum distance
    final_points = [thinned_df.iloc[0]]
    for i in range(1, len(thinned_df)):
        if haversine((final_points[-1]["lat"], final_points[-1]["lon"]),
                     (thinned_df.iloc[i]["lat"], thinned_df.iloc[i]["lon"]), unit="nmi") >= min_distance:
            final_points.append(thinned_df.iloc[i])

    final_df = pd.DataFrame(final_points)
    return final_df


def ramer_douglas_peucker(points, epsilon):
    if len(points) < 3:
        return points

    def get_perpendicular_distance(point, line_start, line_end):
        if np.allclose(line_start, line_end):
            return np.linalg.norm(point - line_start)

        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        point_vec_scaled = point_vec / line_len
        t = np.dot(line_unitvec, point_vec_scaled)
        t = np.clip(t, 0, 1)
        nearest = line_start + t * line_vec
        return np.linalg.norm(point - nearest)

    max_distance = 0
    index = 0
    for i in range(1, len(points) - 1):
        distance = get_perpendicular_distance(points[i], points[0], points[-1])
        if distance > max_distance:
            index = i
            max_distance = distance

    if max_distance > epsilon:
        left_points = ramer_douglas_peucker(points[:index + 1], epsilon)
        right_points = ramer_douglas_peucker(points[index:], epsilon)
        return np.vstack((left_points[:-1], right_points))

    return np.vstack((points[0], points[-1]))


def extract_start_end_lat_lon(ds: xr.Dataset) -> dict[str, float]:
    """
    Extract start and end latitude and longitude from an Xarray dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset containing latitude and longitude variables.
        Expected to have 'ping_time' as a dimension.

    Returns
    -------
    dict[str, float]
        A dictionary containing start and end coordinates with keys:
        - file_start_lat: Starting latitude
        - file_end_lat: Ending latitude
        - file_start_lon: Starting longitude
        - file_end_lon: Ending longitude
        Returns empty dict if coordinates are not available or invalid.

    Notes
    -----
    - Handles NaN values by finding first/last valid coordinates
    - Validates coordinate ranges: latitude [-90, 90], longitude [-180, 180]
    """
    try:
        # Verify required variables exist
        if not all(var in ds.data_vars for var in ["latitude", "longitude"]):
            logging.warning("Dataset missing latitude or longitude variables")
            return {}

        if "ping_time" not in ds.dims:
            logging.warning("Dataset missing ping_time dimension")
            return {}

        # Get coordinate arrays
        lat = ds["latitude"]
        lon = ds["longitude"]

        # Find first and last valid values, skipping NaNs
        start_idx = 0
        end_idx = -1
        
        while start_idx < len(lat) and (np.isnan(lat[start_idx].values) or np.isnan(lon[start_idx].values)):
            start_idx += 1
            
        while end_idx > -len(lat) and (np.isnan(lat[end_idx].values) or np.isnan(lon[end_idx].values)):
            end_idx -= 1

        if start_idx >= len(lat) or abs(end_idx) > len(lat):
            logging.warning("No valid coordinate pairs found in dataset")
            return {}

        # Extract coordinates
        file_start_lat = float(lat.isel(ping_time=start_idx).values)
        file_end_lat = float(lat.isel(ping_time=end_idx).values)
        file_start_lon = float(lon.isel(ping_time=start_idx).values)
        file_end_lon = float(lon.isel(ping_time=end_idx).values)

        # Validate coordinate ranges
        if not (-90 <= file_start_lat <= 90 and -90 <= file_end_lat <= 90):
            logging.warning(f"Invalid latitude values: {file_start_lat}, {file_end_lat}")
            return {}
            
        if not (-180 <= file_start_lon <= 180 and -180 <= file_end_lon <= 180):
            logging.warning(f"Invalid longitude values: {file_start_lon}, {file_end_lon}")
            return {}

        return {
            "file_start_lat": file_start_lat,
            "file_end_lat": file_end_lat,
            "file_start_lon": file_start_lon,
            "file_end_lon": file_end_lon,
        }
        
    except Exception as e:
        logging.error(f"Error extracting coordinates: {str(e)}")
        return {}


def select_location_points(location_data: pd.DataFrame, num_points: int) -> pd.DataFrame:
    """
    Select points at regular intervals from the location data.

    Parameters:
    - location_data: pd.DataFrame
        DataFrame containing the extracted location data.
    - num_points: int
        Number of points to select.

    Returns:
    - pd.DataFrame: A DataFrame containing the selected points.
    """
    total_points = len(location_data)

    if total_points <= num_points:
        return location_data

    # Calculate the interval between points
    interval = max(1, total_points // num_points)

    # Select points at regular intervals
    selected_indices = list(range(0, total_points, interval))[:num_points]
    selected_points = location_data.iloc[selected_indices]

    return selected_points


def create_location_message(point: pd.Series) -> dict:
    """
    Create a message dictionary for a single location point.

    Parameters:
    - point: pd.Series
        A row from the location DataFrame.

    Returns:
    - dict: A dictionary containing the location message to send to IoT Hub.
    """
    return {
        "location": {
            "lat": point["lat"],
            "lon": point["lon"]
        },
        "timestamp": point["dt"].isoformat() if isinstance(point["dt"], pd.Timestamp) else str(point["dt"]),
        "speed_knots": point["knt"]
    }