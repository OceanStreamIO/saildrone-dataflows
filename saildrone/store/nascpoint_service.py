import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

from saildrone.store import PostgresDB


class NASCPointService:
    def __init__(self, db: PostgresDB) -> None:
        """
        Initialize the service with a database connection.

        Parameters
        ----------
        db : PostgresDB
            The database connection object.
        """
        self.db = db
        self.table_name_3d = 'nasc_values'
        self.table_name_2d = 'nasc_points_2d'

    def _delete_existing_nasc(self, file_id: int):
        """
        Delete existing NASC entries for a given file_id from both 3D and 2D tables.
        """
        for table in [self.table_name_3d, self.table_name_2d]:
            self.db.cursor.execute(f"DELETE FROM {table} WHERE file_id = %s;", (file_id,))
        self.db.conn.commit()

    def insert_nasc_points(self, file_id: int, survey_id: int, ds_NASC, average: bool = False, clearTables: bool = False):
        """
        Insert NASC values from an xarray.Dataset into the database.

        Parameters
        ----------
        file_id : int
            The ID of the file segment.
        survey_id : int
            The ID of the survey.
        ds_NASC : xarray.Dataset
            The NASC dataset containing 'latitude', 'longitude', 'depth', 'ping_time', and 'NASC'.
        average : bool
            If True, insert depth-averaged values (2D). Otherwise, insert full-depth values (3D).
        clearTables : bool
            If True, delete existing entries for the file_id before inserting new ones.
        """

        if clearTables:
            self._delete_existing_nasc(file_id)

        # Get bounds for latitude and longitude
        lat_min = ds_NASC.attrs["geospatial_lat_min"]
        lat_max = ds_NASC.attrs["geospatial_lat_max"]
        lon_min = ds_NASC.attrs["geospatial_lon_min"]
        lon_max = ds_NASC.attrs["geospatial_lon_max"]

        # Calculate midpoint latitude and longitude (since distance = 1)
        avg_lat = (lat_min + lat_max) / 2
        avg_lon = (lon_min + lon_max) / 2

        # Extract variables from the dataset
        ping_time = ds_NASC["ping_time"].values[0]  # There's only one distance → one ping_time
        nasc_values = ds_NASC["NASC"].values[0, 0, :]  # Shape: (channel, distance, depth)
        depths = ds_NASC["depth"].values

        ping_time = pd.Timestamp(ping_time).round('us').to_pydatetime() # Handles numpy.datetime64 to datetime conversion

        # Prepare records for insertion
        records = [
            (file_id, survey_id, ping_time, float(depth), float(nasc), f"POINT({avg_lon} {avg_lat})")
            for depth, nasc in zip(depths, nasc_values)
            if not np.isnan(nasc)  # Skip NaN values
        ]

        if not records:
            print("No NASC values to insert.")
            return

        if average:
            valid = ~np.isnan(nasc_values)
            nasc_avg = float(np.nanmean(nasc_values))
            avg_depth = float(np.nanmean(depths[valid]))
            records = [(file_id, survey_id, ping_time, avg_depth, nasc_avg, f"POINT({avg_lon} {avg_lat})")]
            insert_query = f"""
                INSERT INTO nasc_points_2d (file_id, survey_id, ping_time, avg_depth, nasc_value_avg, geom)
                VALUES %s;
            """
        else:
            # Bulk insert using `execute_values`
            insert_query = f"""
                INSERT INTO {self.table_name_3d} (file_id, survey_id, ping_time, depth, nasc_value, geom)
                VALUES %s;
            """

        execute_values(self.db.cursor, insert_query, records)
        self.db.conn.commit()

    def get_nasc_points_by_file(self, file_id: int):
        """
        Retrieve NASC points for a specific file ID.

        Parameters
        ----------
        file_id : int
            The ID of the file.

        Returns
        -------
        list
            A list of dictionaries containing NASC points data.
        """
        self.db.cursor.execute(f"""
            SELECT file_id, survey_id, ping_time, depth, nasc_value, ST_AsText(geom) AS geom
            FROM {self.table_name_3d}
            WHERE file_id = %s;
        """, (file_id,))
        rows = self.db.cursor.fetchall()

        return [
            {
                "file_id": row[0],
                "survey_id": row[1],
                "ping_time": row[2].isoformat(),
                "depth": row[3],
                "nasc_value": row[4],
                "geometry": row[5]
            }
            for row in rows
        ]

    def get_avg_nasc_point_by_file(self, file_id: int):
        """
        Retrieve the depth-averaged NASC point for a specific file ID.

        Since nasc_points_2d stores one entry per file, this fetches that record directly.

        Parameters
        ----------
        file_id : int
            The ID of the file.

        Returns
        -------
        dict or None
            Dictionary containing NASC point data, or None if not found.
        """
        self.db.cursor.execute(f"""
            SELECT file_id, survey_id, ping_time, avg_depth, nasc_value, ST_AsText(geom) AS geom
            FROM {self.table_name_2d}
            WHERE file_id = %s;
        """, (file_id,))
        row = self.db.cursor.fetchone()

        if not row:
            return None

        return {
            "file_id": row[0],
            "survey_id": row[1],
            "ping_time": row[2].isoformat(),
            "avg_depth": row[3],
            "nasc_value": row[4],
            "geometry": row[5]
        }
