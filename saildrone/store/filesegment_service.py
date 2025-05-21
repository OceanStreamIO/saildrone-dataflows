from shapely.geometry import LineString
from typing import Optional, List
from datetime import datetime

from saildrone.store import PostgresDB


class FileSegmentService:
    def __init__(self, db: PostgresDB) -> None:
        """
        Initialize the service with a database connection.

        Parameters
        ----------
        db : PostgresDB
            The database connection object.
        """
        self.db = db
        self.table_name = 'files'

    def is_file_processed(self, file_name: str) -> bool:
        """
        Check if a file has already been processed.

        Parameters
        ----------
        file_name : str
            The name of the file to check.

        Returns
        -------
        bool
            Returns True if the file has already been processed, False otherwise.
        """
        self.db.cursor.execute(f'SELECT id FROM {self.table_name} WHERE file_name=%s AND processed=TRUE', (file_name,))
        return self.db.cursor.fetchone() is not None

    def get_file_info(self, file_name: str):
        """
        Get information about a file from the database.

        Parameters
        ----------
        file_name : str
            The name of the file to check.

        Returns
        -------
        dict
            A dictionary containing information about the file.
        """
        self.db.cursor.execute(f'SELECT id, size, converted, processed FROM {self.table_name} WHERE file_name=%s', (file_name,))
        row = self.db.cursor.fetchone()

        if row:
            return {'id': row[0], 'size': row[1], 'converted': row[2], 'processed': row[3]}

        return None

    def is_file_converted(self, file_name: str) -> bool:
        """
        Check if a file has already been converted.

        Parameters
        ----------
        file_name : str
            The name of the file to check.

        Returns
        -------
        bool
            Returns True if the file has already been converted, False otherwise.
        """
        self.db.cursor.execute(f'SELECT id FROM {self.table_name} WHERE file_name=%s AND converted=TRUE', (file_name,))
        return self.db.cursor.fetchone() is not None

    def is_file_failed(self, file_name: str) -> bool:
        """
        Check if a file has already been converted.

        Parameters
        ----------
        file_name : str
            The name of the file to check.

        Returns
        -------
        bool
            Returns True if the file has already been converted, False otherwise.
        """
        self.db.cursor.execute(f'SELECT id FROM {self.table_name} WHERE file_name=%s AND failed=TRUE', (file_name,))
        return self.db.cursor.fetchone() is not None

    def is_file_downloaded(self, file_name: str, survey_id: int) -> bool:
        """
        Check if a file has already been converted.

        Parameters
        ----------
        file_name : str
            The name of the file to check.
        survey_id : int
            The ID of the survey in the database.

        Returns
        -------
        bool
            Returns True if the file has already been converted, False otherwise.
        """
        self.db.cursor.execute(f'SELECT id FROM {self.table_name} WHERE file_name=%s AND downloaded=TRUE AND survey_db_id=%s',
                               (file_name,survey_id,))

        return self.db.cursor.fetchone() is not None

    def insert_file_record(
        self,
        file_name: str,
        size: Optional[int] = None,
        location: Optional[str] = None,
        converted: Optional[bool] = None,
        last_modified: Optional[str] = None,
        file_npings: Optional[int] = None,
        file_nsamples: Optional[int] = None,
        file_start_time: Optional[str] = None,
        file_end_time: Optional[str] = None,
        file_freqs: Optional[str] = None,
        file_start_depth: Optional[float] = None,
        file_end_depth: Optional[float] = None,
        file_start_lat: Optional[float] = None,
        file_start_lon: Optional[float] = None,
        file_end_lat: Optional[float] = None,
        file_end_lon: Optional[float] = None,
        echogram_files: Optional[List[str]] = None,
        failed: Optional[bool] = None,
        error_details: Optional[str] = None,
        location_data: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
        survey_db_id: Optional[int] = None,
        downloaded: Optional[bool] = None,
    ) -> int:
        """
        Insert a new file record into the database.

        Parameters
        ----------
        file_name : str
            The name of the file.
        size : Optional[int]
            The size of the file in bytes.
        location : Optional[str]
            The file path or location.
        converted : Optional[bool]
            Whether the file has been converted.
        last_modified : Optional[str]
            The last modified timestamp of the file.
        file_npings : Optional[int]
            Number of pings in the file.
        file_nsamples : Optional[int]
            Number of samples in the file.
        file_start_time : Optional[str]
            The start time of the file.
        file_end_time : Optional[str]
            The end time of the file.
        file_freqs : Optional[str]
            Frequencies present in the file.
        file_start_depth : Optional[float]
            The start depth of the file.
        file_end_depth : Optional[float]
            The end depth of the file.
        file_start_lat : Optional[float]
            The starting latitude of the file.
        file_start_lon : Optional[float]
            The starting longitude of the file.
        file_end_lat : Optional[float]
            The ending latitude of the file.
        file_end_lon : Optional[float]
            The ending longitude of the file.
        echogram_files : Optional[List[str]]
            A list of paths to the echogram files associated with this file.
        failed : Optional[bool]
            Whether the file processing failed.
        error_details : Optional[str]
            Details of the error that occurred during processing.
        location_data : Optional[str]
            The location data extracted from the file.

        Returns
        -------
        int
            The ID of the newly inserted file record.

        Args:
            converted:
            converted:
        """
        self.db.cursor.execute('''
            INSERT INTO files (
                file_name, size, location, processed, converted, last_modified, file_npings, file_nsamples, file_start_time, 
                file_end_time, file_freqs, file_start_depth, file_end_depth, file_start_lat, file_start_lon, 
                file_end_lat, file_end_lon, echogram_files, failed, error_details, location_data, processing_time_ms, 
                survey_db_id, downloaded
            ) VALUES (%s, %s, %s, FALSE, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
        ''', (file_name, size, location, converted, last_modified, file_npings, file_nsamples, file_start_time,
              file_end_time, file_freqs, file_start_depth, file_end_depth, file_start_lat, file_start_lon, file_end_lat,
              file_end_lon, echogram_files, failed, error_details, location_data, processing_time_ms, survey_db_id, downloaded))
        file_id = self.db.cursor.fetchone()[0]
        self.db.conn.commit()
        return file_id

    def update_file_record(
        self,
        file_id: int,
        cruise_id: Optional[int] = None,
        file_name: Optional[str] = None,
        size: Optional[int] = None,
        processed: Optional[bool] = None,
        converted: Optional[bool] = None,
        location: Optional[str] = None,
        last_modified: Optional[str] = None,
        file_npings: Optional[int] = None,
        file_nsamples: Optional[int] = None,
        file_start_time: Optional[str] = None,
        file_end_time: Optional[str] = None,
        file_freqs: Optional[str] = None,
        file_start_depth: Optional[float] = None,
        file_end_depth: Optional[float] = None,
        file_start_lat: Optional[float] = None,
        file_start_lon: Optional[float] = None,
        file_end_lat: Optional[float] = None,
        file_end_lon: Optional[float] = None,
        distance: Optional[float] = None,
        echogram_files: Optional[List[str]] = None,
        failed: Optional[bool] = None,
        denoised: Optional[bool] = None,
        seabed_mask: Optional[bool] = None,
        error_details: Optional[str] = None,
        location_data: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
        survey_db_id: Optional[int] = None
    ) -> None:

        self.db.cursor.execute(f'''
            UPDATE {self.table_name}
            SET file_name = COALESCE(%s, file_name),
                size = COALESCE(%s, size),
                processed = COALESCE(%s, processed),
                converted = COALESCE(%s, converted),
                location = COALESCE(%s, location),
                last_modified = COALESCE(%s, last_modified),
                file_npings = COALESCE(%s, file_npings),
                file_nsamples = COALESCE(%s, file_nsamples),
                file_start_time = COALESCE(%s, file_start_time),
                file_end_time = COALESCE(%s, file_end_time),
                file_freqs = COALESCE(%s, file_freqs),
                file_start_depth = COALESCE(%s, file_start_depth),
                file_end_depth = COALESCE(%s, file_end_depth),
                file_start_lat = COALESCE(%s, file_start_lat),
                file_start_lon = COALESCE(%s, file_start_lon),
                file_end_lat = COALESCE(%s, file_end_lat),
                file_end_lon = COALESCE(%s, file_end_lon),
                distance = COALESCE(%s, distance),
                echogram_files = COALESCE(%s, echogram_files),
                failed = COALESCE(%s, failed),
                denoised = COALESCE(%s, denoised),
                seabed_mask = COALESCE(%s, seabed_mask),
                error_details = COALESCE(%s, error_details),
                location_data = COALESCE(%s, location_data),
                processing_time_ms = COALESCE(%s, processing_time_ms),
                survey_db_id = COALESCE(%s, survey_db_id)
            WHERE id = %s
        ''', (file_name, size, processed, converted, location, last_modified, file_npings, file_nsamples, file_start_time,
              file_end_time, file_freqs, file_start_depth, file_end_depth, file_start_lat, file_start_lon, file_end_lat,
              file_end_lon, distance, echogram_files, failed, denoised, seabed_mask, error_details, location_data,
              processing_time_ms, survey_db_id, file_id))
        self.db.conn.commit()

    def update_geospatial_data(
        self,
        file_id: int,
        file_start_lat: float,
        file_start_lon: float,
        file_end_lat: float,
        file_end_lon: float,
        track_geom: Optional[dict] = None
    ) -> None:
        """
        Updates geospatial fields (bounding_geom and track_geom) in the database.

        Args:
            file_id (int): The ID of the file to update.
            file_start_lat (float): Start latitude of the track.
            file_start_lon (float): Start longitude of the track.
            file_end_lat (float): End latitude of the track.
            file_end_lon (float): End longitude of the track.
            track_geom (dict, optional): Track geometry as a LineString GeoJSON object.
                                          If None, track_geom is not updated.

        Returns:
            None
        """
        # Ensure start/end lat/lon are provided
        if any(val is None for val in [file_start_lat, file_start_lon, file_end_lat, file_end_lon]):
            raise ValueError("Start and end lat/lon must be provided and cannot be None.")

        # Construct track_geom if provided
        track_geom_data = None
        if track_geom and track_geom.get("type") == "LineString":
            coordinates = track_geom.get("coordinates", [])
            if coordinates:
                line_string = LineString(coordinates)
                track_geom_data = f"SRID=4326;{line_string.wkt}"

        # Update query
        self.db.cursor.execute(f'''
            UPDATE {self.table_name}
            SET bounding_geom = ST_MakeEnvelope(%s, %s, %s, %s, 4326),
                track_geom = COALESCE(ST_GeomFromEWKT(%s), track_geom)
            WHERE id = %s
        ''', (
            file_start_lon, file_start_lat, file_end_lon, file_end_lat,
            track_geom_data, file_id
        ))

        self.db.conn.commit()

    def mark_file_processed(self, file_id: int) -> None:
        """
        Mark a file as processed by updating the 'processed' field in the database.

        Parameters
        ----------
        file_id : int
            The ID of the file to mark as processed.
        """
        self.db.cursor.execute(f'UPDATE {self.table_name} SET processed=TRUE WHERE id=%s', (file_id,))
        self.db.conn.commit()

    def update_processing_report(self, file_id: int, text: str):
        self.db.cursor.execute(f'''
            UPDATE {self.table_name}
            SET processing_report = COALESCE(%s, processing_report)
            WHERE id = %s
        ''', (text, file_id))

        self.db.conn.commit()

    def mark_file_converted(self, file_id: int) -> None:
        """
        Mark a file as processed by updating the 'converted' field in the database.

        Parameters
        ----------
        file_id : int
            The ID of the file to mark as converted.
        """
        self.db.cursor.execute(f'UPDATE {self.table_name} SET converted=TRUE WHERE id=%s', (file_id,))
        self.db.conn.commit()

    def get_files_by_polygon_and_survey(self, polygon: str, survey_id: int) -> List[tuple]:
        """
        Retrieve files matching a polygon and survey ID.

        Parameters
        ----------
        polygon : str
            The polygon geometry in WKT format.
        survey_id : int
            The ID of the survey.

        Returns
        -------
        List[tuple]
            List of files matching the query.
        """
        self.db.cursor.execute(
            f"""
            WITH region AS (
                SELECT ST_GeomFromText(%s, 4326) AS geom
            )
            SELECT location, file_name, id, location_data, file_freqs, file_start_time, file_end_time
            FROM {self.table_name}, region
            WHERE ST_Intersects(files.bounding_geom, region.geom)
              AND processed = TRUE
              AND survey_db_id = %s
            ORDER BY file_start_time ASC
            """, [polygon, survey_id]
        )

        return self.db.cursor.fetchall()

    def get_files_list_with_condition(self, survey_id: int, condition: str) -> list:
        query = f'''
            SELECT file_name
            FROM {self.table_name}
            WHERE survey_db_id = %s {condition}
            ORDER BY size ASC
        '''

        self.db.cursor.execute(query, (survey_id,))

        return [row[0] for row in self.db.cursor.fetchall()]

    def get_files_data_with_dates(self, survey_id: int, start_datetime: datetime, end_datetime: datetime) -> list:
        query = f'''
            SELECT location, file_name, id, location_data, file_freqs, file_start_time, file_end_time
            FROM {self.table_name}
            WHERE survey_db_id = %s AND file_start_time > '%s' AND file_end_time < '%s'
            ORDER BY size ASC
        '''

        self.db.cursor.execute(query, (survey_id, start_datetime, end_datetime))

        return self.db.cursor.fetchall()

    def get_files_by_survey_id(self, survey_id: int, condition: str = '') -> list:
        """
        Fetch location data for all files associated with a given survey ID.

        Parameters
        ----------
        survey_id : int
            The ID of the survey to fetch location data for.
        condition : str
            Optional condition to filter the files.

        Returns
        -------
        list
            A list of location_data dictionaries from the database.
        """

        query = f'''
            SELECT location, file_name, id, location_data, file_freqs, file_start_time, file_end_time
            FROM {self.table_name}
            WHERE survey_db_id = %s AND processed = TRUE {condition}
            ORDER BY file_start_time ASC
        '''

        self.db.cursor.execute(query, (survey_id,))
        rows = self.db.cursor.fetchall()

        return [
            {
                'location': row[0],
                'file_name': row[1],
                'id': row[2],
                'location_data': row[3],
                'file_freqs': row[4],
                'file_start_time': row[5],
                'file_end_time': row[6]
            }
            for row in rows
        ]

    def file_has_location_data(self, file_id: int) -> bool:
        """
        Check if a file has associated location data.

        Parameters
        ----------
        file_id : int
            The ID of the file to check.

        Returns
        -------
        bool
            True if the file has location data, False otherwise.
        """
        self.db.cursor.execute(f'SELECT location_data FROM {self.table_name} WHERE id=%s', (file_id,))
        return self.db.cursor.fetchone()[0] is not None
