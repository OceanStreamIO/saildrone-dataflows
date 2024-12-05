from saildrone.store import PostgresDB
from typing import Optional, List


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
        self.db.cursor.execute('SELECT id FROM files WHERE file_name=%s AND processed=TRUE', (file_name,))
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
        self.db.cursor.execute('SELECT id, size, converted, processed FROM files WHERE file_name=%s', (file_name,))
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
        self.db.cursor.execute('SELECT id FROM files WHERE file_name=%s AND converted=TRUE', (file_name,))
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
        self.db.cursor.execute('SELECT id FROM files WHERE file_name=%s AND downloaded=TRUE AND survey_db_id=%s',
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
        echogram_files: Optional[List[str]] = None,
        failed: Optional[bool] = None,
        error_details: Optional[str] = None,
        location_data: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
        survey_db_id: Optional[int] = None,
    ) -> None:
        """
        Update an existing file record in the database.

        Parameters
        ----------
        file_id : int
            The ID of the file to update.
        file_name : Optional[str]
            The name of the file.
        size : Optional[int]
            The size of the file in bytes.
        processed : Optional[bool]
            Whether the file has been processed.
        location : Optional[str]
            The file path or location.
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
        """

        bounding_geom = (
            f'ST_MakeEnvelope({file_start_lon}, {file_start_lat}, {file_end_lon}, {file_end_lat}, 4326)'
            if file_start_lat is not None and file_start_lon is not None and file_end_lat is not None and file_end_lon is not None
            else None
        )

        self.db.cursor.execute('''
            UPDATE files
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
                echogram_files = COALESCE(%s, echogram_files),
                failed = COALESCE(%s, failed),
                error_details = COALESCE(%s, error_details),
                location_data = COALESCE(%s, location_data),
                processing_time_ms = COALESCE(%s, processing_time_ms),
                survey_db_id = COALESCE(%s, survey_db_id),
                bounding_geom = COALESCE(%s, bounding_geom)
            WHERE id = %s
        ''', (file_name, size, processed, converted, location, last_modified, file_npings, file_nsamples, file_start_time,
              file_end_time, file_freqs, file_start_depth, file_end_depth, file_start_lat, file_start_lon, file_end_lat,
              file_end_lon, echogram_files, failed, error_details, location_data, processing_time_ms, survey_db_id,
              bounding_geom, file_id))
        self.db.conn.commit()

    def mark_file_processed(self, file_id: int) -> None:
        """
        Mark a file as processed by updating the 'processed' field in the database.

        Parameters
        ----------
        file_id : int
            The ID of the file to mark as processed.
        """
        self.db.cursor.execute('UPDATE files SET processed=TRUE WHERE id=%s', (file_id,))
        self.db.conn.commit()

    def mark_file_converted(self, file_id: int) -> None:
        """
        Mark a file as processed by updating the 'converted' field in the database.

        Parameters
        ----------
        file_id : int
            The ID of the file to mark as converted.
        """
        self.db.cursor.execute('UPDATE files SET converted=TRUE WHERE id=%s', (file_id,))
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
            """
            WITH region AS (
                SELECT ST_GeomFromText(%s, 4326) AS geom
            )
            SELECT location, file_name, id, location_data, file_freqs, file_start_time, file_end_time
            FROM files, region
            WHERE ST_Intersects(files.bounding_geom, region.geom)
              AND processed = TRUE
              AND survey_db_id = %s
            ORDER BY file_start_time ASC
            """,
            [polygon, survey_id]
        )
        return self.db.cursor.fetchall()