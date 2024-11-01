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

    def get_file_info(self, file_name: str) -> dict:
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
        self.db.cursor.execute('SELECT * FROM files WHERE file_name=%s AND processed=TRUE', (file_name,))
        return self.db.cursor.fetchone()

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

    def insert_file_record(
        self,
        file_name: str,
        size: Optional[int] = None,
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
        echogram_files: Optional[List[str]] = None
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

        Returns
        -------
        int
            The ID of the newly inserted file record.
        """
        self.db.cursor.execute('''
            INSERT INTO files (
                file_name, size, location, processed, converted, last_modified, file_npings, file_nsamples, file_start_time, 
                file_end_time, file_freqs, file_start_depth, file_end_depth, file_start_lat, file_start_lon, 
                file_end_lat, file_end_lon, echogram_files
            ) VALUES (%s, %s, %s, FALSE, FALSE, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
        ''', (file_name, size, location, last_modified, file_npings, file_nsamples, file_start_time, file_end_time,
              file_freqs, file_start_depth, file_end_depth, file_start_lat, file_start_lon, file_end_lat, file_end_lon, echogram_files))
        file_id = self.db.cursor.fetchone()[0]
        self.db.conn.commit()
        return file_id

    def update_file_record(
        self,
        file_id: int,
        file_name: Optional[str] = None,
        size: Optional[int] = None,
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
        echogram_files: Optional[List[str]] = None
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
        self.db.cursor.execute('''
            UPDATE files
            SET file_name = COALESCE(%s, file_name),
                size = COALESCE(%s, size),
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
                echogram_files = COALESCE(%s, echogram_files)
            WHERE id = %s
        ''', (file_name, size, location, last_modified, file_npings, file_nsamples, file_start_time, file_end_time,
              file_freqs, file_start_depth, file_end_depth, file_start_lat, file_start_lon, file_end_lat, file_end_lon, echogram_files, file_id))
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
