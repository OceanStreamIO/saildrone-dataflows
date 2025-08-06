import os
import psycopg2
from dotenv import load_dotenv
from typing import Optional


class PostgresDB:
    def __init__(self) -> None:
        self.conn = None
        self.cursor = None

    def __enter__(self) -> 'PostgresDB':
        """
        Enter the runtime context related to this object. This method establishes a connection to the database
        and returns the database object.

        Returns
        -------
        PostgresDB
            The database instance itself for context management.
        """
        load_dotenv()

        self.conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            dbname=os.getenv('DB_NAME', ''),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )
        self.cursor = self.conn.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.conn:
            self.conn.close()

    def create_tables(self) -> None:
        """
        Create the necessary tables for storing file and survey information if they do not exist.

        Tables:
        - `files`: Tracks each file processed, including its size, location, and whether it has been processed.
        - `surveys`: Tracks survey-level metadata, such as the cruise ID, vessel, start/end dates, ports, and description.
        """
        # Create the files table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                id SERIAL PRIMARY KEY,
                file_name TEXT,
                size INTEGER,
                location TEXT,
                processed BOOLEAN DEFAULT FALSE,
                converted BOOLEAN DEFAULT FALSE,
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_npings INTEGER,
                file_nsamples INTEGER,
                file_start_time TIMESTAMP,
                file_end_time TIMESTAMP,
                file_freqs TEXT,
                file_start_depth REAL,
                file_end_depth REAL,
                file_start_lat REAL,
                file_start_lon REAL,
                file_end_lat REAL,
                file_end_lon REAL,
                echogram_files TEXT[],
                failed BOOLEAN DEFAULT FALSE,
                denoised BOOLEAN DEFAULT FALSE,
                error_details TEXT,
                location_data JSON,
                processing_time_ms INTEGER,
                survey_db_id INTEGER,
                downloaded BOOLEAN DEFAULT FALSE,
                bounding_geom GEOMETRY(POLYGON, 4326),
                track_geom GEOMETRY(LINESTRING, 4326),
                processing_report TEXT
            );
        ''')

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS files_bounding_geom_idx ON files USING GIST (bounding_geom);
        ''')

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_files_track_geom ON files USING GIST(track_geom);
        ''')

        # Create the surveys table with more detailed metadata
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS surveys (
                id SERIAL PRIMARY KEY,
                cruise_id TEXT NOT NULL,
                survey_name TEXT NOT NULL,
                vessel TEXT NOT NULL,
                start_port TEXT NOT NULL,
                end_port TEXT NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                description TEXT
            );
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS tenants (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT
            );
        ''')

        # Create platforms table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS platforms (
                id SERIAL PRIMARY KEY,
                tenant_id INTEGER REFERENCES tenants(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                type TEXT NOT NULL CHECK (type IN ('vessel', 'glider', 'auv', 'buoy', 'drifter', 'other')),
                manufacturer TEXT,
                model TEXT,
                operating_depth REAL,
                description TEXT
            );
        ''')

        # Create instrument types table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS instrument_types (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                manufacturer TEXT,
                model TEXT,
                calibration_date DATE,
                operating_depth REAL,
                measurement_type TEXT NOT NULL CHECK (measurement_type IN ('acoustic', 'optical', 'physical', 'chemical', 'other')),
                frequency_range TEXT,
                description TEXT
            );
        ''')

        # Create instruments table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS instruments (
                id SERIAL PRIMARY KEY,
                instrument_type_id INTEGER REFERENCES instrument_types(id) ON DELETE CASCADE,
                survey_id INTEGER REFERENCES surveys(id) ON DELETE CASCADE,
                serial_number TEXT NOT NULL,
                deployment_start DATE NOT NULL,
                deployment_end DATE,
                deployment_notes TEXT,
                description TEXT
            );
        ''')

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS nasc_values (
                id SERIAL PRIMARY KEY,
                file_id INT NOT NULL,
                survey_id INT NOT NULL,
                ping_time TIMESTAMP NOT NULL,
                depth FLOAT NOT NULL,
                nasc_value FLOAT NOT NULL,
                geom GEOMETRY(Point, 4326) NOT NULL
            );
        """)
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_id ON nasc_values (file_id);")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_survey_id ON nasc_values (survey_id);")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_geom ON nasc_values USING GIST (geom);")

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS nasc_points_2d (
                id SERIAL PRIMARY KEY,
                file_id INT NOT NULL,
                survey_id INT NOT NULL,
                ping_time TIMESTAMP NOT NULL,
                avg_depth FLOAT NOT NULL,
                nasc_value_avg FLOAT NOT NULL,
                geom GEOMETRY(Point, 4326) NOT NULL
            );
        """)
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_id ON nasc_points_2d (file_id);")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_survey_id ON nasc_points_2d (survey_id);")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_geom ON nasc_points_2d USING GIST (geom);")

        self.cursor.execute(
            """
            CREATE TYPE IF NOT EXISTS agg_interval_enum AS ENUM
              ('day', 'week');

            CREATE TABLE IF NOT EXISTS exports (
                id                SERIAL PRIMARY KEY,
                container_name    TEXT                      NOT NULL,
                export_key        TEXT UNIQUE               NOT NULL,
                base_url          TEXT                      NOT NULL,
                start_date        TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                end_date          TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                num_files         INTEGER                   NOT NULL,
                denoise_params    JSONB,
                agg_params        JSONB,
                created_at        TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_exports_export_key ON exports(export_key);

            CREATE TABLE IF NOT EXISTS exports_agg_files (
                id               SERIAL PRIMARY KEY,
                created_at       TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
                file_name        VARCHAR                   NOT NULL,
                export_id        INTEGER REFERENCES exports(id) ON DELETE CASCADE,
                type             VARCHAR                   NOT NULL,
                file_start_time  TIMESTAMP WITHOUT TIME ZONE,
                file_end_time    TIMESTAMP WITHOUT TIME ZONE,
                agg_interval     agg_interval_enum,
                echogram_files   TEXT[]
            );

            CREATE TABLE IF NOT EXISTS exports_files (
                id                SERIAL PRIMARY KEY,
                export_id         INTEGER REFERENCES exports(id) ON DELETE CASCADE,
                file_id           INTEGER                   NOT NULL,
                echogram_files    TEXT[],
                sv_zarr_path      TEXT,
                denoised_zarr_path TEXT,
                netcdf_path       TEXT,
                netcdf_size       BIGINT
            );
            """
        )

        self.conn.commit()

    def is_file_processed(self, file_name: str) -> bool:
        """
        Check if a file with the given name has been processed.

        Parameters
        ----------
        file_name : str
            The name of the file to check.

        Returns
        -------
        bool
            Returns True if the file has been processed, False otherwise.
        """
        self.cursor.execute('''
            SELECT id FROM files WHERE file_name=%s AND processed=True
        ''', (file_name,))
        return self.cursor.fetchone() is not None

    def insert_file_record(self, file_name: str, size: int, location: str, last_modified: str) -> int:
        """
        Insert a new file record into the database.

        Parameters
        ----------
        file_name : str
            The name of the file.
        size : int
            The size of the file in bytes.
        location : str
            The file path or location.
        last_modified : str
            The last modified timestamp of the file.

        Returns
        -------
        int
            The ID of the newly inserted file record.
        """
        self.cursor.execute('''
            INSERT INTO files (file_name, size, location, processed, last_modified)
            VALUES (%s, %s, %s, FALSE, %s) RETURNING id
        ''', (file_name, size, location, last_modified))
        file_id = self.cursor.fetchone()[0]
        self.conn.commit()
        return file_id

    def insert_survey_record(self, cruise_id: str, survey_name: str, vessel: str, start_port: str, end_port: str,
                             start_date: str, end_date: str, description: Optional[str], days) -> int:
        """
        Insert a new survey record into the database.

        Parameters
        ----------
        cruise_id : str
            The unique cruise ID for the survey.
        survey_name : str
            The name of the survey.
        vessel : str
            The name of the vessel used in the survey.
        start_port : str
            The port where the survey starts.
        end_port : str
            The port where the survey ends.
        start_date : str
            The start date of the survey in the format YYYY-MM-DD.
        end_date : str
            The end date of the survey in the format YYYY-MM-DD.
        description : Optional[str]
            An optional description of the survey.

        Returns
        -------
        int
            The ID of the newly inserted survey record.
        """
        self.cursor.execute('''
            INSERT INTO surveys (cruise_id, survey_name, vessel, start_port, end_port, start_date, end_date, description, days)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
        ''', (cruise_id, survey_name, vessel, start_port, end_port, start_date, end_date, description, days))
        survey_id = self.cursor.fetchone()[0]
        self.conn.commit()
        return survey_id

    def empty_files_table(self) -> None:
        """
        Empty the 'files' table for testing purposes.
        """
        self.cursor.execute('TRUNCATE TABLE files RESTART IDENTITY CASCADE;')
        self.conn.commit()

    def close(self) -> None:
        """
        Close the database connection.
        """
        self.conn.close()
