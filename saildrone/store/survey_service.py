from typing import Optional

from saildrone.store import PostgresDB


class SurveyService:
    def __init__(self, db: PostgresDB) -> None:
        """
        Initialize the service with a database connection.

        Parameters
        ----------
        db : PostgresDB
            The database connection object.
        """
        self.db = db

    def get_survey_by_cruise_id(self, cruise_id: str) -> Optional[int]:
        """
        Fetch a survey by cruise_id. Returns the survey ID if found, otherwise None.

        Parameters
        ----------
        cruise_id : str
            The unique cruise ID.

        Returns
        -------
        Optional[int]
            The ID of the survey if found, or None if not found.
        """
        self.db.cursor.execute('SELECT id FROM surveys WHERE cruise_id=%s', (cruise_id,))
        result = self.db.cursor.fetchone()
        return result[0] if result else None

    def insert_survey(self, cruise_id: str, survey_name: str = '', vessel: str = '', start_port: str = '', end_port: str = '',
                      start_date: str = '', end_date: str = '', description: Optional[str] = '', days = 0) -> int:
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
            The ID of the newly inserted survey.
        """
        return self.db.insert_survey_record(cruise_id, survey_name, vessel, start_port, end_port, start_date, end_date,
                                            description, days)

    def update_survey(self, survey_id: int, survey_name: str, vessel: str, start_port: str, end_port: str,
                      start_date: str, end_date: str, description: Optional[str]) -> None:
        """
        Update an existing survey record.

        Parameters
        ----------
        survey_id : int
            The ID of the survey to update.
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
        """
        self.db.cursor.execute('''
            UPDATE surveys
            SET survey_name=%s, vessel=%s, start_port=%s, end_port=%s, start_date=%s, end_date=%s, description=%s
            WHERE id=%s
        ''', (survey_name, vessel, start_port, end_port, start_date, end_date, description, survey_id))
        self.db.conn.commit()
