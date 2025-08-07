from typing import Optional, List
import json
import uuid
from psycopg2.extras import Json
from pydantic import BaseModel
from saildrone.store import PostgresDB


class ExportService:
    def __init__(self, db: PostgresDB) -> None:
        self.db = db
        self.table_exports = "exports"
        self.table_exports_files = "exports_files"
        self.table_exports_agg_files = "exports_agg_files"

    def _generate_key(self, container_name: str) -> str:
        return f"{container_name}-{uuid.uuid4().hex[:6]}"

    def create_export(
        self,
        container_name: str,
        base_url: str,
        start_date,
        end_date,
        num_files: int,
        cruise_id: Optional[str] = None,
        denoise_params: Optional[dict] = None,
        agg_params: Optional[dict] = None,
        export_key: Optional[str] = None,
    ) -> tuple[int, str]:
        if export_key is None:
            export_key = self._generate_key(container_name)

        denoise_params_json = _to_jsonb(denoise_params)
        agg_params_json = _to_jsonb(agg_params)

        print('denoise_params_json', denoise_params_json)

        self.db.cursor.execute(
            f"""
            INSERT INTO {self.table_exports}
                (container_name, export_key, base_url,
                 start_date, end_date, num_files, cruise_id,
                 denoise_params, agg_params)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                container_name,
                export_key,
                base_url,
                start_date,
                end_date,
                num_files,
                cruise_id,
                denoise_params_json,
                agg_params_json
            ),
        )
        export_id = self.db.cursor.fetchone()[0]
        self.db.conn.commit()
        return export_id, export_key

    def get_export_details(self, export_key: str) -> Optional[dict]:
        self.db.cursor.execute(
            f"""
            SELECT id, container_name, base_url, start_date, end_date, num_files,
                   denoise_params, agg_params, created_at
            FROM {self.table_exports}
            WHERE export_key = %s
            """,
            (export_key,),
        )
        row = self.db.cursor.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "container_name": row[1],
            "base_url": row[2],
            "start_date": row[3],
            "end_date": row[4],
            "num_files": row[5],
            "denoise_params": row[6],
            "agg_params": row[7],
            "created_at": row[8],
        }

    def add_file(
        self,
        export_id: int,
        file_id: int,
        echogram_files: Optional[List[str]] = None,
        sv_zarr_path: Optional[str] = None,
        denoised_zarr_path: Optional[str] = None,
        netcdf_path: Optional[str] = None,
        netcdf_size: Optional[int] = None,
    ) -> None:
        self.db.cursor.execute(
            f"""
            INSERT INTO {self.table_exports_files}
                (export_id, file_id, echogram_files,
                 sv_zarr_path, denoised_zarr_path,
                 netcdf_path, netcdf_size)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                export_id,
                file_id,
                echogram_files,
                sv_zarr_path,
                denoised_zarr_path,
                netcdf_path,
                netcdf_size,
            ),
        )
        self.db.conn.commit()

    def add_agg_file(
        self,
        export_id: int,
        file_name: str,
        file_type: str,
        file_start_time=None,
        file_end_time=None,
        agg_interval: Optional[str] = None,
        echogram_files: Optional[List[str]] = None,
    ) -> int:
        """
        Insert one record into `exports_agg_files` and return its id.
        """
        self.db.cursor.execute(
            f"""
            INSERT INTO {self.table_exports_agg_files}
                (file_name, export_id, type,
                 file_start_time, file_end_time,
                 agg_interval, echogram_files)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                file_name,
                export_id,
                file_type,
                file_start_time,
                file_end_time,
                agg_interval,
                echogram_files,
            ),
        )
        agg_file_id = self.db.cursor.fetchone()[0]
        self.db.conn.commit()
        return agg_file_id


def _to_jsonb(payload):
    """
    Convert plain dict / list / pydantic model (possibly nested) → Json wrapper
    so psycopg2 stores it in a JSONB column.
    """
    if payload is None:
        return None

    def _default(o):
        if isinstance(o, BaseModel):
            return o.model_dump()

        raise TypeError

    return Json(payload, dumps=lambda obj: json.dumps(obj, default=_default))
