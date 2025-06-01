from typing import Optional, List
import json
import uuid

from saildrone.store import PostgresDB


class ExportService:
    def __init__(self, db: PostgresDB) -> None:
        self.db = db
        self.table_exports = "exports"
        self.table_exports_files = "exports_files"

    def _generate_key(self, container_name: str) -> str:
        return f"{container_name}-{uuid.uuid4().hex[:6]}"

    def insert_export(
        self,
        container_name: str,
        base_url: str,
        start_date,
        end_date,
        num_files: int,
        denoise_params: Optional[dict] = None,
        combined_netcdf_path: Optional[str] = None,
        combined_netcdf_size: Optional[int] = None,
        export_key: Optional[str] = None,
    ) -> tuple[int, str]:
        if export_key is None:
            export_key = self._generate_key(container_name)
        params_json = json.dumps(denoise_params) if denoise_params else None
        self.db.cursor.execute(
            f"""
            INSERT INTO {self.table_exports}
                (container_name, export_key, base_url, start_date, end_date, num_files,
                 denoise_params, combined_netcdf_path, combined_netcdf_size)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
            """,
            (
                container_name,
                export_key,
                base_url,
                start_date,
                end_date,
                num_files,
                params_json,
                combined_netcdf_path,
                combined_netcdf_size,
            ),
        )
        export_id = self.db.cursor.fetchone()[0]
        self.db.conn.commit()
        return export_id, export_key

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
                (export_id, file_id, echogram_files, sv_zarr_path, denoised_zarr_path, netcdf_path, netcdf_size)
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

    def get_export_files(self, export_key: str) -> List[dict]:
        self.db.cursor.execute(
            f"""
            SELECT f.id, f.file_name, f.location, f.file_start_time, f.file_end_time,
                   ef.echogram_files, ef.sv_zarr_path, ef.denoised_zarr_path,
                   ef.netcdf_path, ef.netcdf_size
            FROM {self.table_exports_files} ef
            JOIN {self.table_exports} e ON ef.export_id = e.id
            JOIN files f ON ef.file_id = f.id
            WHERE e.export_key = %s
            ORDER BY f.file_start_time ASC
            """,
            (export_key,),
        )
        rows = self.db.cursor.fetchall()
        return [
            {
                "file_id": r[0],
                "file_name": r[1],
                "location": r[2],
                "file_start_time": r[3],
                "file_end_time": r[4],
                "echogram_files": r[5],
                "sv_zarr_path": r[6],
                "denoised_zarr_path": r[7],
                "netcdf_path": r[8],
                "netcdf_size": r[9],
            }
            for r in rows
        ]

    def get_export_details(self, export_key: str) -> Optional[dict]:
        self.db.cursor.execute(
            f"SELECT id, container_name, base_url, start_date, end_date, num_files, denoise_params, combined_netcdf_path, combined_netcdf_size FROM {self.table_exports} WHERE export_key = %s",
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
            "combined_netcdf_path": row[7],
            "combined_netcdf_size": row[8],
        }
