from pathlib import Path
from typing import List


def map_file_paths(file_paths: List[Path], mounted_folder: Path, local_folder: Path) -> List[Path]:
    """
    Maps file paths from the mounted folder to the local folder.

    Args:
        file_paths: List of file paths from the mounted folder.
        mounted_folder: The mounted folder path (e.g., /Volumes/saildrone).
        local_folder: The local folder path on Ubuntu (e.g., /media/ubuntu/saildrone).

    Returns:
        List of file paths updated to match the local folder location.
    """
    mapped_paths = [
        Path(str(file_path).replace(str(mounted_folder), str(local_folder), 1))
        for file_path in file_paths
    ]

    return mapped_paths


def load_local_files(directory: str, map_to_directory: str, glob_expression: str = '*.raw') -> list[Path]:
    """
    Load and map local raw files from the given directory.

    Parameters
    ----------
    directory : str
        The directory containing the original raw files.

    map_to_directory : str
        The directory to map the raw files to.

    glob_expression : str
        The glob expression to match the raw files. Default is '*.raw'.

    Returns
    -------
    list[Path]
        A list of mapped file paths, where the base path is replaced by the value of the RAW_DATA_MOUNT environment variable.
    """
    # Get the mounted and local directories as Path objects
    mounted_folder = Path(directory)
    mount_base_path = Path(map_to_directory)

    # Get the list of all .raw files sorted by name
    raw_files = sorted(mounted_folder.glob(glob_expression))

    mapped_files = [
        mount_base_path / file.name
        for file in raw_files
    ]

    return mapped_files


def get_metadata_for_files(zarr_paths: list[Path], files: list[dict]) -> list[tuple[Path, dict]]:
    """
    Given:
      - zarr_paths: a list of Path objects pointing to <something>/*.zarr
      - files:      a list of dicts from FileSegment_Service.get_files_by_survey_id(), each with a 'file_name' key

    Returns:
      A list of (zarr_path, file_record) tuples, matching on the base file_name.
      Raises RuntimeError if any zarr_path has no matching DB row.
    """

    # 1) Build a lookup from filename (no extension) → record
    #    e.g. { "cruiseid_file001": {"file_name":"mysurvey_001", "location_data":..., ...}, ... }
    metadata_by_name: dict[str, dict] = {
        record["file_name"]: record
        for record in files
    }

    paired: list[tuple[Path, dict]] = []
    for z in zarr_paths:
        # Extract just the final component “something.zarr” → “something”
        name = z.name
        if not name.endswith(".zarr"):
            raise RuntimeError(f"Unexpected zarr path that does not end with .zarr: {z!r}")

        base_name = name[: -len(".zarr")]

        if base_name not in metadata_by_name:
            # No matching DB record for this zarr folder
            raise RuntimeError(f"No database record found for zarr store {z!r} (looking for file_name={base_name!r})")

        record = metadata_by_name[base_name]
        paired.append((z, record))

    return paired
