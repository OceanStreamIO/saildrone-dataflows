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
