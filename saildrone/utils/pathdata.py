import os
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


def load_local_files(directory: str, map_to_directory: str, extension: str = '*.raw') -> list[Path]:
    """
    Load and map local raw files from the given directory.

    Parameters
    ----------
    directory : str
        The directory containing the original raw files.

    map_to_directory : str
        The directory to map the raw files to.

    extension : str, optional
        The file extension to search for, by default '*.raw'.

    Returns
    -------
    list[Path]
        A list of mapped file paths, where the base path is replaced by the value of the RAW_DATA_MOUNT environment variable.

    Args:
        extension:
    """
    # Get the mounted and local directories as Path objects
    mounted_folder = Path(directory)
    mount_base_path = Path(map_to_directory)

    # Get the list of all .raw files sorted by name
    raw_files = sorted(mounted_folder.glob(extension))
    mapped_files = [
        mount_base_path / file.name
        for file in raw_files
    ]

    return mapped_files
