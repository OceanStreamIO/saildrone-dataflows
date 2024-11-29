from pathlib import Path

from saildrone.store import list_zarr_files


def test_list_zarr_files_integration():
    """Integration test for the list_zarr_files function with actual Azure Blob Storage."""
    path = 'converted/SD_TPOS2023_v03'

    # Run the function under test
    result = list_zarr_files(path)

    assert result, "The result should not be empty."

    first_item = result[0]
    assert isinstance(first_item, Path), f"Expected 'name' to be an instance of Path, got {type(first_item['name'])}"
