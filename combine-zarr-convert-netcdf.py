import os
import logging
import xarray as xr
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define directories
ZARR_ROOT = Path(__file__).parent / "rekaexport"
OUTPUT_DIR = Path(__file__).parent / "netcdf_output"
OUTPUT_DIR.mkdir(exist_ok=True)


print(f"Looking for Zarr files in: {ZARR_ROOT.resolve()}")


# Function to find valid file IDs
def find_valid_ids(base_directory):
    """Finds valid dataset IDs in subdirectories."""
    valid_ids = []

    for survey_dir in base_directory.iterdir():

        print(f"survey_dir: {survey_dir}")

        if survey_dir.is_dir():
            for file_id_dir in survey_dir.iterdir():
                print(f"file_id_dir: {file_id_dir}")

                if file_id_dir.is_dir():
                    normal_zarr = file_id_dir / f"{file_id_dir.name}.zarr"
                    denoised_zarr = file_id_dir / f"{file_id_dir.name}_denoised.zarr"
                    if normal_zarr.exists() and denoised_zarr.exists():
                        valid_ids.append((normal_zarr, denoised_zarr))

    return valid_ids

# Function to find matching Zarr files
def find_zarr_files(directory):
    """Finds pairs of normal and denoised Zarr files in the directory."""
    zarr_files = {fp.stem: fp for fp in directory.glob("*.zarr")}
    valid_ids = [fid for fid in zarr_files if f"{fid}_denoised" in zarr_files]
    return [(zarr_files[fid], zarr_files[f"{fid}_denoised"]) for fid in valid_ids]

# Function to open and merge Zarr files
def open_and_merge_zarr(normal_path, denoised_path):
    """Loads and concatenates normal and denoised datasets."""
    logging.info(f"Loading {normal_path} and {denoised_path}")
    ds_normal = xr.open_zarr(normal_path)
    ds_denoised = xr.open_zarr(denoised_path)

    # Merge datasets while preserving metadata
    ds_merged = xr.merge([ds_normal, ds_denoised], compat="override")
    return ds_merged

# Function to save as NetCDF
def save_to_netcdf(dataset, output_path):
    """Saves dataset as NetCDF."""
    logging.info(f"Saving NetCDF file to {output_path}")
    dataset.to_netcdf(output_path, format="NETCDF4")

# Main processing loop
if __name__ == "__main__":


    zarr_pairs = find_valid_ids(ZARR_ROOT)
    logging.info(f"Found {len(zarr_pairs)} valid Zarr pairs: {[(n.name, d.name) for n, d in zarr_pairs]}")



    for normal_zarr, denoised_zarr in zarr_pairs:
        file_id = normal_zarr.stem
        output_file = OUTPUT_DIR / f"{file_id}.nc"

        ds_merged = open_and_merge_zarr(normal_zarr, denoised_zarr)
        save_to_netcdf(ds_merged, output_file)

    logging.info("Processing complete.")
