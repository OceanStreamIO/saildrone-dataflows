import os
import logging
import xarray as xr
from pathlib import Path
from saildrone.store import PostgresDB, FileSegmentService
from saildrone.process.plot import plot_sv_data


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define directories
ZARR_ROOT = Path("./rekaexport")
OUTPUT_DIR = Path("./netcdf_output")
OUTPUT_DIR.mkdir(exist_ok=True)


# Define where plots should be saved
PLOT_OUTPUT_DIR = "./echogram_plots"
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

def plot_netcdf(netcdf_path):
    """Loads a NetCDF file and plots the echogram."""
    ds = xr.open_dataset(netcdf_path)
    file_base_name = os.path.basename(netcdf_path).replace(".nc", "")

    # Generate echograms for each channel
    plot_sv_data(ds, file_base_name=file_base_name, output_path=PLOT_OUTPUT_DIR, depth_var="depth")

    print(f"Plots saved in: {PLOT_OUTPUT_DIR}")

# Function to find valid file IDs
def find_valid_ids(base_directory):
    """Finds valid dataset IDs in subdirectories."""
    valid_ids = []

    for survey_dir in base_directory.iterdir():
        if survey_dir.is_dir():
            for file_id_dir in survey_dir.iterdir():
                if file_id_dir.is_dir():
                    normal_zarr = file_id_dir / f"{file_id_dir.name}.zarr"
                    denoised_zarr = file_id_dir / f"{file_id_dir.name}_denoised.zarr"
                    if normal_zarr.exists() and denoised_zarr.exists():
                        valid_ids.append((normal_zarr, denoised_zarr, file_id_dir.name))

    return valid_ids

# Function to fetch metadata from the database
def fetch_metadata(file_id):
    """Retrieve metadata from the database for a given file ID."""
    with PostgresDB() as db:
        file_service = FileSegmentService(db)
        metadata = file_service.get_file_metadata(file_id)
    return metadata or {}

# Function to open and merge Zarr files with metadata
def open_and_merge_zarr(normal_path, denoised_path, metadata):
    """Loads and concatenates normal and denoised datasets, preserving metadata."""
    logging.info(f"Loading {normal_path} and {denoised_path}")

    ds_normal = xr.open_zarr(normal_path)
    ds_denoised = xr.open_zarr(denoised_path)

    # Merge datasets while preserving metadata
    ds_merged = xr.merge([ds_normal, ds_denoised], compat="override")

    # Convert and inject metadata into global attributes
    for key, value in metadata.items():
        if isinstance(value, bool):
            value = int(value)  # Convert boolean to integer (0 or 1)
        elif isinstance(value, bytes):
            value = value.decode()  # Convert bytes to string
        ds_merged.attrs[key] = value  # Store metadata safely

    return ds_merged

# Function to save as NetCDF
def save_to_netcdf(dataset, output_path):
    """Saves dataset as NetCDF."""
    logging.info(f"Saving NetCDF file to {output_path}")
    dataset.to_netcdf(output_path, format="NETCDF4")

# Main processing loop
if __name__ == "__main__":
    zarr_pairs = find_valid_ids(ZARR_ROOT)

    logging.info(f"Found {len(zarr_pairs)} valid dataset pairs")
    for normal_zarr, denoised_zarr, file_id in zarr_pairs:
        output_file = OUTPUT_DIR / f"{file_id}.nc"

        metadata = fetch_metadata(file_id)
        ds_merged = open_and_merge_zarr(normal_zarr, denoised_zarr, metadata)
        save_to_netcdf(ds_merged, output_file)

        # Plot the newly created NetCDF file
        plot_netcdf(output_file)

    logging.info("Processing complete.")
