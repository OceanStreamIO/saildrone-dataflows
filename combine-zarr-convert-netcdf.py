import os
import logging
import xarray as xr
import numpy as np
from pathlib import Path
from collections import defaultdict

from saildrone.store import PostgresDB, FileSegmentService
from saildrone.process.plot import plot_sv_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define directories
ZARR_ROOT = Path("./rekaexport")
OUTPUT_DIR = Path("./netcdf_output")
PLOT_OUTPUT_DIR = Path("./echogram_plots")
OUTPUT_DIR.mkdir(exist_ok=True)
PLOT_OUTPUT_DIR.mkdir(exist_ok=True)

# Function to find valid file IDs
def find_valid_ids(base_directory):
    """Finds valid dataset IDs in subdirectories and retrieves metadata."""
    grouped_files = defaultdict(lambda: {"normal": [], "denoised": [], "metadata": []})

    with PostgresDB() as db:
        file_service = FileSegmentService(db)

        for survey_dir in base_directory.iterdir():
            if survey_dir.is_dir():
                for file_id_dir in survey_dir.iterdir():
                    if file_id_dir.is_dir():
                        file_id = file_id_dir.name
                        normal_zarr = file_id_dir / f"{file_id}.zarr"
                        denoised_zarr = file_id_dir / f"{file_id}_denoised.zarr"

                        if normal_zarr.exists() and denoised_zarr.exists():
                            metadata = file_service.get_file_metadata(file_id)

                            file_freqs = metadata.get("file_freqs", "unknown")
                            category = "short_pulse" if file_freqs == "38000.0,200000.0" else "long_pulse" if file_freqs == "38000.0" else "exported_ds"

                            print(f"Found valid file ID: {file_id} | file_freqs: {file_freqs} | categ ({category})")
                            grouped_files[category]["normal"].append(normal_zarr)
                            grouped_files[category]["denoised"].append(denoised_zarr)
                            grouped_files[category]["metadata"].append(metadata)

    return grouped_files

# Function to combine datasets per frequency
def combine_zarr_files(zarr_files):
    """Loads and combines multiple Zarr files while ensuring consistent dimension alignment."""
    datasets = [xr.open_zarr(f) for f in zarr_files]
    return xr.concat(datasets, dim="ping_time")

# Function to save as NetCDF
def save_to_netcdf(dataset, output_path):
    """Saves dataset as NetCDF."""
    logging.info(f"Saving NetCDF file to {output_path}")
    dataset.to_netcdf(output_path, format="NETCDF4")

# Function to plot NetCDF file
def plot_netcdf(netcdf_path):
    """Loads a NetCDF file and plots the echogram."""
    ds = xr.open_dataset(netcdf_path)
    file_base_name = netcdf_path.stem

    print(f"Plotting echograms for {file_base_name}\n{ds}")

    plot_sv_data(ds, file_base_name=file_base_name, output_path=PLOT_OUTPUT_DIR, depth_var="depth")
    logging.info(f"Plots saved in: {PLOT_OUTPUT_DIR}")

# Main processing loop
if __name__ == "__main__":
    grouped_files = find_valid_ids(ZARR_ROOT)

    for category, data in grouped_files.items():

        print(f"\n\n---- Processing category: {category}----\n\n")
        if data["normal"]:
            normal_ds = combine_zarr_files(data["normal"])
            output_file = OUTPUT_DIR / f"{category}.nc"
            save_to_netcdf(normal_ds, output_file)
            plot_netcdf(output_file)

        if data["denoised"]:
            denoised_ds = combine_zarr_files(data["denoised"])
            output_file = OUTPUT_DIR / f"{category}_denoised.nc"
            save_to_netcdf(denoised_ds, output_file)
            plot_netcdf(output_file)

    logging.info("Processing complete.")
