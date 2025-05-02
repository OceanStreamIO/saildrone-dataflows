import os
import logging
import xarray as xr
import numpy as np
from pathlib import Path
from collections import defaultdict

from saildrone.store import PostgresDB, FileSegmentService
from saildrone.process.plot import plot_sv_data
from saildrone.process.concat import merge_location_data
from echopype.commongrid import compute_NASC, compute_MVBS

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
def combine_zarr_files(zarr_files, metadata):
    """Loads and combines multiple Zarr files while ensuring consistent dimension alignment."""
    datasets = [xr.open_zarr(f) for f in zarr_files]

    # Merge location data for each dataset
    for i, ds in enumerate(datasets):
        location_data = metadata[i].get("location_data", {})
        # datasets[i] = merge_location_data(ds, location_data)

        if "time" in ds.dims:
            datasets[i] = ds.drop_dims("time", errors="ignore")

        if "filenames" in ds.dims:
            datasets[i] = ds.drop_dims("filenames", errors="ignore")

    sorted_datasets = sorted(datasets, key=lambda ds: ds["ping_time"].min().values)

    # sorted_datasets = [
    #     ds.rename({"source_filenames": f"source_filenames_{i}"})
    #     for i, ds in enumerate(sorted_datasets)
    # ]

    # Concatenate along the specified dimension
    concatenated_ds = xr.merge(sorted_datasets)

    if "ping_time" in concatenated_ds.dims and "time" in concatenated_ds.dims:
        concatenated_ds = concatenated_ds.drop_vars("time", errors="ignore")

    return concatenated_ds

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
            normal_ds = combine_zarr_files(data["normal"], data["metadata"])
            output_file = OUTPUT_DIR / f"{category}.nc"
            output_file_mvbs = OUTPUT_DIR / f"{category}_MVBS.nc"
            output_file_nasc = OUTPUT_DIR / f"{category}_NASC.nc"
            # netcdf
            save_to_netcdf(normal_ds, output_file)
            plot_netcdf(output_file)

            # MVBS
            # ds_MVBS = compute_MVBS(
            #     normal_ds,
            #     range_var="depth",
            #     range_bin='1m',  # in meters
            #     ping_time_bin='5s',  # in seconds
            # )
            # save_to_netcdf(ds_MVBS, output_file_mvbs)
            # plot_netcdf(output_file_mvbs)

            # # NASC
            # ds_NASC = compute_NASC(
            #     normal_ds,
            #     range_bin="10m",
            #     dist_bin="0.5nmi"
            # )
            # # Log-transform the NASC values for plotting
            # ds_NASC["NASC_log"] = 10 * np.log10(ds_NASC["NASC"])
            # ds_NASC["NASC_log"].attrs = {
            #     "long_name": "Log of NASC",
            #     "units": "m2 nmi-2"
            # }
            # save_to_netcdf(ds_NASC, output_file_nasc)
            # plot_netcdf(output_file_nasc)



        if data["denoised"]:
            denoised_ds = combine_zarr_files(data["denoised"], data["metadata"])
            output_file_denoised = OUTPUT_DIR / f"{category}_denoised.nc"
            output_file_denoised_mvbs = OUTPUT_DIR / f"{category}_denoised_MVBS.nc"
            output_file_denoised_nasc = OUTPUT_DIR / f"{category}_denoised_NASC.nc"

            # MVBS
            # ds_MVBS = compute_MVBS(
            #     denoised_ds,
            #     range_var="depth",
            #     range_bin='1m',  # in meters
            #     ping_time_bin='5s',  # in seconds
            # )
            # save_to_netcdf(ds_MVBS, output_file_denoised_mvbs)
            # plot_netcdf(output_file_denoised_mvbs)

            # NASC
            # ds_NASC = compute_NASC(
            #     denoised_ds,
            #     range_bin="10m",
            #     dist_bin="0.5nmi"
            # )
            # # Log-transform the NASC values for plotting
            # ds_NASC["NASC_log"] = 10 * np.log10(ds_NASC["NASC"])
            # ds_NASC["NASC_log"].attrs = {
            #     "long_name": "Log of NASC",
            #     "units": "m2 nmi-2"
            # }
            # save_to_netcdf(ds_NASC, output_file_denoised_nasc)
            # plot_netcdf(output_file_denoised_nasc)

            save_to_netcdf(denoised_ds, output_file_denoised)
            plot_netcdf(output_file_denoised)

    logging.info("Processing complete.")
