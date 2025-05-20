import os
import shutil
import traceback
import re
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

from saildrone.store import upload_folder_to_blob_storage


def plot_sv_data(ds_Sv: xr.Dataset, file_base_name: str, output_path: str = None, cmap: str = 'ocean_r',
                 depth_var: str = 'range_sample') -> list:
    """
    Plot Sv data for each channel and save the echogram plots.

    Parameters:
    - ds_Sv: xr.Dataset
        The dataset containing Sv data.
    - file_base_name: str
        The base name for output files.
    - output_path: str
        The path to save the output plots.
    - echogram_path: str
        Path to save individual echogram files.
    - cmap: str
        The colormap for plotting.

    Returns:
    - list: A list of file paths for the saved echograms.
    """
    if not plt.isinteractive():
        plt.switch_backend('Agg')  # Use non-interactive backend for plotting

    echogram_files = []
    for channel in range(ds_Sv.dims['channel']):
        echogram_file_path = plot_individual_channel_simplified(ds_Sv, channel, file_base_name, output_path, cmap,
                                                                depth_var)
        echogram_files.append(echogram_file_path)

    return echogram_files


def plot_individual_channel_simplified(ds_Sv: xr.Dataset, channel: int, file_base_name: str,
                                       echogram_path: str, cmap: str, depth_var='range_sample') -> str:
    """
    Plot and save echogram for a single channel with optional regions and enhancements.

    Parameters:
    - ds_Sv: xr.Dataset
        The dataset containing Sv data.
    - channel: int
        The channel number to plot.
    - output_path: str
        The output path for the plot.
    - file_base_name: str
        The base file name for the plot.
    - echogram_path: str
        Path to save the echogram file.
    - cmap: str
        Colormap for plotting.

    Returns:
    - str: The path to the saved echogram file.
    """

    full_channel_name = str(ds_Sv.channel.values[channel])
    label = str(ds_Sv.channel_label.values[channel])
    channel_name = label.replace(" ", "-")
    filtered_ds = ds_Sv['Sv']

    if 'beam' in filtered_ds.dims:
        filtered_ds = filtered_ds.isel(beam=0).drop('beam')

    if 'channel' in filtered_ds.coords:
        # Ensure frequency is fully computed for swap_dims
        if 'frequency' in ds_Sv.coords:
            freq = ds_Sv['frequency']
            if isinstance(freq.data, da.Array):
                freq = freq.compute()

            filtered_ds = filtered_ds.assign_coords(frequency=("channel", np.asarray(freq)))

        try:
            filtered_ds = filtered_ds.swap_dims({'channel': 'frequency'})
            # if filtered_ds.frequency.size == 1:
            #     filtered_ds = filtered_ds.isel(frequency=0)
        except Exception as e:
            print(f"Error swapping dims while plotting echogram: {e}")

    plt.figure(figsize=(20, 12))

    try:
        da = filtered_ds.isel(frequency=channel)
        coord_vals = da[depth_var].data
        valid_coord = np.isfinite(coord_vals)

        da = da.isel({depth_var: valid_coord})
        da = da.dropna(dim=depth_var, how="all")
        da = da.sortby(depth_var)

        top_depth = float(da[depth_var].isel({depth_var: 0}).compute().item())
        bottom = float(da[depth_var].isel({depth_var: -1}).compute().item())

        da.T.plot(
            x='ping_time',
            y=depth_var,
            yincrease=False,
            vmin=-80,
            vmax=-50,
            cmap=cmap,
            cbar_kwargs={'label': 'Volume backscattering strength (Sv re 1 m⁻¹)'},
            ylim=(bottom, top_depth),
        )
    except Exception as e:
        print(f"Error plotting echogram: {e}")
        traceback.print_exc()

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('Ping time', fontsize=14)
    plt.ylabel('Depth [m]' if depth_var != "range_sample" else "Sample #", fontsize=14)
    plt.title(f'{channel_name}', fontsize=16, fontweight='bold')

    echogram_file_name = f"{file_base_name}_{channel_name}.png"
    echogram_output_path = os.path.join(echogram_path, echogram_file_name)
    plt.savefig(echogram_output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return echogram_output_path


def plot_and_upload_echograms(sv_dataset, cruise_id=None, file_base_name=None, save_to_blobstorage=False,
                              file_name=None, output_path=None, upload_path=None, container_name=None, cmap='ocean_r', depth_var='depth'):
    if save_to_blobstorage:
        echograms_output_path = f'/tmp/osechograms/{cruise_id}/{file_base_name}'
    else:
        echograms_output_path = f'{output_path}/echograms/{file_base_name}'

    if file_name is None:
        file_name = file_base_name

    os.makedirs(echograms_output_path, exist_ok=True)

    echogram_files = plot_sv_data(sv_dataset,
                                  depth_var=depth_var,
                                  file_base_name=file_name,
                                  output_path=echograms_output_path,
                                  cmap=cmap)

    if save_to_blobstorage:
        upload_path = upload_path or f'{cruise_id}/{file_base_name}'
        upload_folder_to_blob_storage(echograms_output_path, container_name, upload_path)
        shutil.rmtree(echograms_output_path, ignore_errors=True)
        uploaded_files = [f"{cruise_id}/{file_base_name}/{str(Path(e).name)}" for e in echogram_files]
    else:
        uploaded_files = [str(Path(e).name) for e in echogram_files]

    return uploaded_files


def plot_noise_mask(mask, file_base_name, echogram_path, depth_var='depth'):
    mask_channel = mask

    finite_mask = np.isfinite(mask_channel.values)
    data_exists_along_depth = np.any(finite_mask, axis=0)
    max_depth_index = np.max(np.where(data_exists_along_depth))
    max_depth = mask_channel[depth_var].values[max_depth_index]

    plt.figure(figsize=(30, 18))
    try:
        mask_channel.plot(
            x='ping_time',
            y=depth_var,
            yincrease=False,
            cmap='binary',
            cbar_kwargs={'label': 'Impulse Noise Mask (True/False)'},
            ylim=(max_depth, 0)
        )
    except Exception as e:
        print(f"Error plotting impulse noise mask: {e}")
        traceback.print_exc()

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('Ping time', fontsize=14)
    plt.ylabel('Depth', fontsize=14)
    plt.title(f"Impulse Noise Mask", fontsize=16, fontweight='bold')

    mask_file_name = f"{file_base_name}_mask.png"
    mask_output_path = os.path.join(echogram_path, mask_file_name)
    plt.savefig(mask_output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return mask_output_path


def ensure_channel_labels(
    ds: xr.Dataset,
    *,
    chan_dim: str = "channel",
    label_coord: str = "channel_label",
    add_freq: bool = False,                  # ← new switch
) -> xr.Dataset:
    """
    Guarantee that *ds* has a textual coordinate ``label_coord`` aligned with
    *chan_dim*.

    Each label starts with the **original channel name** and, if ``add_freq=True`` and a numeric frequency can be
    found, appends ``"(38 kHz)"`` style text.

    """
    # Nothing to do if the label coord already exists
    if label_coord in ds.coords:
        return ds

    # Single-channel dataset → make a trivial label and return
    if chan_dim not in ds.dims:
        return ds.assign_coords({label_coord: ("channel", ["single"])})

    orig_names = [str(v) for v in ds.coords[chan_dim].values]

    if "frequency_nominal" in ds:
        fn_hz = ds["frequency_nominal"].compute().values
    else:                                          # try to parse from orig name
        fn_hz = []
        for name in orig_names:
            m = re.search(r"(\d+(?:\.\d+)?)", name)
            fn_hz.append(float(m.group(1)) * 1e3 if (m and float(m.group(1)) < 1e3) else
                         float(m.group(1)) if m else np.nan)
    labels = []
    for oname, hz in zip(orig_names, fn_hz):
        if add_freq and not np.isnan(hz):
            labels.append(f"{oname} ({hz/1e3:.0f} kHz)")
        else:
            labels.append(oname)

    return ds.assign_coords({label_coord: (chan_dim, labels)})
