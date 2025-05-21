import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.dates as mdates
import xarray as xr
import shutil
import traceback
import re
from pathlib import Path
from saildrone.store import upload_folder_to_blob_storage


def plot_sv_data(ds_Sv: xr.Dataset, file_base_name: str, output_path: str = None, cmap: str = 'ocean_r',
                 depth_var: str = 'range_sample', colorbar_orientation: str = 'vertical') -> list:
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
                                                                depth_var, colorbar_orientation)
        echogram_files.append(echogram_file_path)

    return echogram_files


def plot_individual_channel_simplified(ds_Sv: xr.Dataset, channel: int, file_base_name: str,
                                       echogram_path: str, cmap: str, depth_var='range_sample',
                                       colorbar_orientation='horizontal') -> str:
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

    # 0) labels & select the DataArray
    ch_lab = ds_Sv.channel_label.values[channel]
    safe_lab = ch_lab.replace(" ", "-")
    da_Sv = ds_Sv['Sv'].isel(channel=channel)

    if 'beam' in da_Sv.dims:
        da_Sv = da_Sv.isel(beam=0).drop_vars('beam')

    # 1) choose & clean the vertical axis
    if 'depth' in da_Sv.coords:
        ydim = 'depth'
    elif 'echo_range' in da_Sv.coords:
        ydim = 'echo_range'
    else:
        ydim = depth_var

    # drop bins whose coordinate is NaN, then drop all-NaN rows and sort
    valid = np.isfinite(da_Sv[ydim].data)
    da_Sv = da_Sv.isel({ydim: valid}) \
        .dropna(dim=ydim, how='all') \
        .sortby(ydim)

    # compute top/bottom limits
    # top = float(ds_Sv[ydim].min().compute().item())
    # bot = float(ds_Sv[ydim].max().compute().item())

    top = float(da_Sv[ydim].isel({ydim: 0}).compute().item())
    bot = float(da_Sv[ydim].isel({ydim: -1}).compute().item())

    # 2) plot with xarray’s .plot
    # plt.figure(figsize=(20, 12))
    fig, ax = plt.subplots(figsize=(20, 12))

    da_plot = da_Sv.T
    mesh = da_plot.plot(
        x='ping_time',
        y=ydim,
        yincrease=False,
        vmin=-80, vmax=-50,
        cmap=cmap,
        add_colorbar=False,
        ylim=(bot, top)
    )
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['right'].set_linewidth(1.0)

    # 3) subtle background & grid
    ax.set_facecolor('#f9f9f9')

    locator = mdates.AutoDateLocator(maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # 4) overlay seabed if available
    if 'seabed_idx' in ds_Sv.coords:
        idx = ds_Sv.coords["seabed_idx"]
        if "channel" in idx.dims:
            # use .sel on the label to be safer than integer isel
            ch = ds_Sv.channel.values[channel]
            idx = idx.sel(channel=ch)

        idx_int = idx.astype('int64')
        max_idx = da_Sv[ydim].shape[0] - 1
        idx_int = idx_int.clip(min=0, max=max_idx)

        seabed_depth = da_Sv[ydim].isel({ydim: idx_int})

        ax.plot(
            ds_Sv.ping_time,
            seabed_depth,
            'k--',
            lw=1.5,
            label='Seabed'
        )
        ax.fill_between(
            ds_Sv.ping_time,
            seabed_depth,
            y2=bot,
            step='post',
            color='lightgray',
            alpha=0.5
        )
        ax.legend(loc='lower left', frameon=True)

    # 5) add a neat horizontal colorbar
    if colorbar_orientation == 'horizontal':
        cbar = plt.colorbar(mesh, pad=0.08, orientation='horizontal', aspect=40, shrink=0.8)
    else:
        cbar = fig.colorbar(mesh,
                            ax=ax,
                            fraction=0.04,
                            pad=0.02,
                            shrink=0.8)
    fig.subplots_adjust(left=0.06, right=0.82, top=0.93, bottom=0.10)
    ax.tick_params(which='major', length=6, width=1, labelsize=11)
    ax.tick_params(which='minor', length=3, width=0.5)

    cbar.set_label('Volume backscattering strength (Sv re 1 m⁻¹)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # 6) labels, title, layout
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_xlabel('Ping time [UTC]', fontsize=16, labelpad=14)
    ax.set_ylabel('Depth [m]' if ydim != 'range_sample' else 'Sample #', fontsize=16, labelpad=14)
    ax.set_title(ch_lab, fontsize=18, fontweight='bold', pad=16)

    plt.tight_layout(pad=2)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # 7) save & close
    out_name = f"{file_base_name}_{safe_lab}.png"
    out_path = os.path.join(echogram_path, out_name)

    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def plot_and_upload_echograms(sv_dataset, cruise_id=None, file_base_name=None, save_to_blobstorage=False,
                              file_name=None, output_path=None, upload_path=None, container_name=None,
                              cmap='ocean_r', depth_var='depth'):
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
