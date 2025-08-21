import os
from typing import Union, Mapping

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import holoviews as hv
import hvplot.xarray
import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import HoverTool
import dask.array as dsa
import matplotlib.dates as mdates
import xarray as xr
import shutil
import traceback
import re

import plotly.graph_objects as go
import matplotlib.cm as _mpl_cm
import matplotlib.colors as _mpl_colors

from bokeh.resources import INLINE
from pathlib import Path
from saildrone.store import upload_folder_to_blob_storage

hv.extension("bokeh")
pn.extension(design="material", notifications=False)


def plot_sv_data(ds_Sv: xr.Dataset, file_base_name: str, output_path: str = None, cmap: str = 'ocean_r',
                 channel: int = None, colorbar_orientation: str = 'vertical', plot_var='Sv', title_template='{channel_label}') -> list:
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

    def plot_fn(ch):
        return plot_sv_channel(ds_Sv,
                               channel=ch,
                               file_base_name=file_base_name,
                               echogram_path=output_path,
                               colorbar_orientation=colorbar_orientation,
                               cmap=cmap,
                               plot_var=plot_var,
                               title_template=title_template)

    if channel is not None:
        # If a specific channel is provided, plot only that channel
        try:
            echogram_files.append(plot_fn(channel))
        except Exception as e:
            print(f"Error plotting echogram for channel {channel}: {e}")
            traceback.print_exc()
    else:
        for ch in range(ds_Sv.dims['channel']):
            try:
                echogram_files.append(plot_fn(ch))
            except Exception as e:
                print(f"Error plotting echogram for {file_base_name}: {e}")
                traceback.print_exc()

    return echogram_files


def plot_sv_channel(
    ds_Sv: xr.Dataset,
    channel: int,
    file_base_name,
    echogram_path: str = ".",
    cmap: str = "ocean_r",
    colorbar_orientation: str = "vertical",
    plot_var: str = "Sv",
    title_template: str = "{channel_label}",
    vmin: float = -80,
    vmax: float = -50,
    hour_grid: bool = False,
    inches_per_hour: float = 1.0,
    min_width: float = 14.0,
    max_width: float = 34.0,
    dpi: int = 180,
    height_in: float = 12.0,  # base figure height (inches)
    min_aspect_short: float = 1.8,  # width/height at ~1h
    target_aspect_24h: float = 3.2,
):
    if isinstance(ds_Sv, xr.Dataset):
        channel_idx = channel
    else:
        if channel not in ds_Sv:
            raise KeyError(f"{channel} not found among frequencies {list(ds_Sv)}")
        ds_Sv = ds_Sv[channel]
        channel_idx = None

    da_Sv, meta = prepare_channel_da(ds_Sv, channel_idx, var_name=plot_var)

    t = pd.to_datetime(da_Sv[meta["xdim"]].values)
    hours = max(1.0, (t[-1] - t[0]).total_seconds() / 3600.0)

    time_width = hours * inches_per_hour
    if hours <= 1.0:
        aspect_target = min_aspect_short
    elif hours >= 24.0:
        aspect_target = target_aspect_24h
    else:
        w0, w1 = min_aspect_short, target_aspect_24h
        aspect_target = w0 + (w1 - w0) * ((hours - 1.0) / (24.0 - 1.0))

    width_in = max(time_width, min_width, aspect_target * height_in)
    fig, ax = plt.subplots(figsize=(width_in, height_in))

    da_Sv.T.plot.pcolormesh(
        x=meta["xdim"],
        y=meta["ydim"],
        shading="auto",
        yincrease=False,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        add_colorbar=False,
        ylim=(meta["bot"], meta["top"]),
        ax=ax,
        rasterized=True,
    )

    if hour_grid and meta["xdim"] == "ping_time":
        start = t[0].floor("1H")
        end = t[-1].ceil("1H")
        for dt in pd.date_range(start, end, freq="1H"):
            ax.axvline(dt, color="k", lw=0.6, alpha=0.18, zorder=3)

    # --- cosmetics (mostly yours, with a few tweaks) ---
    ax.set_facecolor("#f9f9f9")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_linewidth(1.0)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=18))
    ax.xaxis.set_major_formatter(meta["x_formatter"])

    # colorbar
    cbar_kw = dict(pad=0.08, shrink=0.8)
    if colorbar_orientation == "horizontal":
        cbar_kw.update(orientation="horizontal", aspect=40)
    else:
        cbar_kw.update(fraction=0.04, pad=0.02)
    cbar = plt.colorbar(ax.collections[0], **cbar_kw)
    cbar.set_label("Volume backscattering strength (Sv re 1 m⁻¹)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # labels & title
    ax.set_xlabel(meta["x_label"], fontsize=16, labelpad=14)
    ax.set_ylabel("Depth [m]" if meta["ydim"] != "range_sample" else "Sample #",
                  fontsize=16, labelpad=14)
    ax.set_title(title_template.format(channel_label=meta["ch_label"]),
                 fontsize=18, fontweight="bold", pad=16)
    ax.tick_params(which="major", length=6, width=1, labelsize=11)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout(pad=2)

    # save the figure (higher DPI = better detail at same pixel size)
    out_path = Path(echogram_path) / f"{file_base_name}_{meta['safe_label']}.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    return out_path


def __plot_individual_channel_simplified(ds_Sv: xr.Dataset, channel: int, file_base_name: str,
                                         echogram_path: str, cmap: str, depth_var='range_sample',
                                         colorbar_orientation='horizontal', plot_var='Sv') -> str:
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
    da_Sv = ds_Sv[plot_var].isel(channel=channel)

    if 'beam' in da_Sv.dims:
        da_Sv = da_Sv.isel(beam=0).drop_vars('beam')

    # 1a) choose x-axis: prefer 'distance', fall back to 'ping_time'
    if "distance" in da_Sv.coords:
        xdim = "distance"
        x_label = "Along-track distance [nmi]"  # tweak as needed
        x_formatter = None  # plain numeric
    else:
        xdim = "ping_time"
        x_label = "Ping time [UTC]"
        x_formatter = mdates.DateFormatter("%H:%M")

    # 1b) choose and clean y-axis
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
    top = float(da_Sv[ydim].isel({ydim: 0}).compute().item())
    bot = float(da_Sv[ydim].isel({ydim: -1}).compute().item())

    # 2) plot with xarray’s .plot
    fig, ax = plt.subplots(figsize=(20, 12))

    da_plot = da_Sv.T
    mesh = da_plot.plot(
        x=xdim,
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
                              export_filename=None, create_interactive_pages=False,
                              channel=None, cmap='ocean_r', plot_var='Sv',
                              title_template='{channel_label}'):
    if save_to_blobstorage:
        echograms_output_path = f'/tmp/osechograms/{cruise_id}/{file_base_name}'
    else:
        echograms_output_path = f'{output_path}/echograms/{file_base_name}'

    if file_name is None:
        file_name = file_base_name

    os.makedirs(echograms_output_path, exist_ok=True)

    echogram_files = plot_sv_data(sv_dataset,
                                  file_base_name=file_name,
                                  output_path=echograms_output_path,
                                  plot_var=plot_var,
                                  cmap=cmap,
                                  channel=channel,
                                  title_template=title_template)

    if create_interactive_pages:
        n_ch = sv_dataset.dims["channel"]
        for ch in range(n_ch):
            html_path = f"{file_base_name}_{ch}.html"
            export_interactive_echogram(sv_dataset, ch, out_html=f"{echograms_output_path}/{html_path}")

    if save_to_blobstorage:
        upload_path = upload_path or f'{cruise_id}/{file_base_name}'
        upload_folder_to_blob_storage(echograms_output_path, container_name, upload_path)
        shutil.rmtree(echograms_output_path, ignore_errors=True)

        if export_filename is not None:
            uploaded_files = [export_filename(e) for e in echogram_files]
        else:
            uploaded_files = [f"{cruise_id}/{file_base_name}/{str(Path(e).name)}" for e in echogram_files]
    else:
        uploaded_files = [str(Path(e).name) for e in echogram_files]

    return uploaded_files


def plot_and_upload_masks(ds, file_base_name=None, upload_path=None, container_name=None,
                          title_template="{channel_label} – {cube_name}"):

    output_path = f'/tmp/osechograms/{file_base_name}'
    os.makedirs(output_path, exist_ok=True)

    paths = plot_masks_vertical(ds, file_base_name=file_base_name, output_path=output_path)

    upload_folder_to_blob_storage(output_path, container_name, upload_path)
    shutil.rmtree(output_path, ignore_errors=True)

    uploaded_files = [f"{file_base_name}/{str(Path(e).name)}" for e in paths]

    return uploaded_files


def ensure_channel_labels(
    ds: xr.Dataset,
    *,
    chan_dim: str = "channel",
    label_coord: str = "channel_label",
    add_freq: bool = False,  # ← new switch
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
    else:  # try to parse from orig name
        fn_hz = []
        for name in orig_names:
            m = re.search(r"(\d+(?:\.\d+)?)", name)
            fn_hz.append(float(m.group(1)) * 1e3 if (m and float(m.group(1)) < 1e3) else
                         float(m.group(1)) if m else np.nan)
    labels = []
    for oname, hz in zip(orig_names, fn_hz):
        if add_freq and not np.isnan(hz):
            labels.append(f"{oname} ({hz / 1e3:.0f} kHz)")
        else:
            labels.append(oname)

    return ds.assign_coords({label_coord: (chan_dim, labels)})


def prepare_channel_da(
    ds: xr.Dataset | xr.DataArray,
    channel: int | None,
    var_name: str,
    depth_fallback: str = "depth",
):
    """
    Extract a single-channel DataArray and return it ready for plotting.

    Parameters
    ----------
    ds             : echopype Dataset **or** DataArray containing *var_name*
    channel        : channel index to plot (ignored if ds has no 'channel')
    var_name       : variable to extract (e.g. 'Sv', 'mask')
    depth_fallback : use this name if neither 'depth' nor 'echo_range'
                     exists as a coordinate

    Returns
    -------
    da_clean : 2-D DataArray (ping_time, range)
    plotting_metadata : dict with labels and axis information
    """
    # 1 pick the DataArray
    if isinstance(ds, xr.Dataset):
        da = ds[var_name]
    else:
        da = ds

    # slice by positional index
    if "channel" in da.dims:
        da = da.isel(channel=channel)

    if "channel_label" in da.coords:
        ch_label = da.coords["channel_label"].values.item()  # 0-D scalar
    elif "channel_label" in getattr(ds, "coords", {}):
        # full dataset has 1-D label vector – use the same index
        ch_label = ds["channel_label"].isel(channel=channel).values.item()
    else:
        ch_label = f"Ch-{channel}"

    safe_label = ch_label.replace(" ", "-")

    # choose vertical axis
    if "depth" in da.coords:
        ydim = "depth"
    elif "echo_range" in da.coords:
        ydim = "echo_range"
    else:
        ydim = depth_fallback

    # tidy NaNs & sort
    valid = np.isfinite(da[ydim])
    da_clean = (
        da.isel({ydim: valid})
        .dropna(dim=ydim, how="all")
        .sortby(ydim)
    )

    # limits
    try:
        top = float(da_clean[ydim].isel({ydim: 0}).item())
        bot = float(da_clean[ydim].isel({ydim: -1}).item())
    except Exception as e:
        print(f"Error determining depth limits for {ch_label}: {e}")
        traceback.print_exc()

        raise ValueError(
            f"Failed to determine depth limits for channel {ch_label}. "
            "Ensure that the data contains valid depth information."
        )

    plotting_metadata = dict(
        ch_label=ch_label,
        safe_label=safe_label,
        xdim="ping_time",
        x_label="Ping time [UTC]",
        x_formatter=mdates.DateFormatter("%H:%M"),
        ydim=ydim,
        top=top,
        bot=bot,
    )

    return da_clean, plotting_metadata


def plot_mask_channel(
    mask_da: xr.DataArray | xr.Dataset,
    channel: int | None,
    file_base_name: str,
    echogram_path: str = ".",
    cmap: str = "Greys",
    title_template: str = "{channel_label} – attenuated-signal mask",
):
    # reuse the same preparation helper
    da_mask, meta = prepare_channel_da(mask_da, channel, var_name="mask")

    fig, ax = plt.subplots(figsize=(20, 12))
    da_mask.T.plot(
        x=meta["xdim"],
        y=meta["ydim"],
        yincrease=False,
        vmin=0,
        vmax=1,
        cmap=cmap,
        add_colorbar=False,
        ylim=(meta["bot"], meta["top"]),
        ax=ax,
    )

    # match style
    ax.set_facecolor("#f9f9f9")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_linewidth(1.0)
    locator = mdates.AutoDateLocator(maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(meta["x_formatter"])

    # colour-bar
    cbar = plt.colorbar(
        ax.collections[0], pad=0.08, orientation="horizontal", aspect=40, shrink=0.8
    )
    cbar.set_label("Boolean mask", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # labels & title
    ax.set_xlabel(meta["x_label"], fontsize=16, labelpad=14)
    ax.set_ylabel("Depth [m]" if meta["ydim"] != "range_sample" else "Sample #",
                  fontsize=16, labelpad=14)
    ax.set_title(title_template.format(channel_label=meta["ch_label"]),
                 fontsize=18, fontweight="bold", pad=16)
    ax.tick_params(which="major", length=6, width=1, labelsize=11)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout(pad=2)
    out_path = Path(echogram_path) / f"{file_base_name}_{meta['safe_label']}_mask.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def plot_all_masks(
    mask_cube: xr.DataArray,
    ds_source: xr.Dataset,
    stage_name: str,
    file_base_name: str,
    output_path: str = "./echograms",
    **plot_kw,
):
    """
    Draw one mask‐echogram per channel for a given denoising *stage*.

    Parameters
    ----------
    mask_cube : DataArray[bool]  (channel, ping_time, depth)
        The Boolean mask returned by `build_full_mask` (or a per-stage slice
        from that function).  True = cell to be removed.
    ds_source : original Dataset
        Provides `channel` coordinate and, if present, `channel_label`.
    stage_name : str
        Human-readable filter name (“Attenuation”, “Impulsive noise”, …) that
        appears in every figure title.
    file_base_name  : str
        Prefix used when writing the PNGs, e.g. “impulsive_noise”.  The channel
        index is appended automatically.
    output_path    : str or Path
        Directory in which PNGs are saved (created if missing).
    plot_kw    : any additional keyword arguments forwarded to
                 `plot_mask_channel` (e.g. `cmap="Greys"`).

    Returns
    -------
    List[Path]
        Paths to the PNG files that were written.
    """

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ch = mask_cube.sizes["channel"]
    out_paths = []

    # default title template can be overridden via **plot_kw
    default_tpl = "{channel_label} – " + stage_name
    title_tpl = plot_kw.pop("title_template", default_tpl)

    for idx in range(n_ch):
        # ---- isolate this channel’s mask (2-D) ----------------------------
        mask_da = (
            mask_cube.isel(channel=idx)
            .expand_dims(channel=[ds_source.channel.values[idx]])
            .astype(int)  # 0/1 for plotting
            .rename("mask")
        )

        if "channel_label" in ds_source.coords:
            # extract the scalar label
            lbl = ds_source["channel_label"].isel(channel=idx).values.item()
            # assign as a length-1 coord along the 'channel' axis
            mask_da = mask_da.assign_coords(channel_label=("channel", [lbl]))

        # ---- plot ---------------------------------------------------------
        chan_idx = 0 if mask_da.sizes["channel"] == 1 else idx

        png_path = plot_mask_channel(
            mask_da=mask_da,
            channel=chan_idx,
            file_base_name=f"{file_base_name}_ch{idx}",
            echogram_path=str(out_dir),
            title_template=title_tpl,
            **plot_kw,
        )
        out_paths.append(Path(png_path))

    return out_paths


def plot_masks_vertical(
    ds_source: xr.Dataset,
    file_base_name: str,
    output_path: str = "./echograms",
    cmap: str = "Greys",
    title_template: str = "{channel_label} – {cube_name}",
) -> Mapping[str, Path]:
    """
    Plot every mask cube in `mask_cubes` to its own PNG.

    Returns
    -------
    dict
        stage_name → pathlib.Path of the PNG written.
    """
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_cubes: dict[str, xr.DataArray] = {}

    for var in ds_source.data_vars:
        if var.startswith("mask_"):
            cube_name = var[len("mask_"):]  # e.g. "impulsive"
            mask_cubes[cube_name] = ds_source[var]

    paths = {}
    for name, cube in mask_cubes.items():
        png = _plot_single_mask_cube(
            cube,
            ds_source,
            file_base_name,
            out_dir,
            cmap=cmap,
            title_tmpl=title_template,
            cube_name=name,
        )
        paths[name] = png

    return paths


def _plot_single_mask_cube(
    cube: xr.DataArray,
    ds_source: xr.Dataset,
    fname: str,
    out_dir: Path,
    cmap: str = "Greys",
    title_tmpl: str = "{channel_label} – {cube_name}",
    cube_name: str = "mask",
):
    """Render one mask cube (all channels) → PNG path."""
    n_ch = cube.sizes["channel"]
    fig_h = 6 * n_ch
    fig, axes = plt.subplots(n_ch, 1, figsize=(20, fig_h), sharex=True)
    axes = np.atleast_1d(axes)

    for ch, ax in enumerate(axes):
        da = (
            cube.isel(channel=ch)
            .expand_dims(channel=[ds_source.channel.values[ch]])
            .astype(int)
            .rename("mask")
        )
        lbl = (
            ds_source["channel_label"].isel(channel=ch).values.item()
            if "channel_label" in ds_source.coords
            else f"CH{ch}"
        )
        da_p, meta = prepare_channel_da(da, 0, var_name="mask")
        da_p.T.plot(
            x=meta["xdim"],
            y=meta["ydim"],
            yincrease=False,
            vmin=0,
            vmax=1,
            cmap=cmap,
            add_colorbar=False,
            ylim=(meta["bot"], meta["top"]),
            ax=ax,
        )
        ax.set_facecolor("#f9f9f9")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_linewidth(1.0)
        ax.set_ylabel(
            "Depth [m]" if meta["ydim"] != "range_sample" else "Sample #",
            fontsize=14,
        )
        ax.set_title(
            title_tmpl.format(channel_label=lbl, cube_name=cube_name),
            fontsize=16,
            pad=8,
        )
        ax.tick_params(labelsize=10)

    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
    axes[-1].xaxis.set_major_formatter(meta["x_formatter"])
    axes[-1].set_xlabel(meta["x_label"], fontsize=16)

    fig.tight_layout(pad=2)
    out_path = out_dir / f"{fname}_{cube_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def export_interactive_echogram(
    ds_Sv,
    channel,
    out_html: str,
    *,
    var: str = "Sv",
    y: str = "depth",
    vmin: float = -80,
    vmax: float = -50,
    time_split: str | None = None,
    title: str | None = None,
    cmap: str = "ocean_r",
    page_bg_color: str = "#ffffff",
    plot_bg_color: str | None = None,
    tools: str = "xwheel_zoom,box_zoom,reset,hover",
    lock_zoom_out: bool = True,
    constrain_pan: bool = True,
    px_per_second: float = 0.40,
    px_per_meter: float = 1.5,
    min_width: int = 900, max_width: int = 2600,
    min_height: int = 300, max_height: int = 900,
    auto_clip_shallow: bool = True,
    auto_clip_margin: float = 0.0,
    bokeh_resources: str = "cdn",
):
    """
    Export an interactive echogram HTML:
      - crisp rendering via QuadMesh + rasterize + upsample (works with irregular ping_time)
      - auto-clip shallow channels (based on where data actually exists)
      - minimal tooltips: 'Sv' only
      - prevents zooming/panning beyond initial view if requested
    """
    pn.extension(design="material", raw_css=[f"body {{ background-color: {page_bg_color}; }}"])

    xdim, ydim = "ping_time", y

    def _canvas_size(chunk) -> tuple[int, int]:
        x = chunk["ping_time"].values
        is_dt = np.issubdtype(x.dtype, np.datetime64)
        if is_dt:
            t0 = pd.to_datetime(x.min());
            t1 = pd.to_datetime(x.max())
            dt_s = max((t1 - t0).total_seconds(), 1.0)
            w = int(np.clip(dt_s * px_per_second, min_width, max_width))
        else:
            n_x = int(chunk.sizes.get("ping_time", chunk["ping_time"].size))
            w = int(np.clip(n_x, min_width, max_width))

        yy = chunk[y].values
        y0, y1 = float(np.nanmin(yy)), float(np.nanmax(yy))
        span_m = max(abs(y1 - y0), 1.0)
        h = int(np.clip(span_m * px_per_meter, min_height, max_height))
        return w, h

    def _sanitize_attrs(da):
        da = da.copy()
        da.name = "Sv"
        ln = str(da.attrs.get("long_name", "") or "")
        # Pull the "(Sv ...)" part if present, else keep "Sv"
        m = re.search(r"\((\s*Sv[^)]*)\)", ln, flags=re.IGNORECASE)
        sv_label = m.group(1).strip() if m else "Sv"
        # Use ASCII "m-1" etc. to avoid font glyph issues that can render as "???"
        sv_label = (sv_label
                    .replace("m^-1", "m-1")
                    .replace("m⁻¹", "m-1"))
        # Set the cleaned label as long_name; keep/ensure units
        attrs = dict(getattr(da, "attrs", {}))
        attrs["long_name"] = sv_label
        attrs.setdefault("units", "dB")
        da.attrs = attrs
        return da

    def _sv_label_and_units(da):
        label = str(da.attrs.get("long_name", "Sv") or "Sv")
        units = (da.attrs.get("units", "") or "").strip()
        return label, units

    def _auto_clip_depth(da):
        if ydim != "depth":
            return da
        valid_by_depth = np.isfinite(da).any(dim=xdim)
        if not bool(valid_by_depth.any()):
            return da
        depth_with_data = da[ydim].where(valid_by_depth, drop=True)
        dmin = float(depth_with_data.min().values)
        dmax_data = float(depth_with_data.max().values)
        dmax_full = float(da[ydim].max().values)
        if dmax_data < dmax_full - 1e-6:
            upper = dmax_data + float(auto_clip_margin)
            return da.sel({ydim: slice(dmin, upper)})
        return da

    def _plot(chunk, label: str):
        w, h = _canvas_size(chunk)
        sv_label, sv_units = _sv_label_and_units(chunk)

        tool_list = tools.split(",") if isinstance(tools, str) else list(tools)
        tool_list = list(dict.fromkeys(t.strip() for t in tool_list if t.strip()))

        p = chunk.hvplot.quadmesh(
            x=xdim,
            y=ydim,
            rasterize=True,
            upsample=True,
            aggregator="mean",
            width=w,
            height=h,
            cmap=cmap,
            clim=(vmin, vmax),
            cnorm="linear",
            flip_yaxis=(ydim == "depth"),
            clabel=f"{sv_label}" + (f" ({sv_units})" if sv_units else ""),
            colorbar=True,
            title=label,
            tools=[]
        )

        # Ranges for bounds + zoom-out lock
        xvals = chunk[xdim].values
        yvals = chunk[ydim].values
        is_dt = np.issubdtype(xvals.dtype, np.datetime64)
        x0, x1 = (pd.to_datetime(xvals.min()), pd.to_datetime(xvals.max())) if is_dt else (float(np.nanmin(xvals)), float(np.nanmax(xvals)))
        y0, y1 = float(np.nanmin(yvals)), float(np.nanmax(yvals))
        x_span = (x1 - x0).total_seconds() * 1000.0 if is_dt else (x1 - x0)
        y_span = (y1 - y0)

        def _lock_ranges(hv_plot, _):
            xr = hv_plot.handles.get("x_range")
            yr = hv_plot.handles.get("y_range")
            if xr is not None:
                if constrain_pan:
                    xr.bounds = (x0, x1)
                if lock_zoom_out:
                    xr.max_interval = x_span
            if yr is not None:
                lo, hi = (min(y0, y1), max(y0, y1))
                if constrain_pan:
                    yr.bounds = (lo, hi)
                if lock_zoom_out:
                    yr.max_interval = abs(y_span)

        def _style_fonts_and_grid(plot, element):
            fig = plot.state
            fig.title.text_font_size = "13pt"
            fig.xaxis.axis_label_text_font_size = "11pt"
            fig.yaxis.axis_label_text_font_size = "11pt"
            fig.xaxis.major_label_text_font_size = "10pt"
            fig.yaxis.major_label_text_font_size = "10pt"

            # grid styling
            fig.xgrid.grid_line_color = "#e0e0e0"
            fig.xgrid.grid_line_alpha = 0.5
            fig.ygrid.grid_line_color = "#e0e0e0"
            fig.ygrid.grid_line_alpha = 0.5

        p = p.opts(
            active_tools=['xwheel_zoom'],
            tools=[],
            hooks=[_lock_ranges, _style_fonts_and_grid],
            bgcolor=plot_bg_color if plot_bg_color is not None else None,
            padding=0.02,
            fontsize=11,
            title_format=label,
        )

        return p

    da = ds_Sv[var].isel(channel=channel)
    if da.dims != (xdim, ydim):
        da = da.transpose(xdim, ydim)

    if ydim == "depth" and np.any(np.diff(da[ydim].values) < 0):
        da = da.sortby(ydim)

    try:
        da = da.compute()
    except Exception:
        pass

    da = _sanitize_attrs(da)

    # auto-clip shallow channels if requested
    if auto_clip_shallow:
        da = _auto_clip_depth(da)

    if time_split:
        t0 = pd.to_datetime(da[xdim].values[0]).floor(time_split)
        t1 = pd.to_datetime(da[xdim].values[-1]).ceil(time_split)
        edges = pd.date_range(t0, t1, freq=time_split)

        tabs = []
        for start, end in zip(edges[:-1], edges[1:]):
            sl = da.sel({xdim: slice(start, end)})
            if sl.size == 0:
                continue
            try:
                sl = sl.compute()
            except Exception:
                pass
            sl = _sanitize_attrs(sl)
            if auto_clip_shallow:
                sl = _auto_clip_depth(sl)
            tabs.append((f"{start:%Y-%m-%d %H:%M} – {end:%H:%M}", _plot(sl, "")))

        panel_obj = pn.Tabs(*tabs) if len(tabs) > 1 else (pn.panel(tabs[0][1]) if tabs else pn.pane.Markdown("No data"))
    else:
        panel_obj = pn.panel(_plot(da, title or ""))

    out = Path(out_html)
    panel_obj.save(str(out), embed=True, resources=bokeh_resources, title=(title or "Echogram"))
    return out
