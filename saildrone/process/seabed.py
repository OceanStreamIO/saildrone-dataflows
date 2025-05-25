import os
import pathlib
import warnings
import re

from pandas import Index

import dask.array as da
import numpy as np
import xarray as xr

from dask_image.ndfilters import convolve
from scipy.ndimage import binary_erosion as _erode
from scipy.ndimage import binary_dilation as _dilate
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

DEBUG = os.getenv("DEBUG", "1") == "1"
MAX_SV_DEFAULT_PARAMS = {"r0": 10, "r1": 1000, "roff": 0, "thr": (-40, -60)}
DELTA_SV_DEFAULT_PARAMS = {"r0": 10, "r1": 1000, "roff": 0, "thr": 20}
BLACKWELL_DEFAULT_PARAMS = {
    "r0": 10,
    "r1": 1000,
    "tSv": -75,
    "ttheta": 60,
    "tphi": 60,
    "wtheta": 28,
    "wphi": 52,
}
BLACKWELL_MOD_DEFAULT_PARAMS = {
    "r0": 10,
    "r1": 1000,
    "tSv": -75,
    "ttheta": 702,
    "tphi": 282,
    "wtheta": 28,
    "wphi": 52,
    "rlog": None,
    "tpi": None,
    "freq": None,
    "rank": 50,
}
EXPERIMENTAL_DEFAULT_PARAMS = {
    "r0": 10,
    "r1": 1000,
    "roff": 0,
    "thr": (-30, -70),
    "ns": 150,
    "n_dil": 3,
}
ARIZA_DEFAULT_PARAMS = {
    "r0": 10,
    "r1": 1300,
    "roff": 0,
    "thr": -40,
    "ec": 1,
    "ek": (3, 3),
    "dc": 3,
    "dk": (3, 3),
}

ARIZA_SPIKE_DEFAULT_PARAMS = {
    "r0": 10,
    "r1": 10000,
    "roff": 0,
    "thr": (-40, -40),
    "ec": 1,
    "ek": (3, 3),
    "dc": 3,
    "dk": (3, 3),
    "maximum_spike": 200,
}

ARIZA_EXPERIMENTAL_DEFAULT_PARAMS = {
    "r0": 10,
    "r1": 1000,
    "roff": 0,
    "thr": (-40, -70),
    "ec": 1,
    "ek": (3, 3),
    "dc": 3,
    "dk": (3, 3),
}


def mask_true_seabed(ds: xr.Dataset, r0_m=10, r1_m=1000, thr_dB=-45.0, delta_db: float = 3.0) -> xr.Dataset:
    ds_copy = ds.copy(deep=True)
    ds_38 = find_38khz_channel(ds_copy)

    ch_label = ds_38.channel.values.item(0)

    seabed_idx, seabed_depth, ydim = compute_seabed_line(
        ds_38,
        channel=ch_label,
        r0_m=r0_m,
        r1_m=r1_m,
        thr_dB=thr_dB,
        delta_db=delta_db
    )

    ds_copy = ds_copy.assign_coords(
        seabed_idx=("ping_time", seabed_idx.data),
        seabed_depth=("ping_time", seabed_depth.data)
    )

    range_m = ds_38["echo_range"]
    # build a True-above-bottom mask on that grid:
    bottom_of_ping = range_m.isel({ydim: -1})
    seabed_safe = seabed_depth.fillna(bottom_of_ping)
    mask2d = range_m < seabed_safe

    mask3d = broadcast_mask_to_all_channels(
        mask2d.rename("seabed_water_mask"),
        original_ds=ds_copy
    )
    ds_copy["Sv"] = ds_copy["Sv"].where(mask3d)

    return ds_copy

    # mask_38_2d = get_true_seabed_mask(ds_38,
    #                                   channel=ds_38.channel.values.item(0),
    #                                   r0_m=r0_m,
    #                                   r1_m=r1_m,
    #                                   thr_dB=thr_dB
    #                                   )
    #
    # mask_all = broadcast_mask_to_all_channels(mask_38_2d, original_ds=ds)
    #
    # Sv_clean = ds_copy["Sv"].where(mask_all)
    # ds_copy["Sv"] = Sv_clean

    return ds_copy


def _dbg(msg: str):
    if DEBUG:
        print(msg)


def compute_seabed_line(
    ds: xr.Dataset,
    channel: str,
    r0_m: float,
    r1_m: float,
    thr_dB: float = -40.0,
    fixed_percentile: float = 75.0,
    delta_db: float = 3.0,
    erosion_kernel: tuple[int, int] = (1, 3),
    dilation_kernel: tuple[int, int] = (3, 5),
    use_adaptive_thresholding: bool = False,
    adaptive_window_pings: int = 10,
    adaptive_std_multiplier: float = 1.0
) -> xr.DataArray:
    """
    Compute a single‐ping seabed depth line (no masking) for plotting.

    Returns
    -------
    seabed_depth : xr.DataArray
        dims: (ping_time,)
        coords: ping_time, depth
        values: seabed depth [m] per ping
    """
    # 1) Select the channel slice
    if "channel" in ds.dims:
        sub = ds.sel(channel=channel, drop=True) if channel in ds.coords.get("channel", []) else ds.isel(channel=0)
    elif "frequency" in ds.dims:
        sub = ds.sel(frequency=channel, drop=True) if channel in ds.coords.get("frequency", []) else ds.isel(
            frequency=0)
    else:
        sub = ds

    _dbg(f"[1] Selected channel {channel!r}: shape {sub['Sv'].shape}")
    Sv = sub["Sv"]
    er = sub["echo_range"]
    vdim = "range_sample" if "range_sample" in Sv.dims else "depth"
    ping_dim = 'ping_time'

    # 2) Median‐filter full Sv to kill isolated spikes
    Sv_filtered = xr.apply_ufunc(
        median_filter,
        Sv,
        kwargs={"size": (3, 3)},
        dask="parallelized",
        output_dtypes=[Sv.dtype],
    )
    _dbg("[2] Median filter applied")

    # 3) Gate for threshold computation only
    gate = (er >= r0_m) & (er <= r1_m)
    n_gate = int(gate.sum().compute())

    _dbg(f"[3] Depth gate: {r0_m}–{r1_m} m  →  {n_gate:,} voxels inside")

    Sv_gate = Sv_filtered.where(gate).chunk({vdim: -1})

    # 4) Choose thresholding method
    if use_adaptive_thresholding:
        # Rolling median & std over pings
        local_med = Sv_filtered.rolling(ping_time=adaptive_window_pings,
                                        center=True, min_periods=1).median().compute()
        local_std = Sv_filtered.rolling(ping_time=adaptive_window_pings,
                                        center=True, min_periods=1).std().compute()
        # Fill any NaNs
        local_med = local_med.fillna(Sv_filtered.median())
        local_std = local_std.fillna(Sv_filtered.std())
        thr2 = local_med - adaptive_std_multiplier * local_std

    else:
        # Fixed‐percentile dynamic threshold
        perc_thr = xr.apply_ufunc(
            np.nanpercentile,
            Sv_gate,
            input_core_dims=[[vdim]],
            kwargs={"q": fixed_percentile},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[Sv.dtype],
        )
        thr2 = (perc_thr - delta_db).broadcast_like(Sv_filtered)

    # 5) Enforce static floor thr_dB
    thr2 = thr2.clip(min=thr_dB)
    _dbg(f"[5] Static floor enforced at {thr_dB} dB")
    # thr2 = xr.where(thr2 > thr_dB, thr2, thr_dB)

    # 6) Build candidate mask on filtered Sv
    cand = (Sv_filtered > thr2) & gate
    pings_with_cand = int(cand.any(dim=vdim).sum().compute())
    pings_total = Sv.sizes[ping_dim]
    _dbg(f"[6] Raw candidates detected in {pings_with_cand} / {pings_total} pings")

    # 7) Morphology (erosion then dilation)
    def _morph(xda, fn, kernel):
        struct = np.ones(kernel, bool)
        pad = {vdim: max(1, kernel[0] // 2), Sv.dims[0]: max(1, kernel[1] // 2)}
        return xr.apply_ufunc(
            lambda blk, **_: fn(blk, structure=struct),
            xda,
            dask="parallelized", vectorize=False,
            output_dtypes=[bool],
            dask_gufunc_kwargs=dict(depth=pad, boundary="nearest"),
        )

    cand_eroded = _morph(cand, _erode, erosion_kernel)
    cand_dilated = _morph(cand_eroded, _dilate, dilation_kernel)
    empty_ping = ~cand_dilated.any(dim=vdim)
    cand = xr.where(empty_ping, cand, cand_dilated)
    cand = cand & gate
    n_empty = int(empty_ping.sum().compute())
    _dbg(f"[7] Morphology: {n_empty} pings wiped out and reverted to pre-morph mask")

    # 8) Require continuity across pings
    continuity_window = 5
    rolling_pres = cand.rolling(ping_time=continuity_window,
                                center=True, min_periods=1).sum()
    cand = rolling_pres >= 2
    pings_after_cont = int(cand.any(dim=vdim).sum().compute())
    _dbg(f"[8] Continuity ≥2 of 5 pings: {pings_after_cont} pings keep candidates")

    # 9) Pick the deepest True per ping
    vdim_indices = xr.DataArray(
        np.arange(cand.sizes[vdim]), dims=vdim, coords={vdim: cand[vdim]}
    )
    idx_primary = (cand * vdim_indices).max(dim=vdim)
    _dbg("[9] Deepest indices extracted")

    # 10) Safe fallback (if no cand True, pick first Sv > thr2 or bottom)
    fallback_mask = ((Sv_filtered > thr2) & gate)  # ❶
    idx_fb = (fallback_mask * vdim_indices).max(dim=vdim).fillna(cand.sizes[vdim] - 1)
    has_bot = cand.any(dim=vdim)
    idx = xr.where(has_bot, idx_primary, idx_fb)

    bad_pings = int((~has_bot).sum().compute())
    if bad_pings:
        _dbg(f"[10] Fallback used on {bad_pings} pings (no candidate survived)")

    # 11) De-spike with Savitzky–Golay
    ping_vals = idx.ping_time.values
    idx_arr = idx.compute().values.astype(int)
    N = len(idx_arr)
    win = max(3, min(31, N if N % 2 else N - 1))
    if N >= 3:
        idx_sm = savgol_filter(idx_arr, win, polyorder=1, mode="interp")
        jumps = np.abs(np.diff(idx_sm, prepend=idx_sm[0])) > 5
        idx_sm[jumps] = np.nan
        nan_mask = np.isnan(idx_sm)
        valid_mask = ~nan_mask

        if nan_mask.any() and valid_mask.any():
            interp_vals = np.interp(
                np.flatnonzero(nan_mask),  # x positions to fill
                np.flatnonzero(valid_mask),  # x of known points
                idx_sm[valid_mask]  # y of known points
            )
            idx_sm[nan_mask] = interp_vals
    else:
        idx_sm = idx_arr
    _dbg(f"[11] Savitzky–Golay smoothing applied (window={win})")

    idx_clean = xr.DataArray(idx_sm.astype(int), dims=("ping_time",),
                             coords={"ping_time": ping_vals}, name="seabed_idx")

    # 12) Map index to seabed depth
    seabed_depth = er.isel({vdim: idx_clean}).fillna(er.isel({vdim: -1}))
    seabed_depth.name = "seabed_depth"
    _dbg("[12] Seabed depth mapped; function finished\n")

    return idx_clean, seabed_depth, vdim


def __compute_seabed_line(
    ds: xr.Dataset,
    channel: str,
    r0_m: float,
    r1_m: float,
    thr_dB: float = -40.0,
    fixed_percentile: float = 75.0,
    delta_db: float = 3.0,
    erosion_kernel: tuple[int, int] = (3, 3),
    dilation_kernel: tuple[int, int] = (5, 5),
    use_adaptive_thresholding: bool = False,
    continuity_window_pings: int = 5,
    continuity_min_pings: int = 2,
    adaptive_window_pings: int = 10,
    adaptive_std_multiplier: float = 1.0,
    savgol_max_index_jump: int = 10,
    savgol_window_length: int = 31,
    savgol_polyorder: int = 1
) -> xr.DataArray:
    """
    Compute a single‐ping seabed depth line (no masking) for plotting.

    Returns
    -------
    seabed_depth : xr.DataArray
        dims: (ping_time,)
        coords: ping_time, depth
        values: seabed depth [m] per ping
    """
    # 1) Select the channel slice
    if "channel" in ds.dims:
        sub = ds.sel(channel=channel, drop=True) if channel in ds.coords.get("channel", []) else ds.isel(channel=0)
    elif "frequency" in ds.dims:
        sub = ds.sel(frequency=channel, drop=True) if channel in ds.coords.get("frequency", []) else ds.isel(
            frequency=0)
    else:
        sub = ds

    Sv = sub["Sv"]
    er = sub["echo_range"]
    vdim = "range_sample" if "range_sample" in Sv.dims else "depth"
    ping_dim = Sv.dims[0] if Sv.dims[0] != vdim else Sv.dims[1]  # a bit more robust

    # 2) Median‐filter full Sv to kill isolated spikes
    Sv_filtered = xr.apply_ufunc(
        median_filter,
        Sv,
        kwargs={"size": (3, 3)},  # Assuming Sv is (ping_time, range_sample)
        dask="parallelized",
        output_dtypes=[Sv.dtype],
    )

    # 3) Gate for threshold computation only
    gate = (er >= r0_m) & (er <= r1_m)
    Sv_gate_for_threshold = Sv_filtered.where(gate).chunk({vdim: -1})  # Use full depth chunk for percentile

    # 4) Choose thresholding method
    # Fixed‐percentile dynamic threshold
    perc_thr = xr.apply_ufunc(
        np.nanpercentile,
        Sv_gate_for_threshold,
        input_core_dims=[[vdim]],
        kwargs={"q": fixed_percentile},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[Sv.dtype],
    )
    thr2 = (perc_thr - delta_db).broadcast_like(Sv_filtered)

    # 5) Enforce static floor thr_dB
    thr2 = thr2.clip(min=thr_dB)
    # thr2 = xr.where(thr2 > thr_dB, thr2, thr_dB)

    # 6) Build candidate mask on filtered Sv
    cand_initial = (Sv_filtered > thr2) & gate

    # 7) Morphology (erosion then dilation)
    struct_erosion = np.ones(erosion_kernel, dtype=bool)
    struct_dilation = np.ones(dilation_kernel, dtype=bool)

    cand_morphed = xr.apply_ufunc(
        _erode, cand_initial, kwargs={'structure': struct_erosion, 'border_value': 0},
        dask='parallelized', output_dtypes=[bool]
    )
    cand_morphed = xr.apply_ufunc(
        _dilate, cand_morphed, kwargs={'structure': struct_dilation, 'border_value': 0},
        dask='parallelized', output_dtypes=[bool]
    )

    # 8) Require continuity across pings
    if continuity_window_pings % 2 == 0:
        actual_continuity_window = continuity_window_pings + 1  # Make it odd for symmetrical window
    else:
        actual_continuity_window = continuity_window_pings

    rolling_pres = cand_morphed.rolling({ping_dim: actual_continuity_window},
                                        center=True,
                                        min_periods=1).sum(skipna=True)  # sum booleans (True=1, False=0)
    cand_final = rolling_pres >= continuity_min_pings

    # 9) Pick the deepest True per ping
    vdim_indices = xr.DataArray(np.arange(cand_final.sizes[vdim]), dims=vdim, coords={vdim: cand_final[vdim]})
    indexed_cand = xr.where(cand_final, vdim_indices, np.nan)
    seabed_idx_primary = indexed_cand.max(dim=vdim)

    # 10) Safe fallback (if no cand True, pick first Sv > thr2 or bottom)
    indexed_cand_fallback = xr.where(cand_initial, vdim_indices, np.nan)
    seabed_idx_fallback = indexed_cand_fallback.max(dim=vdim)
    seabed_idx = xr.where(seabed_idx_primary.notnull(), seabed_idx_primary, seabed_idx_fallback)

    # 11) De-spike with Savitzky–Golay
    ping_vals = seabed_idx.coords[ping_dim].values
    idx_arr_raw = seabed_idx.compute().data  # Get numpy array

    valid_mask = ~np.isnan(idx_arr_raw)
    if np.any(valid_mask):  # Proceed if there's any non-NaN data
        interp_indices = np.arange(len(idx_arr_raw))
        idx_arr_interp = np.interp(interp_indices, interp_indices[valid_mask], idx_arr_raw[valid_mask])

        N = len(idx_arr_interp)
        win = min(savgol_window_length, N if N % 2 else N - 1)  # Ensure window is odd and <= N
        win = max(3, win)  # Ensure window is at least 3

        if N >= 3 and win > 0:  # Savgol needs window > polyorder and N > window
            # Ensure polyorder < win
            actual_polyorder = min(savgol_polyorder, win - 1) if win > savgol_polyorder else 1
            if win <= actual_polyorder:
                actual_polyorder = win - 1  # must be less
            if actual_polyorder < 0:
                actual_polyorder = 0  # for win=1 case.

            if win > actual_polyorder >= 0:
                sm = savgol_filter(idx_arr_interp, win, polyorder=actual_polyorder, mode="interp")
            else:  # Not enough points for savgol or invalid params
                sm = idx_arr_interp.copy()
        else:  # Not enough data points for SavGol
            sm = idx_arr_interp.copy()

        # Post-savgol jump removal (more cautious)
        # This replaces a jump with NaN, to be potentially re-interpolated later or left as missing
        diff_sm = np.abs(np.diff(sm, prepend=sm[0]))
        sm[diff_sm > savgol_max_index_jump] = np.nan  # Mark large jumps as NaN

        # Re-interpolate NaNs created by jump removal or if original data was all NaNs after first interp
        final_valid_mask = ~np.isnan(sm)
        if np.any(final_valid_mask) and np.any(
                ~final_valid_mask):  # if there are some nans to fill and some valid points
            sm = np.interp(interp_indices, interp_indices[final_valid_mask], sm[final_valid_mask])
        elif not np.any(final_valid_mask):  # All NaNs
            pass  # Keep all NaNs

        idx_clean_np = sm.astype(float)  # Keep as float due to NaNs and interpolation
    else:  # All NaNs to begin with
        idx_clean_np = idx_arr_raw.astype(float)

    idx_clean = xr.DataArray(idx_clean_np,
                             dims=(ping_dim,),
                             coords={ping_dim: ping_vals},
                             name="seabed_idx")

    # 12) Map index to seabed depth
    seabed_depth_values = np.full_like(idx_clean.data, np.nan, dtype=float)

    valid_idx_mask = idx_clean.notnull()
    if valid_idx_mask.any():
        max_echo_range_idx = er.sizes[vdim] - 1
        clean_integer_indices = idx_clean.where(valid_idx_mask).astype(int).clip(0, max_echo_range_idx)

        if er.ndim == 1:
            selected_depths = er.isel({vdim: clean_integer_indices.where(valid_idx_mask, drop=True)})
            seabed_depth_values[valid_idx_mask.data] = selected_depths.data
        elif er.ndim == 2 and ping_dim in er.dims and vdim in er.dims:
            temp_depths = xr.DataArray(np.nan, dims=[ping_dim], coords={ping_dim: ds[ping_dim]})
            for i, p_time in enumerate(ds[ping_dim].जहां(valid_idx_mask, drop=True)[ping_dim].values):
                p_idx_val = clean_integer_indices.sel({ping_dim: p_time}).item()
                temp_depths.loc[{ping_dim: p_time}] = er.sel({ping_dim: p_time, vdim: p_idx_val}).item()
            seabed_depth_values[valid_idx_mask.data] = temp_depths.where(valid_idx_mask, drop=True).data

        else:  # er is 1D
            valid_clean_indices = idx_clean.where(valid_idx_mask).astype(int).clip(0, er.sizes[vdim] - 1)
            _seabed_depth_temp = xr.DataArray(np.nan, coords=idx_clean.coords, dims=idx_clean.dims)

            # Get depths for pings that have a valid index
            actual_indices_to_select = valid_clean_indices.where(valid_idx_mask, drop=True)
            if actual_indices_to_select.size > 0:
                selected_depths = er.isel({vdim: actual_indices_to_select})
                _seabed_depth_temp[valid_idx_mask] = selected_depths.data
            seabed_depth_values = _seabed_depth_temp.data

    seabed_depth = xr.DataArray(seabed_depth_values,
                                dims=(ping_dim,),
                                coords={ping_dim: ds[ping_dim].coords[ping_dim]},  # Use original ping_time coordinate
                                name="seabed_depth")
    # seabed_depth = seabed_depth.assign_coords(depth=("ping_time", seabed_depth.data))

    return idx_clean, seabed_depth


def _default_mask(Sv: xr.DataArray) -> xr.DataArray:
    """Return an “all-water” mask and a zero seabed‐index."""
    mask = xr.ones_like(Sv, bool, drop=True)
    zeros = np.zeros(len(mask.ping_time), int)
    return mask.assign_coords(seabed_idx=("ping_time", zeros))


def _safe_apply_ufunc(xda, func, kernel):
    """Chunk‐safe morphology with halo‐padding on a 2D DataArray."""
    struct = np.ones(kernel, bool)
    pad = {xda.dims[0]: kernel[0] // 2, xda.dims[1]: kernel[1] // 2}
    return xr.apply_ufunc(
        lambda blk, **_: func(blk, structure=struct),
        xda,
        dask="parallelized",
        vectorize=False,
        output_dtypes=[bool],
        dask_gufunc_kwargs=dict(depth=pad, boundary="nearest")
    )


def get_true_seabed_mask(
        ds: xr.Dataset,
        channel: str,
        *,
        r0_m: float,
        r1_m: float,
        thr_dB: float = -40.0,
        delta_db: float = 3.0,
        erosion_kernel: tuple[int, int] = (3, 3),
        dilation_kernel: tuple[int, int] = (5, 5),
) -> xr.DataArray:
    """
    Detect the true seabed and return a boolean mask that **keeps everything
    above it** and masks everything below.

    Steps
    -----
    1. Threshold Sv within a user gate (r0_m–r1_m).
    2. Morphological clean-up (erosion ▸ dilation).
    3. For each ping, locate the shallowest True → seabed depth.
    4. Broadcast depth line downward to mask the seabed and deeper.

    Returns
    -------
    xr.DataArray(bool) with dims ('ping_time','range_sample')
        True  – water column to keep
        False – samples at/under detected seabed
    """
    if "channel" in ds.dims:
        sub = ds.sel(channel=channel) if "channel" in ds.coords else ds.isel(channel=0)
    elif "frequency" in ds.dims:
        sub = ds.sel(frequency=channel) if "frequency" in ds.coords else ds.isel(frequency=0)
    else:
        sub = ds

    Sv = sub["Sv"]  # dims: (ping_time, range_sample | depth)
    vdim = "range_sample" if "range_sample" in Sv.dims else "depth"
    er = sub["echo_range"]  # metres

    # ------------------------------------------------------------------ #
    # 1) gate + rough threshold                                          #
    # ------------------------------------------------------------------ #
    gate = (er >= r0_m) & (er <= r1_m)
    if not bool(gate.any()):
        warnings.warn("Gate outside data – using default mask")
        return _default_mask(Sv)

    cand0 = (Sv > thr_dB) & gate
    if not bool(cand0.any()):
        warnings.warn("No echoes above thr_dB – using default mask")
        return _default_mask(Sv)

    # ------------------------------------------------------------------ #
    # 2) morphology (chunk-safe)                                         #
    # ------------------------------------------------------------------ #
    Sv_gate = Sv.where(gate).chunk({vdim: -1})
    med_ping = xr.apply_ufunc(
        np.nanmedian,
        Sv_gate,
        input_core_dims=[[vdim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[Sv.dtype]
    )
    thr2 = (med_ping - delta_db).broadcast_like(Sv)

    cand = (Sv > thr2) & gate
    cand = _safe_apply_ufunc(cand, _erode, erosion_kernel)
    cand = _safe_apply_ufunc(cand, _dilate, dilation_kernel)

    # ------------------------------------------------------------------ #
    # 3) seabed index per ping  (robust fallback)                        #
    # ------------------------------------------------------------------ #
    main_idx = cand.argmax(dim=vdim)
    fb_idx = (Sv > thr2).argmax(dim=vdim)
    has_bot = cand.any(dim=vdim)

    idx = xr.where(has_bot, main_idx, fb_idx)
    idx = (
        idx.where(idx > 0)
        .ffill("ping_time")
        .bfill("ping_time")
        .fillna(Sv[vdim].max())
    )

    # 4) two‐stage moving‐window smooth to kill narrow spikes
    #    ensure full‐axis chunk so rolling works in one pass
    ping_np = idx["ping_time"].values
    idx_np = idx.compute().values.astype(int)

    # 4.1) Savitzky–Golay smoothing (window ≤ 31 pings)
    N = len(idx_np)
    win = min(31, N if N % 2 == 1 else N - 1)
    sm = savgol_filter(idx_np, win, polyorder=1, mode="interp")

    # 4.2) clamp any single-ping jump > 5 bins
    max_jump = 5
    for i in range(1, N):
        if abs(sm[i] - sm[i - 1]) > max_jump:
            sm[i] = sm[i - 1]

    # rebuild a clean DataArray
    idx_clean = xr.DataArray(
        sm.astype(int),
        dims=("ping_time",),
        coords={"ping_time": ping_np},
        name="seabed_idx"
    )

    # 5) final mask
    mask = (Sv[vdim] < idx_clean.broadcast_like(Sv))
    mask.name = "seabed_water_mask"

    return mask.assign_coords(seabed_idx=idx_clean)


def _is_38khz(value, tol_hz=500.0) -> bool:
    """Return True if *value* represents 38 kHz within ±tol_hz."""
    # 1) numeric (Hz or kHz)
    if np.issubdtype(type(value), np.number):
        hz = float(value) * 1e3 if value < 1e3 else float(value)
        return abs(hz - 38_000.) <= tol_hz

    # 2) string  e.g.  "38", "38kHz", "ES38-18", "EKA … 38 …"
    if isinstance(value, (bytes, str)):
        txt = value.decode() if isinstance(value, bytes) else value
        # pull first number token
        m = re.search(r"(\d+(?:\.\d+)?)", txt)
        if m:
            return _is_38khz(float(m.group(1)), tol_hz=tol_hz)

    return False


def find_38khz_channel(ds: xr.Dataset, tol_hz: float = 500.0) -> xr.Dataset:
    """
    Return a *view* of *ds* containing **only the 38 kHz channel**.

    Priority order for matching:
    1. `frequency_nominal` coordinate (Hz)
    2. Dimension coordinates named 'channel' or 'frequency'
    3. Already single-channel dataset → returned unchanged.

    Parameters
    ----------
    ds : xr.Dataset
        Multi-channel dataset with variable 'Sv'.
    tol_hz : float
        Allowed deviation from 38 000 Hz (default ±500 Hz).

    Raises
    ------
    KeyError if no coordinate matches 38 kHz.
    """
    # ---------------------------------------------------------------
    # Case 0: dataset has no multi-channel dimension
    # ---------------------------------------------------------------
    if {"channel", "frequency"}.isdisjoint(ds.dims):
        return ds  # already single-channel

    chan_dim = "channel" if "channel" in ds.dims else "frequency"

    # ---------------------------------------------------------------
    # Case 1: look for 'frequency_nominal'
    # ---------------------------------------------------------------
    if "frequency_nominal" in ds.data_vars:
        fn_var = ds["frequency_nominal"]
        if chan_dim in fn_var.dims:
            fn_vec = _to_numpy(fn_var)
            for idx, val in enumerate(fn_vec):
                if _is_38khz(val, tol_hz):
                    return ds.isel({chan_dim: idx})

    # ---------------------------------------------------------------
    # Case 2: fall back to textual / numeric channel labels
    # ---------------------------------------------------------------
    coord = ds.coords.get(chan_dim, None)
    if coord is None:  # fabricate numeric index coord
        coord = xr.DataArray(np.arange(ds.dims[chan_dim]), dims=chan_dim)
        ds = ds.assign_coords({chan_dim: coord})

    for idx, val in enumerate(coord.values):
        if _is_38khz(val, tol_hz):
            return ds.isel({chan_dim: idx})

    # ---------------------------------------------------------------
    # No match found
    # ---------------------------------------------------------------
    def _vals(x):
        if x is None:
            return "<<missing>>"
        if hasattr(x, "values"):
            return x.values
        return x

    raise KeyError(
        "Could not locate a 38 kHz channel.\n"
        f"  frequency_nominal: {_vals(ds.coords.get('frequency_nominal', None))}\n"
        f"  channel coord    : {_vals(ds.coords.get('channel', None))}\n"
        f"  frequency coord  : {_vals(ds.coords.get('frequency', None))}"
    )


def _to_numpy(arr, max_size: int = 1_000_000) -> np.ndarray:
    """
    Return *arr* as a NumPy ndarray **only if** it contains ≤ `max_size`
    elements.  Works for xarray objects, NumPy arrays and Dask arrays.

    Raises
    ------
    ValueError  – if the array is larger than `max_size`.
    """
    # unwrap DataArray → .data
    data = arr.data if hasattr(arr, "data") else arr

    if data.size > max_size:
        raise ValueError(
            f"Refusing to materialise large array (size={data.size}) "
            "— adjust `max_size` if you really need this."
        )

    if isinstance(data, da.Array):
        return data.compute()  # still small, so safe to compute

    return np.asarray(data)


def broadcast_mask_to_all_channels(
        mask_2d: xr.DataArray,
        original_ds: xr.Dataset,
        chan_dim: str | None = None,
        mask_name: str = "seabed_water_mask",
) -> xr.DataArray:
    """
    Expand a 2-D (ping_time × range_sample) mask to a 3-D
    (channel × ping_time × range_sample) mask that aligns with *original_ds*.

    Parameters
    ----------
    mask_2d : xr.DataArray(bool)
        Output of `seabed_mask` for the 38 kHz slice; **dims must be exactly
        ('ping_time', 'range_sample')** (order doesn’t matter).
    original_ds : xr.Dataset
        The full multi-channel dataset from which the 38 kHz slice was taken.
    chan_dim : str, optional
        Name of the channel dimension ('channel' or 'frequency').
        If None, it is auto-detected; if the dataset has no such dimension the
        original 2-D mask is returned unchanged.
    mask_name : str
        Name to assign to the returned DataArray.

    Returns
    -------
    xr.DataArray(bool)
        Broadcast mask with dims (channel, ping_time, range_sample) that can be
        directly applied to `original_ds["Sv"]`.
    """
    # 1.  Detect the channel dimension; skip if single-channel
    if chan_dim is None:
        for cand in ("channel", "frequency"):
            if cand in original_ds.dims:
                chan_dim = cand
                break
    if chan_dim is None:
        return mask_2d  # nothing to broadcast

    # 2.  Ensure mask dims/coords match target dims
    #     (transpose/reindex keeps it lazy)
    mask_aligned = mask_2d.transpose(...).reindex_like(
        original_ds.drop_vars(set(original_ds.data_vars)),
        # drop data_vars → dataset with only coords/dims for alignment
        method=None,
        copy=False,
    )

    # 3.  Expand along the channel axis – broadcast, no compute
    mask_3d = mask_aligned.expand_dims(
        {chan_dim: original_ds[chan_dim]},
        axis=0 if mask_aligned.dims[0] != chan_dim else None,
    )

    # 4.  Give it a nice name and return
    return mask_3d.rename(mask_name)


def get_seabed_mask_multichannel(ds: xr.Dataset, parameters: dict = None) -> xr.DataArray:
    channel_list = ds["channel"].values
    mask_list = []

    if parameters is None:
        parameters = BLACKWELL_DEFAULT_PARAMS

    for channel in channel_list:
        mask = max_sv_seabed_mask(
            ds,
            channel=channel,
            depth_min=parameters["r0"],
            depth_max=parameters["r1"],
            threshold_db=(-40.0, -60.0)
        )

        # mask = get_seabed_mask(
        #     source_Sv,
        #     parameters=parameters,
        #     desired_channel=channel,
        #     method=method,
        # )
        mask_list.append(mask)
    mask = create_multichannel_mask(mask_list, channel_list)
    return mask


def create_multichannel_mask(masks: [xr.Dataset], channels: [str]) -> xr.Dataset:
    if len(masks) != len(channels):
        raise ValueError("number of masks and of channels provided should be the same")

    for i in range(0, len(masks)):
        mask = masks[i]
        if "channel" in mask.dims:
            masks[i] = mask.isel(channel=0)

    result = xr.concat(
        masks, Index(channels, name="channel"), data_vars="all", coords="all", join="exact"
    )
    return result


def get_seabed_mask(source_Sv: xr.Dataset,
                    parameters: dict,
                    desired_channel: str = None,
                    desired_frequency: int = None,
                    method: str = "ariza"
                    ) -> xr.DataArray:
    """
    Create a mask based on the identified signal attenuations of Sv values at 38KHz.
    Parameters
    ----------
    source_Sv: xr.Dataset or str or pathlib.Path
        If a Dataset this value contains the Sv data to create a mask for,
        else it specifies the path to a zarr or netcdf file containing
        a Dataset. This input must correspond to a Dataset that has the
        coordinate ``channel`` and variables ``frequency_nominal`` and ``Sv``.
    desired_channel: str - channel to generate the mask for
    desired_frequency: int - desired frequency, in case the channel isn't directly specified
    method: str with either "ariza", "blackwell", based on the preferred method
        for seabed mask generation
    Returns
    -------
    xr.DataArray
        A DataArray containing the mask for the Sv data. Regions satisfying the thresholding
        criteria are filled with ``True``, else the regions are filled with ``False``.

    Raises
    ------
    ValueError
        If neither "ariza", "blackwell" are given

    Notes
    -----


    Examples
    --------

    """
    mask_map = {
        "ariza": _ariza,
        "blackwell": _blackwell
    }

    if method not in mask_map.keys():
        raise ValueError(f"Unsupported method: {method}")

    mask = mask_map[method](source_Sv, desired_channel, parameters)

    return mask


def _get_seabed_range(mask: xr.DataArray):
    """
    Given a seabed mask, returns the depth of the seabed

    Args:
        mask (xr.DataArray): seabed mask

    Returns:
        xr.DataArray: a ping_time-sized array containing the depth seabed depth,
        or max depth if no seabed is detected

    """
    seabed_depth = mask.argmax(dim="depth").compute()
    seabed_depth[seabed_depth == 0] = mask.depth.max().item()
    return seabed_depth


def _morpho(mask: xr.DataArray, operation: str, c: int, k: int):
    """
    Given a preexisting 1/0 mask, run erosion or dilation cycles on it to remove noise

    Args:
        mask (xr.DataArray): xr.DataArray with 1 and 0 data
        operation(str): dilation, erosion
        c (int): number of cycles.
        k (int): 2-elements tuple with vertical and horizontal dimensions
                      of the kernel.

    Returns:
        xr.DataArray: A DataArray containing the denoised mask.
            Regions satisfying the criteria are 1, others are 0
    """
    function_dict = {"dilation": binary_dilation, "erosion": binary_erosion}

    if c > 0:
        dask_mask = da.asarray(mask, allow_unknown_chunksizes=False)
        dask_mask.compute_chunk_sizes()
        dask_mask = function_dict[operation](
            dask_mask,
            structure=da.ones(shape=k, dtype=bool),
            iterations=c,
        ).compute()
        dask_mask = da.asarray(dask_mask, allow_unknown_chunksizes=False)
        dask_mask.compute()
        mask.values = dask_mask.compute()
    return mask


def _erode_dilate(mask: xr.DataArray, ec: int, ek: int, dc: int, dk: int):
    """
    Given a preexisting 1/0 mask, run erosion and dilation cycles on it to remove noise

    Args:
        mask (xr.DataArray): xr.DataArray with 1 and 0 data
        ec (int): number of erosion cycles.
        ek (int): 2-elements tuple with vertical and horizontal dimensions
                      of the erosion kernel.
        dc (int): number of dilation cycles.
        dk (int): 2-elements tuple with vertical and horizontal dimensions
                      of the dilation kernel.

    Returns:
        xr.DataArray: A DataArray containing the denoised mask.
            Regions satisfying the criteria are 1, others are 0
    """
    mask = _morpho(mask, "erosion", ec, ek)
    mask = _morpho(mask, "dilation", dc, dk)
    return mask


def _create_range_mask(Sv_ds: xr.DataArray, desired_channel: str, thr: int, r0: int, r1: int):
    """
    Return a raw threshold/range mask for a certain dataset and desired channel

    Args:
        Sv_ds (xr.DataArray): xr.DataArray with Sv data for multiple channels (dB)
        desired_channel(str): Name of the desired frequency channel
        r0 (int): minimum range below which the search will be performed (m).
        r1 (int): maximum range above which the search will be performed (m).
        thr (int): Sv threshold above which seabed might occur (dB).

    Returns:
        dict: a dict containing the mask and whether or not further processing is necessary
            mask (xr.DataArray): a basic range/threshold mask.
                                Regions satisfying the criteria are 1, others are 0
            ok (bool): should the mask be further processed  or is there no data to be found?

    """
    try:
        channel_Sv = Sv_ds.sel(channel=desired_channel)
        Sv = channel_Sv["Sv"]
        r = channel_Sv["echo_range"][0]

        # Early exit if out of bounds
        if (r0 > r.max()) or (r1 < r.min()):
            warnings.warn(
                "The searching range is outside the echosounder range. "
                "Returning default unmasked data."
            )
            mask = xr.DataArray(
                np.ones_like(Sv, dtype=bool),
                dims=("ping_time", "depth"),
                coords={"ping_time": Sv.ping_time, "depth": Sv.depth},
            )
            return {"mask": mask, "ok": False, "Sv": Sv, "range": r}

        # Lazily calculate bounding indices
        # up_idx = abs(r - r0).argmin(dim="depth")
        # lw_idx = abs(r - r1).argmin(dim="depth")

        # Create depth bounds directly from values instead of indices
        # upper_range = r.isel(depth=up_idx)
        # lower_range = r.isel(depth=lw_idx)

        base_mask = xr.where(Sv > thr, 1, 0).drop_vars("channel", errors="ignore")

        # Lazy mask
        echo_range = channel_Sv["echo_range"]
        range_filter = (echo_range >= r0) & (echo_range <= r1)
        masked = base_mask.where(range_filter, other=0)

        # Check if any signal survives — lazily
        # nonzero = masked.sum()

        # Return everything deferred
        return {
            "mask": masked,
            "ok": xr.apply_ufunc(lambda x: x > 0, masked.sum(), dask="allowed"),
            "Sv": Sv,
            "range": r
        }
    except Exception as e:
        print(e)
        raise ValueError("Error in creating range mask: " + str(e))


def _mask_down(mask: xr.DataArray):
    """
    Given a seabed mask, masks all signal under the detected seabed

    Args:
          mask (xr.DataArray): seabed mask

    Returns:
           xr.DataArray(mask with area under seabed masked)
    """
    seabed_depth = _get_seabed_range(mask)
    mask = (mask["depth"] <= seabed_depth).transpose()
    return mask


# move to utils and rewrite transient noise/fielding to use this once merged
def _erase_floating_zeros(mask: xr.DataArray):
    """
    Given a boolean mask, turns back to True any "floating" False values,
    e.g. not attached to the max range

    Args:
        mask: xr.DataArray - mask to remove floating values from

    Returns:
        xr.DataArray - mask with floating False values removed

    """
    flipped_mask = mask.isel(depth=slice(None, None, -1))
    flipped_mask["depth"] = mask["depth"]
    ft = len(flipped_mask.depth) - flipped_mask.argmax(dim="depth")

    first_true_indices = xr.DataArray(
        line_to_square(ft, mask, dim="depth").transpose(),
        dims=("ping_time", "depth"),
        coords={"ping_time": mask.ping_time, "depth": mask.depth},
    )

    indices = xr.DataArray(
        line_to_square(mask["depth"], mask, dim="ping_time"),
        dims=("ping_time", "depth"),
        coords={"ping_time": mask.ping_time, "depth": mask.depth},
    )
    spike_mask = mask.where(indices > first_true_indices, True)

    mask = spike_mask
    return mask


def _experimental_correction(mask: xr.DataArray, Sv: xr.DataArray, thr: int):
    """
    Given an existing seabed mask, the single-channel dataset it was created on
    and a secondary, lower threshold, it builds the mask up until the Sv falls below the threshold

    Args:
          mask (xr.DataArray): seabed mask
          Sv (xr.DataArray): single-channel Sv data the mask was build on
          thr (int): secondary threshold

    Returns:
          xr.DataArray: mask with secondary threshold correction applied

    """
    secondary_mask = xr.where(Sv < thr, 1, 0).drop("channel")
    secondary_mask.fillna(1)
    fill_mask = secondary_mask & mask
    spike_mask = _erase_floating_zeros(fill_mask)
    return spike_mask


def _cut_spikes(mask: xr.DataArray, maximum_spike: int):
    """
    In the Ariza seabed detecting method, large shoals can be falsely detected as
    seabed. Their appearance on the seabed mask is large vertical "spikes".
    We want to remove any such spikes from the dataset using maximum_spike
    as a control parameter.

    If this option is used, we also recommend applying the _experimental_correction,
    even if with the same threshold as the initial threshold, to fill up any
    imprecisions in the interpolated seabed

    Args:
        mask (xr.DataArray):
        maximum_spike(int): maximum height, in range samples, acceptable before
                            we start removing that data

    Returns:
        xr.DataArray: the corrected mask
    """
    int_mask = (~mask.copy()).astype(int)
    seabed = _get_seabed_range(int_mask)
    shifted_seabed = seabed.shift(ping_time=-1, fill_value=seabed[-1])
    spike = seabed - shifted_seabed
    spike_sign = xr.where(abs(spike) > maximum_spike, xr.where(spike > 0, 1, -1), 0)
    spike_cs = spike_sign.cumsum(dim="ping_time")
    # for i in spike:
    #     print(i.item())

    # mask spikes
    nan_mask = xr.where(spike_cs > 0, np.nan, xr.where(spike_sign == -1, np.nan, mask))

    # fill in with interpolated values from non-spikes
    mask_interpolated = nan_mask.interpolate_na(dim="ping_time", method="nearest").astype(bool)

    return mask_interpolated


def _ariza(Sv_ds: xr.DataArray, desired_channel: str, parameters: dict = ARIZA_DEFAULT_PARAMS):
    """
    Mask Sv above a threshold to get potential seabed features. These features
    are eroded first to get rid of fake seabeds (spikes, schools, etc.) and
    dilated afterwards to fill in seabed breaches. Seabed detection is coarser
    than other methods (it removes water nearby the seabed) but the seabed line
    never drops when a breach occurs. Suitable for pelagic assessments and
    reconmended for non-supervised processing.

    Args:
        Sv_ds (xr.DataArray): xr.DataArray with Sv data for multiple channels (dB)
        desired_channel(str): Name of the desired frequency channel
        parameters: parameter dict, should contain:
            r0 (int): minimum range below which the search will be performed (m).
            r1 (int): maximum range above which the search will be performed (m).
            roff (int): seabed range offset (m).
            thr (int): Sv threshold above which seabed might occur (dB).
                Can be a tuple, case in which a secondary experimental
                thresholding correction is applied
            ec (int): number of erosion cycles.
            ek (int): 2-elements tuple with vertical and horizontal dimensions
                      of the erosion kernel.
            dc (int): number of dilation cycles.
            dk (int): 2-elements tuple with vertical and horizontal dimensions
                      of the dilation kernel.
            maximum_spike(int): optional, if not None, used to determine the maximum
                    allowed height of the "spikes" potentially created by
                    dense shoals before masking them out. If used, applying
                    a secondary threshold correction is recommended


    Returns:
        xr.DataArray: A DataArray containing the mask for the Sv data.
            Regions satisfying the thresholding criteria are True, others are False
    """
    parameter_names = ["r0", "r1", "roff", "thr", "ec", "ek", "dc", "dk"]
    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError(
            "Missing parameters - should be: "
            + str(parameter_names)
            + ", are: "
            + str(parameters.keys())
        )
    r0 = parameters["r0"]
    r1 = parameters["r1"]
    thr = parameters["thr"]
    ec = parameters["ec"]
    ek = parameters["ek"]
    dc = parameters["dc"]
    dk = parameters["dk"]
    secondary_thr = None
    maximum_spike = None
    if "maximum_spike" in parameters.keys():
        maximum_spike = parameters["maximum_spike"]

    if isinstance(thr, int) is False:
        secondary_thr = thr[1]
        thr = thr[0]

    # create raw range and threshold mask, if no seabed is detected return empty
    raw = _create_range_mask(Sv_ds, desired_channel=desired_channel, thr=thr, r0=r0, r1=r1)
    mask = raw["mask"]
    if raw["ok"] is False:
        return mask

    # run erosion and dilation denoising cycles
    mask = _erode_dilate(mask, ec, ek, dc, dk)

    # mask areas under the detected seabed
    mask = _mask_down(mask)

    # apply spike correction
    if maximum_spike is not None:
        mask = _cut_spikes(mask, maximum_spike)

    # apply experimental correction, if specified
    if secondary_thr is not None:
        mask = _experimental_correction(mask, raw["Sv"], secondary_thr)

    return mask


def max_sv_seabed_mask(
        ds: xr.Dataset,
        channel: str = None,
        depth_min: float = 10.0,
        depth_max: float = 1000.0,
        range_offset: float = 0.0,
        threshold_db: tuple = (-40.0, -60.0),
        smooth_pings: int = 5,
        mean_len: int = 5
) -> xr.DataArray:
    """
    Detects seabed based on maximum Sv within a specified depth window for a given channel.
    Applies an upward trace until Sv falls below a secondary threshold, with optional smoothing.

    Parameters:
        ds (xr.Dataset): Dataset with variables Sv (in dB), dimensions: (channel, ping_time, depth)
        channel (str): Channel name to select from the 'channel' dimension
        depth_min (float): Minimum range (m) for search window
        depth_max (float): Maximum range (m) for search window
        range_offset (float): Range offset to subtract from final bottom pick (m)
        threshold_db (tuple): (primary_threshold, secondary_threshold) in dB
        smooth_pings (int): Window size (in pings) for moving median smoothing
        mean_len (int): Window size for rolling mean calculation

    Returns:
        xr.DataArray: Boolean mask (True = seabed and below), shape: (channel, ping_time, depth)
    """
    # guarantee depth is a single chunk
    Sv = ds["Sv"].sel(channel=channel).chunk({"depth": -1})
    depth = ds["depth"]

    win = Sv.sel(depth=slice(depth_min, depth_max))
    win = win.where(win.notnull().any("depth"))

    # 1) primary bottom pick (keeps NaNs)
    # idxmax works lazily & returns depth coordinate
    bottom_depth = win.where(win >= threshold_db[0]).idxmax("depth")
    bottom_depth = bottom_depth.where(~bottom_depth.isnull())
    bottom_depth = bottom_depth.assign_coords(ping_time=ds.ping_time)

    # 2) 5-sample linear-mean to test “step-up” threshold
    lin = 10.0 ** (Sv / 10.0)
    mean5 = lin.rolling(depth=mean_len, min_periods=1).mean()

    thresh_ok = (10 * np.log10(mean5) <= threshold_db[1])

    # 3) vectorised upward search
    def _step(depth_pick, ok_flags, mean_len):
        if np.isnan(depth_pick):
            return np.nan

        idx = np.searchsorted(depth.values, depth_pick)

        while idx >= mean_len and not ok_flags[idx - 1]:
            idx -= 1

        return float(idx)

    def _to_idx(depth_pick, shift, range_offset):
        """
        Convert depth coordinate to integer index minus shift & range-offset.
        Works on plain numpy scalars/arrays – safe for dask='parallelized'.
        """
        if np.isnan(depth_pick) or np.isnan(shift):
            return np.nan

        idx = np.searchsorted(depth.values, depth_pick)
        res = idx - shift - range_offset

        return float(res if res > 0 else 0.0)

    shift = xr.apply_ufunc(
        _step,
        bottom_depth,
        thresh_ok,
        kwargs=dict(mean_len=mean_len),
        input_core_dims=[[], ["depth"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    shift = shift.reindex(ping_time=ds.ping_time, fill_value=np.nan)

    bottom_idx = xr.apply_ufunc(
        _to_idx,
        bottom_depth,  # depth coordinate per ping (scalar/array)
        shift,  # shift returned by the first gufunc
        kwargs=dict(range_offset=range_offset),
        input_core_dims=[[], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    bottom_idx = bottom_idx.reindex(ping_time=ds.ping_time)

    # 4) optional ping-median smoothing
    if smooth_pings > 1:
        bottom_idx = (
            bottom_idx
            .rolling(ping_time=smooth_pings, center=True, min_periods=1)
            .median()
        )

    # 5) boolean mask by broadcasting (lazy)
    mask = depth >= bottom_idx
    mask = mask.transpose("ping_time", "depth")

    print(
        bottom_idx.sizes,
        mask.sizes,
        ds.ping_time.size
    )

    mask_da = xr.DataArray(
        mask,
        dims=("ping_time", "depth"),
        coords={"ping_time": ds.ping_time,
                "depth": depth},
        name="seabed_mask",
        attrs={"long_name": "bottom & below mask (True = seabed)"},
    ).expand_dims(channel=[channel])

    return mask_da


""" ///////////////////////////// BLACKWELL METHOD ////////////////////////////////// """


def _blackwell(Sv_ds: xr.DataArray, desired_channel: str, parameters: dict):
    """
    Detects and mask seabed using the split-beam angle and Sv, based in
    "Blackwell et al (2019), Aliased seabed detection in fisheries acoustic
    data". Complete article here: https://arxiv.org/abs/1904.10736

    Args:
        Sv_ds (xr.DataArray): xr.DataArray with Sv data for multiple channels (dB)
        desired_channel(str): Name of the desired frequency channel
        parameters: parameter dict, should contain:
            r0 (int): minimum range below which the search will be performed (m)
            r1 (int): maximum range above which the search will be performed (m)
            tSv (float): Sv threshold above which seabed is pre-selected (dB)
            ttheta (int): Theta threshold above which seabed is pre-selected (dB)
            tphi (int): Phi threshold above which seabed is pre-selected (dB)
            wtheta (int): window's size for mean square operation in Theta field
            wphi (int): window's size for mean square operation in Phi field

    Returns:
        xr.DataArray: A DataArray containing the mask for the Sv data.
            Regions satisfying the thresholding criteria are True, others are False
    """
    parameter_names = ["r0", "r1", "tSv", "ttheta", "tphi", "wtheta", "wphi"]
    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError(
            "Missing parameters - should be: "
            + str(parameter_names)
            + ", are: "
            + str(parameters.keys())
        )

    ttheta = parameters["ttheta"]
    tphi = parameters["tphi"]
    wtheta = parameters["wtheta"]
    wphi = parameters["wphi"]

    channel_Sv = Sv_ds.sel(channel=desired_channel)

    if not all(attr in channel_Sv.data_vars for attr in ["angle_alongship", "angle_athwartship"]):
        raise ValueError("Missing required angle variables: 'angle_alongship' and/or 'angle_athwartship'")

    theta = channel_Sv["angle_alongship"]
    phi = channel_Sv["angle_athwartship"]
    scaling_factor = (22.0 * 128.0) / 180.0  # Ensure float division
    theta = theta * scaling_factor
    phi = phi * scaling_factor

    dask_theta = da.asarray(theta, allow_unknown_chunksizes=False)
    dask_theta.compute_chunk_sizes()

    theta.values = convolve(
        dask_theta,
        weights=da.ones(shape=(wtheta, wtheta), dtype=float) / wtheta ** 2,
        mode="nearest",
    ).compute()

    dask_phi = da.asarray(phi, allow_unknown_chunksizes=False)
    dask_phi.compute_chunk_sizes()

    phi.values = convolve(
        dask_phi,
        weights=da.ones(shape=(wphi, wphi), dtype=float) / wphi ** 2,
        mode="nearest",
    ).compute()

    angle_mask = ~((theta > ttheta) | (phi > tphi))

    if angle_mask.all():
        warnings.warn(
            "No aliased seabed detected in Theta & Phi. "
            "A default mask with all True values is returned."
        )
        return angle_mask

    mask = theta > ttheta

    return mask


def line_to_square(one: xr.DataArray, two: xr.DataArray, dim: str):
    """
    Given a single dimension dataset and an example dataset with 2 dimensions,
    returns a two-dimensional dataset that is the single dimension dataset
    repeated as often as needed

    Args:
        one (xr.DataArray): data
        two (xr.DataArray): shape dataset
        dim (str): name of dimension to concat against

    Returns:
        xr.DataArray: the input dataset, with the same coords as dataset_size and
        the values repeated to fill it up
    """
    length = len(two[dim])
    array_list = [one for _ in range(0, length)]
    array = xr.concat(array_list, dim=dim)
    # return_data = xr.DataArray(data=array.values, dims=two.dims, coords=two.coords)
    return array.values


def frequency_nominal_to_channel(source_Sv, frequency_nominal: int):
    """
    Given a value for a nominal frequency, returns the channel associated with it
    """
    channels = source_Sv["frequency_nominal"].coords["channel"].values
    freqs = source_Sv["frequency_nominal"].values
    chan = channels[freqs == frequency_nominal]
    assert len(chan) == 1, "Frequency not uniquely identified"
    channel = chan[0]
    return channel
