import warnings

import numpy as np
import xarray as xr
import dask.array as da
from numpy.lib.stride_tricks import sliding_window_view
from functools import partial
from typing import Tuple, Dict


def _nearest_idx(r: np.ndarray, x: float) -> int:
    x = float(x)
    if x <= r[0]:  return 0
    if x >= r[-1]: return r.size - 1
    return int(np.abs(r - x).argmin())


def _db2lin(x: np.ndarray) -> np.ndarray:
    return np.power(10.0, x / 10.0)


def _lin2db(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(x)


def _mov_nanmedian_1d(y: np.ndarray, win: int) -> np.ndarray:
    """Centered moving NaN-median on 1D (len T) → shape (T,) with NaNs at edges."""
    T = y.shape[0]
    if win <= 1 or T == 0:
        return y.copy()

    if T < win:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m = np.nanmedian(y)
        return np.full(T, m)

    w = sliding_window_view(y, win)  # (T-win+1, win)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        core = np.nanmedian(w, axis=1)
    pre = win // 2
    post = win - 1 - pre
    return np.pad(core, (pre, post), mode="constant", constant_values=np.nan)


def _mov_nanmedian_rows(Y: np.ndarray, win: int) -> np.ndarray:
    """Centered moving NaN-median along axis=1 for 2D (S, T) → shape (S, T)."""
    S, T = Y.shape
    if win <= 1 or T == 0:
        return Y.copy()

    if T < win:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m = np.nanmedian(Y, axis=1, keepdims=True)
        return np.broadcast_to(m, (S, T)).copy()
    W = sliding_window_view(Y, win, axis=1)  # (S, T-win+1, win)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        core = np.nanmedian(W, axis=2)  # (S, T-win+1)
    pre = win // 2
    post = win - 1 - pre

    return np.pad(core, ((0, 0), (pre, post)), mode="constant", constant_values=np.nan)


def _fielding_mask_kernel(
    arr_dB: np.ndarray,  # shape (Z, T) == (depth, time)
    up: int, lw: int,  # deep reference band indices [up:lw)
    rmin: int,  # min depth index to consider masking (exclude_above)
    sf: int,  # vertical step in samples (≈ jumps in meters)
    n: int,  # pings on each side (block width = 2n+1)
    thr0: float,  # initial far-range threshold (dB)
    thr1: float,  # upward stop threshold (dB)
    maxts: float,  # max transient permitted (avoid seabed) (dB)
) -> np.ndarray:
    Z, T = arr_dB.shape
    # guardrails
    up = max(0, min(up, Z - 1))
    lw = max(up + 1, min(lw, Z))
    rmin = max(0, min(rmin, Z - 1))
    sf = max(1, int(sf))
    n = max(1, int(n))
    win_t = 2 * n + 1

    # linear array
    arr_lin = _db2lin(arr_dB)

    # deep-band ping stats (vectorized over time)
    deep = arr_lin[up:lw, :]  # (Ddeep, T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ping_med_lin = np.nanmedian(deep, axis=0)  # (T,)
        ping_p75_lin = np.nanpercentile(deep, 75, axis=0)

    ping_med_db = _lin2db(ping_med_lin)
    ping_p75_db = _lin2db(ping_p75_lin)

    # block median over time (centered, width = 2n+1) on deep-band medians
    blk_med_db = _lin2db(_mov_nanmedian_1d(ping_med_lin, win_t))  # (T,)

    # initial flag at far range (per time)
    init_flag = (ping_p75_db < maxts) & ((ping_med_db - blk_med_db) > thr0)  # (T,)

    if not init_flag.any():
        return np.zeros((Z, T), dtype=bool)  # nothing to do in this chunk

    # upward stepping: s = 1..Smax windows of height sf
    Smax = max((up - rmin) // sf, 0)
    if Smax == 0:
        # mask whole column below up for flagged pings
        cut = np.where(init_flag, up, Z)  # depth cutoff per time
        d = np.arange(Z)[:, None]
        return d >= cut[None, :]

    starts = up - sf * np.arange(1, Smax + 1)  # (Smax,)
    # per-step segment medians (linear), shape (Smax, T)
    seg_med_lin = np.empty((Smax, T), dtype=arr_lin.dtype)
    for k, s0 in enumerate(starts):
        s1 = s0 + sf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            seg_med_lin[k] = np.nanmedian(arr_lin[s0:s1, :], axis=0)

    # per-step block medians over time (linear → dB after)
    blk_seg_med_lin = _mov_nanmedian_rows(seg_med_lin, win_t)  # (Smax, T)

    # Δ(s, t) in dB
    seg_med_db = _lin2db(seg_med_lin)
    blk_seg_med_db = _lin2db(blk_seg_med_lin)
    delta = seg_med_db - blk_seg_med_db  # (Smax, T)

    # find first step where Δ < thr1 (only for initially flagged pings)
    meets = delta < thr1  # (Smax, T)
    meets[:, ~init_flag] = False

    any_true = meets.any(axis=0)
    first_idx = np.argmax(meets, axis=0)  # 0..Smax-1
    # stopping step index (1..Smax) or Smax if none
    s_stop = np.where(any_true, first_idx + 1, Smax)  # (T,)

    r0 = up - s_stop * sf  # (T,)
    r0 = np.maximum(r0, rmin)
    # if not flagged, set cutoff below array to produce False mask
    r0 = np.where(init_flag, r0, Z)

    # vectorized column-wise fill: mask depths ≥ r0
    d = np.arange(Z)[:, None]

    return d >= r0[None, :]


def transient_noise_mask(
    channel_dataset: xr.Dataset,
    params: Dict,
) -> Tuple[xr.DataArray, xr.DataArray]:
    print('params', params)

    rng = params.get("range_coord", "depth")
    ping_win = int(params.get("ping_window", 5))  # pings on each side
    thr = params.get("threshold", (10.0, 7.0))  # (thr0, thr1) or single
    excl_above = float(params.get("exclude_above", 250.0))
    ref_min_m = params.get("ref_min", None)  # meters
    ref_max_m = params.get("ref_max", None)  # meters
    jumps_m = float(params.get("jumps", 5.0))
    maxts = float(params.get("maxts", -35.0))

    # thresholds
    if isinstance(thr, (tuple, list)) and len(thr) == 2:
        thr0, thr1 = float(thr[0]), float(thr[1])
    else:
        thr0 = float(thr);
        thr1 = max(2.0, thr0 - 3.0)

    # ensure 2-D (T,Z) and reasonable chunks
    Sv_db_TZ = channel_dataset["Sv"].transpose("ping_time", rng)
    Sv_db_ZT = Sv_db_TZ.transpose(rng, "ping_time")
    # pick chunks >> n along time; leave depth chunked (or full column)
    Sv_db_ZT = Sv_db_ZT.chunk({"ping_time": max(1024, 4 * (2 * ping_win + 1)), rng: -1})

    r = channel_dataset[rng].values.astype(float)
    Z = r.size

    # default deep reference if not provided
    if ref_min_m is None or ref_max_m is None:
        max_r = float(r[-1])
        ref_min_m = max(excl_above + 50.0, min(150.0, max_r * 0.5))
        ref_max_m = min(max_r, ref_min_m + 200.0)

    up = _nearest_idx(r, float(ref_min_m))
    lw = _nearest_idx(r, float(ref_max_m)) + 1
    rmin = _nearest_idx(r, excl_above)
    # vertical step (samples) from meter jump
    # if vertical spacing varies slightly, nearest_idx still works fine.
    sf = max(1, _nearest_idx(r, float(jumps_m)) or 1)

    # run the kernel lazily with time overlap = n
    da_in = da.asarray(Sv_db_ZT.data)  # (Z, T)
    mask_da = da.map_overlap(
        _fielding_mask_kernel,
        da_in,
        depth={1: ping_win},  # overlap along time axis only
        boundary=np.nan,
        trim=True,
        meta=np.empty((0, 0), dtype=bool),  # dtype hint only
        up=up,
        lw=lw,
        rmin=rmin,
        sf=sf,
        n=ping_win,
        thr0=thr0,
        thr1=thr1,
        maxts=maxts,
    )

    # wrap mask back to xarray with original dims
    mask_T = xr.DataArray(mask_da, dims=(rng, "ping_time"),
                          coords={rng: Sv_db_ZT[rng], "ping_time": Sv_db_ZT["ping_time"]})
    mask_T = mask_T.transpose("ping_time", rng)

    # unfeasible mask: cheap, no overlap needed
    # edges in time, and deep layer fully NaN
    T = Sv_db_TZ.sizes["ping_time"]

    edge = xr.zeros_like(Sv_db_TZ.isel({rng: 0}), dtype=bool)  # (T,)
    if ping_win > 0:
        edge[:ping_win] = True
        edge[-ping_win:] = True

    deep = channel_dataset["Sv"].transpose(rng, "ping_time").isel({rng: slice(up, lw)})
    deep_any = xr.apply_ufunc(np.isfinite, deep, dask="parallelized", output_dtypes=[bool]).any(rng)
    deep_allnan = ~deep_any  # (T,)

    mask_U_1d = edge | deep_allnan
    mask_U = xr.broadcast(mask_U_1d, mask_T)[0]

    # outside the TN zone (shallower than excl_above) => False
    in_zone = channel_dataset[rng] >= excl_above
    mask_T = mask_T.where(in_zone, False)
    mask_U = mask_U.where(in_zone, False)

    return mask_T, mask_U


def _nanpct_keepdims(block: np.ndarray, *, q: float, axis=None):
    return np.nanpercentile(block, q, axis=axis, keepdims=True)


def rolling_nanpercentile(arr, *, q: float, axis=None):
    """
    Percentile reducer for xarray.rolling.reduce that works with NumPy and Dask.
    Drops the rolling axes so there are no hidden `_rolling_dim_*` left to align.
    """
    if isinstance(arr, da.Array):
        # normalize axis to a tuple of positive ints
        if axis is None:
            raise ValueError("rolling_nanpercentile requires 'axis' from xarray")
        if np.isscalar(axis):
            axis_tuple = (int(axis) % arr.ndim,)
        else:
            axis_tuple = tuple(int(a) % arr.ndim for a in axis)

        def _nanpct(x, q):
            with np.errstate(all="ignore"):
                return np.nanpercentile(x, q, axis=axis_tuple)  # keepdims=False

        return da.map_blocks(
            _nanpct,
            arr,
            q,
            dtype=arr.dtype,
            drop_axis=axis_tuple,
        )
    # NumPy path
    with np.errstate(all="ignore"):
        return np.nanpercentile(arr, q, axis=axis)  # keepdims=False


def transient_noise_mask_ryan(
    channel_dataset: xr.Dataset,
    params: Dict,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Ryan et al. (2015) transient‑noise filter for a single channel.

    Parameters
    ----------
    channel_dataset : Dataset with "Sv" (dB)
    params : dict
        range_coord            : vertical coordinate name (default "echo_range")
        ping_window            : half‑width (pings)      (default 5)
        range_window           : half‑width (samples)    (default 3)
        threshold              : dB above block statistic (default 6)
        exclude_above          : min range to apply (m)  (default 250)
        percentile             : percentile for block (default 15)
        min_pings / min_samples: override min_periods

    Returns
    -------
    mask_transient, mask_unfeasible : Boolean DataArrays
    """
    rng_var = params.get("range_coord", "echo_range")
    half_ping = params.get("ping_window", 5)
    half_range = params.get("range_window", 3)
    thr_db = params.get("threshold", 6.0)
    excl_above = params.get("exclude_above", 250.0)
    perc = params.get("percentile", 15)
    min_pings = params.get("min_pings", 2 * half_ping + 1)
    min_samples = params.get("min_samples", 2 * half_range + 1)

    Sv_db = channel_dataset["Sv"]
    range_values = channel_dataset[rng_var]
    ping_dim = "ping_time"
    range_dim = range_values.dims[0]

    block_ping = 2 * half_ping + 1
    block_range = 2 * half_range + 1
    min_periods = min_pings * min_samples

    # apply exclusion range mask
    in_zone = range_values >= excl_above

    # convert to linear & rolling block pXX
    Sv_lin = 10.0 ** (Sv_db / 10.0)

    rolled = Sv_lin.rolling(
        {ping_dim: block_ping, range_dim: block_range},
        center=True,
        min_periods=min_periods,
    )

    pct_func = partial(rolling_nanpercentile, q=perc)
    block_lin = rolled.reduce(pct_func, keep_attrs=True)

    # back to dB
    block_db = 10.0 * np.log10(block_lin)

    # compute masks
    diff_db = Sv_db - block_db
    mask_transient = (diff_db > thr_db) & in_zone
    mask_unfeasible = block_db.isnull() & in_zone

    mask_transient = mask_transient.where(in_zone, False)
    mask_unfeasible = mask_unfeasible.where(in_zone, False)

    return mask_transient, mask_unfeasible
