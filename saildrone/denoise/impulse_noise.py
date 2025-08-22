import numpy as np
import xarray as xr
from typing import Dict, Tuple


def _dilate_mask_shift_or(mask: xr.DataArray, *, pings: int = 0, samples: int = 0) -> xr.DataArray:
    """
    Very cheap rectangular dilation: OR a few shifted copies.
    Separable (time then depth) → O(pings + samples) shifts, not (pings*samples).
    Stays fully Dask-lazy.
    """
    if (pings <= 0) and (samples <= 0):
        return mask

    ping_dim, range_dim = "ping_time", mask.dims[1]
    out = mask

    # time dilation (±pings)
    if pings > 0:
        acc = out.data  # dask array
        for dt in range(1, pings + 1):
            acc = da.logical_or(acc, out.shift({ping_dim: dt},  fill_value=False).data)
            acc = da.logical_or(acc, out.shift({ping_dim: -dt}, fill_value=False).data)
        out = out.copy(data=acc)

    # depth dilation (±samples)
    if samples > 0:
        acc = out.data
        for dz in range(1, samples + 1):
            acc = da.logical_or(acc, out.shift({range_dim: dz},  fill_value=False).data)
            acc = da.logical_or(acc, out.shift({range_dim: -dz}, fill_value=False).data)
        out = out.copy(data=acc)

    return out


def impulsive_noise_mask(
    channel_ds: xr.Dataset,
    params: Dict[str, float],
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Multi-lag Ryan-style impulsive-noise detector (vectorised).

    Parameters
    ----------
    channel_ds : xr.Dataset
        Single-channel slice of the full dataset; must contain "Sv" (dB).
    params : dict with keys
        range_coord          : str   – vertical coordinate name ("echo_range"/"depth")
        vertical_bin_size    : int   – # samples in vertical mean (≥1, default 1 → none)
        ping_lags            : tuple – side-ping offsets, e.g. (1, 2, 3)
        threshold_db         : float – Sv diff threshold (dB, default 10)
        exclude_shallow_above: float – optional range cut-off (m) to skip processing

    Returns
    -------
    mask_impulse     : True where impulse noise detected
    mask_unfeasible  : True for pings where comparison impossible (edges or NaNs)
    """

    # 1. unpack & validate
    range_coord = params.get("range_coord", "echo_range")
    bin_cfg = params.get("vertical_bin_size", '2m')
    lags = tuple(sorted(set(params.get("ping_lags", (1,)))))
    thr_db = params.get("threshold_db", 10.0)
    cut_above = params.get("exclude_shallow_above", None)
    vote_k = params.get("vote_k_of_n", None)  # e.g., 2 means ">= 2 lags must vote True"
    post = params.get("post_dilate", None)  # e.g., {"pings": 1, "samples": 2}

    if any(l < 1 for l in lags):
        raise ValueError("ping_lags must contain positive integers")

    Sv_db = channel_ds["Sv"]
    range_values = channel_ds[range_coord]  # coordinate
    ping_dim = "ping_time"
    range_dim = range_values.dims[0]

    if isinstance(bin_cfg, (float, str)):
        s = str(bin_cfg).strip()
        if not s.isdigit() or s.lower().endswith("m"):
            # interpret as metres
            window_m = float(s.rstrip("mM"))
            dz = channel_ds[range_coord].diff(range_dim).median().item()
            bin_sz = max(1, int(round(window_m / abs(dz))))
        else:
            bin_sz = max(1, int(s))
    else:
        bin_sz = max(1, int(bin_cfg))

    # 2. vertical pooling in linear domain (optional)
    if np.all(np.diff(range_values) < 0):  # descending grid → sort ascending
        Sv_db = Sv_db.sortby(range_coord)
        range_values = range_values.sortby(range_coord)

    Sv_lin = 10.0 ** (Sv_db / 10.0)

    if bin_sz > 1:
        Sv_lin = (
            Sv_lin
            .coarsen({range_dim: bin_sz}, boundary="trim")
            .mean(skipna=True)
            .interp({range_dim: range_values}, method="nearest")
        )
    Sv_sm_db = 10.0 * np.log10(Sv_lin)

    # 3. multi-lag forward & backward differences
    count = xr.zeros_like(Sv_sm_db, dtype="uint8")
    for lag in lags:
        fwd = Sv_sm_db - Sv_sm_db.shift({ping_dim: -lag}, fill_value=-np.inf)
        bwd = Sv_sm_db - Sv_sm_db.shift({ping_dim: lag}, fill_value=-np.inf)
        hit = ((fwd > thr_db) & (bwd > thr_db)).astype("uint8")
        # add using dask arrays to avoid alignment overhead
        count = count.copy(data=(count.data + hit.data))

    if vote_k is None or int(vote_k) <= 1:
        impulse_mask = count > 0
    else:
        impulse_mask = count >= np.uint8(int(vote_k))

    # 4. unfeasible mask (first/last max(lag) pings or NaNs)
    max_lag = max(lags)
    n_pings = Sv_sm_db[ping_dim].size
    edge_vec = np.zeros(n_pings, dtype=bool)
    edge_vec[:max_lag] = True
    edge_vec[-max_lag:] = True

    mask_edges = xr.DataArray(
        edge_vec, coords={ping_dim: Sv_sm_db[ping_dim]}, dims=ping_dim
    ).broadcast_like(Sv_sm_db)

    mask_nan = Sv_sm_db.isnull()
    mask_unfeasible = mask_edges | mask_nan

    # 5. optional shallow exclusion
    if cut_above is not None:
        valid_depth = range_values >= cut_above
        mask_unfeasible |= ~valid_depth
        impulse_mask = impulse_mask.where(valid_depth, False)
    else:
        valid_depth = xr.ones_like(Sv_sm_db, dtype=bool)

    # 6. final tidy-up & return
    impulse_mask = impulse_mask & ~mask_unfeasible  # edges/NaNs not impulses

    if isinstance(post, dict):
        pd_p = int(post.get("pings", 0))
        pd_s = int(post.get("samples", 0))
        if (pd_p > 0) or (pd_s > 0):
            impulse_mask = _dilate_mask_shift_or(impulse_mask, pings=pd_p, samples=pd_s)
            # re-guard
            impulse_mask = impulse_mask & ~mask_unfeasible & valid_depth

    return impulse_mask, mask_unfeasible
