import numpy as np
import xarray as xr
from typing import Dict, Tuple


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

    channel_label = channel_ds["channel_label"].values.item()
    # print('channel_label:', channel_label)
    # print('bin_sz:', bin_sz)
    # print('cut_above:', cut_above)
    dz = channel_ds.depth.diff('depth').median().item()
    # print(f"Grid spacing: {dz:.3f} m")
    # print(f"bin_sz chosen: {bin_sz}")
    # print(f"Physical window: {bin_sz * dz:.2f} m")

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
    impulse_mask = xr.zeros_like(Sv_sm_db, dtype=bool)
    for lag in lags:
        fwd = Sv_sm_db - Sv_sm_db.shift({ping_dim: -lag}, fill_value=-np.inf)
        bwd = Sv_sm_db - Sv_sm_db.shift({ping_dim:  lag}, fill_value=-np.inf)
        impulse_mask |= (fwd > thr_db) & (bwd > thr_db)

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

    # 6. final tidy-up & return
    impulse_mask = impulse_mask & ~mask_unfeasible  # edges/NaNs not impulses

    return impulse_mask, mask_unfeasible
