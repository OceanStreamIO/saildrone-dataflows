import dask.array as da
import numpy as np
import xarray as xr
from typing import Tuple


def background_noise_mask(
    ds_channel: xr.Dataset,
    params: dict,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    De Robertis & Higginbottom (2007) background‐noise filter for one channel.

    Parameters
    ----------
    ds_channel : xr.Dataset
        Single‐channel slice containing "Sv" (dB) and the vertical coordinate.
    params : dict

    Dictionary of parameters (all optional; defaults shown):
      • range_coord (str): vertical coord name, e.g. "echo_range"       [ "echo_range" ]
      • sound_absorption (float): sound absorption α (dB m⁻¹)  [ 0.001 ]
      • range_window (int): # of range‐samples in blocking window       [ 20 ]
      • ping_window (int): # of pings in blocking window                [ 50 ]
      • background_noise_max (float): lowest allowed background Sv      [ -125.0 ]
      • SNR_threshold (float): minimum SNR (dB)                         [ 3.0 ]
      • minimal_linear (float): floor for linear power before log10     [ 1e-30 ]

    Returns
    -------
    mask_low_snr     : xr.DataArray[bool]
        True where (Sv_clean − background) < snr_threshold_db.
    mask_non_positive: xr.DataArray[bool]
        True where (linear Sv − linear background) ≤ 0.
    """

    # 1. unpack with defaults
    rng_var = params.get("range_coord", "echo_range")
    sound_absorption = params.get("sound_absorption", 0.001)
    rng_win = params.get("range_window", 20)
    ping_win = params.get("ping_window", 50)
    background_noise_max = params.get("background_noise_max", -125.0)
    SNR_threshold = params.get("SNR_threshold", 3.0)
    minimal_linear = params.get("minimal_linear", 1e-30)

    # 2. extract Sv & range
    if rng_win == 'auto' and rng_var == 'depth':
        # auto-detect range window based on ping window
        dz = float(ds_channel.depth.diff('depth').median())
        rng_win = max(1, round(1.0 / dz))

    ds_channel = ds_channel.assign(sound_absorption=sound_absorption)
    Sv = ds_channel["Sv"]

    range_values = ds_channel[rng_var]

    # 3. remove TVG: 20 log10(r) + 2 α r
    r_safe = xr.where(range_values > 0, range_values, np.nan)
    tvg = 20.0 * np.log10(r_safe) + 2.0 * sound_absorption * r_safe
    Sv_flat_db = Sv - tvg

    # 4. linear domain for block‐min
    Sv_lin = 10.0 ** (Sv_flat_db / 10.0)
    block_min_lin = (
        Sv_lin
        .coarsen(
            ping_time=ping_win,
            **{rng_var: rng_win},
            boundary="trim"
        )
        .reduce(da.nanmin if Sv_lin.chunks else np.nanmin)
        .broadcast_like(Sv_lin)
    )

    # 5. back to dB ‐ cap floor
    bgn_flat_db = 10.0 * np.log10(np.maximum(block_min_lin, minimal_linear))
    bgn_flat_db = bgn_flat_db.where(bgn_flat_db > background_noise_max, background_noise_max)

    # 6. restore TVG
    background_db = bgn_flat_db + tvg

    # 7. compute masks
    Sv_lin_tot = 10.0 ** (Sv / 10.0)
    bgn_lin_tot = 10.0 ** (background_db / 10.0)
    lin_diff = Sv_lin_tot - bgn_lin_tot

    mask_non_positive = lin_diff <= 0

    Sv_clean_db = xr.where(
        mask_non_positive,
        np.nan,
        10.0 * np.log10(np.maximum(lin_diff, minimal_linear))
    )

    snr_db = Sv_clean_db - background_db
    mask_low_snr = snr_db < SNR_threshold

    return mask_low_snr, mask_non_positive
