import dask.array as da
import numpy as np
import xarray as xr
from typing import Tuple
from echopype.clean.utils import extract_dB


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
    background_noise_max = params.get("background_noise_max", "-125.0dB")
    SNR_threshold = params.get("SNR_threshold", "3.0dB")
    minimal_linear = params.get("minimal_linear", 1e-30)
    background_noise_max = extract_dB(background_noise_max)
    SNR_threshold = extract_dB(SNR_threshold)
    depth_stat = params.get("depth_stat", "quantile")  # "min" | "quantile"
    depth_quantile = float(params.get("depth_quantile", 0.15))  # used when depth_stat="quantile"

    # optional guard so the depth statistic ignores DSL
    guard_mode = params.get("guard_mode")  # None | "above" | "outside_band"
    guard_depth = params.get("guard_depth")  # meters if mode=="above"
    guard_band = params.get("guard_band")  # [z0, z1] if mode=="outside_band"

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
    power_lin = 10.0 ** (Sv_flat_db / 10.0)  # TVG-removed power
    binned_lin = power_lin.coarsen(
        ping_time=ping_win,
        **{rng_var: rng_win},
        boundary="pad",
    ).mean()  # dask-aware

    # convert to dB for taking a depth statistic
    binned_db = 10.0 * np.log10(binned_lin.where(binned_lin > 0))

    binned_db = binned_db.chunk({rng_var: -1})

    # optional guard: exclude DSL from the depth statistic
    if guard_mode:
        z = ds_channel[rng_var]
        if guard_mode == "above":
            if guard_depth is None:
                raise ValueError("guard_depth required when guard_mode='above'")
            region = z <= float(guard_depth)
        elif guard_mode == "outside_band":
            if not guard_band or len(guard_band) != 2:
                raise ValueError("guard_band=[z0,z1] required when guard_mode='outside_band'")
            z0, z1 = sorted(map(float, guard_band))
            region = (z < z0) | (z > z1)
        else:
            raise ValueError("guard_mode must be None|'above'|'outside_band'")
        binned_db = binned_db.where(region)

    # 5. back to dB ‐ cap floor
    if depth_stat == "min":
        noise_1d_db = binned_db.min(dim=rng_var, skipna=True)
    elif depth_stat == "quantile":
        noise_1d_db = binned_db.quantile(depth_quantile, dim=rng_var, skipna=True)
        # squeeze the helper 'quantile' dim if present
        if "quantile" in noise_1d_db.dims:
            noise_1d_db = noise_1d_db.squeeze("quantile", drop=True)
    else:
        raise ValueError("depth_stat must be 'min' or 'quantile'")

    # align ping_time indices to the **first** ping of each coarsened bin (like echopype)
    noise_1d_db = noise_1d_db.assign_coords(ping_time=ping_win * np.arange(noise_1d_db.sizes["ping_time"]))
    power_lin = power_lin.assign_coords(ping_time=np.arange(power_lin.sizes["ping_time"]))

    # optional cap (LESS negative => more aggressive; MORE negative => gentler)
    if background_noise_max is not None:
        noise_1d_db = noise_1d_db.where(noise_1d_db < background_noise_max, background_noise_max)

    # 6. restore TVG
    Sv_noise_db = (
            noise_1d_db
            .reindex({"ping_time": power_lin["ping_time"]}, method="ffill")
            .assign_coords(ping_time=ds_channel["ping_time"])
            + tvg
    )

    # 7. compute masks
    Sv_lin_tot = 10.0 ** (Sv / 10.0)
    bgn_lin_tot = 10.0 ** (Sv_noise_db / 10.0)
    lin_diff = Sv_lin_tot - bgn_lin_tot

    mask_non_positive = lin_diff <= 0
    Sv_clean_db = xr.where(mask_non_positive, np.nan, 10.0 * np.log10(lin_diff))
    snr_db = Sv_clean_db - Sv_noise_db
    mask_low_snr = snr_db < SNR_threshold

    return mask_low_snr, mask_non_positive
