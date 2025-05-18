import numpy as np
import xarray as xr


def remove_background_noise(
    ds_Sv: xr.Dataset,
    ping_num: int,
    range_sample_num: int,
    background_noise_max: float = None,
    SNR_threshold: float = 3.0
) -> xr.Dataset:
    """
    Remove noise by using estimates of background noise
    from mean calibrated power of a collection of pings.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        dataset containing ``Sv`` and ``echo_range`` [m]
    ping_num : int
        Number of pings to obtain noise estimates.
    range_sample_num : int
        Number of samples along the ``range_sample`` dimension to obtain noise estimates.
    background_noise_max : str, default None
        The upper limit for background noise expected under the operating conditions.
    SNR_threshold : str, default "3.0dB"
        Acceptable signal-to-noise ratio, default to 3 dB.

    Returns
    -------
    The input dataset with background noise removed.

    Notes
    -----
    This function's implementation is based on the following text reference:

        De Robertis & Higginbottom. 2007.
        A post-processing technique to estimate the signal-to-noise ratio
        and remove echosounder background noise.
        ICES Journal of Marine Sciences 64(6): 1282–1291.
    """

    # Step 1: Compute Sv_noise
    Sv_noise = estimate_background_noise(
        ds_Sv, ping_num, range_sample_num, background_noise_max=background_noise_max
    )

    ds_Sv = ds_Sv.assign(sound_absorption=0.001)

    # Step 2: Convert Sv and noise to linear scale
    linear_Sv = 10 ** (ds_Sv["Sv"] / 10)
    linear_noise = 10 ** (Sv_noise / 10)

    # Step 3: Subtract noise and avoid negative/near-zero values
    linear_corrected_Sv = linear_Sv - linear_noise
    linear_corrected_Sv = linear_corrected_Sv.where(linear_corrected_Sv > 1e-12, other=np.nan)

    # Step 4: Convert back to logarithmic scale
    corrected_Sv = 10 * np.log10(linear_corrected_Sv)

    # Step 5: Apply SNR threshold
    SNR_mask = corrected_Sv - Sv_noise > SNR_threshold
    corrected_Sv = corrected_Sv.where(SNR_mask, other=np.nan)

    # Step 7: Update dataset
    ds_Sv["Sv"] = corrected_Sv

    return ds_Sv


def estimate_background_noise(
    ds_Sv: xr.Dataset,
    ping_num: int,
    depth_bin_size: int,
    background_noise_max: float = None,
    depth_var: str = "echo_range"
) -> xr.DataArray:
    """
    Estimate background noise using a rolling window approach based on depth and pings.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        Must contain
          * ``Sv``            – volume back-scattering strength [dB re 1 m-1],
          * ``sound_absorption`` – attenuation coefficient α [dB m-1],
          * a 1-D coordinate named *depth_var* (default ``"echo_range"``).
    ping_num : int
        Number of pings per horizontal bin (rolling window along *ping_time*).
    depth_bin_size : float
        Vertical bin size (metres) used for noise statistics.
    background_noise_max : float, default None
        Maximum allowable background noise level (in dB). Values above this threshold will be capped.
    depth_var : str, default "echo_range"
        Name of the vertical coordinate to use.

    Returns
    -------
    Sv_noise : xr.DataArray
        Estimated noise level [dB] with the same dimensions as ``ds_Sv["Sv"]``.

    Notes
    -----
    Based on De Robertis and Higginbottom (2007), ICES Journal of Marine Sciences.
    """
    if "Sv" not in ds_Sv:
        raise ValueError("`ds_Sv` must contain a variable named 'Sv' (dB).")
    if depth_var not in ds_Sv.coords:
        raise ValueError(f"Vertical coordinate '{depth_var}' not found.")
    if "sound_absorption" not in ds_Sv:
        raise ValueError("Dataset must contain 'sound_absorption' (dB m-1).")

    # Transmission loss calculations
    rng = ds_Sv[depth_var]
    rng_clip = xr.where(rng < 1.0, 1.0, rng)  # avoid log(0)
    tl = 20.0 * np.log10(rng_clip) + 2.0 * ds_Sv["sound_absorption"] * rng

    # Sv → linear, compensate for transmission loss
    p_lin = 10.0 ** ((ds_Sv["Sv"] - tl) / 10.0)

    # bin vertically (depth bins)
    depth_edges = np.arange(float(rng.min()),
                            float(rng.max()) + depth_bin_size,
                            depth_bin_size)

    p_lin_binned = p_lin.groupby_bins(
        depth_var, bins=depth_edges
    ).mean(skipna=True)

    # Name of the new pseudo-dimension produced by groupby_bins:
    depth_bin_dim = f"{depth_var}_bins"

    # coarsen horizontally (ping_time axis)
    p_lin_coarse = p_lin_binned.coarsen(
        ping_time=ping_num, boundary="trim"
    ).mean()

    # noise floor = min over depth bins
    noise_lin = p_lin_coarse.min(dim=depth_bin_dim, skipna=True)

    # optional upper-bound clipping
    if background_noise_max is not None:
        noise_lin = noise_lin.clip(max=10.0 ** (background_noise_max / 10.0))

    #  back to full dimensions
    # 1. interpolate back to every ping
    noise_ping = noise_lin.interp(ping_time=ds_Sv.ping_time, method="nearest")

    # 2. broadcast to (channel, ping_time, depth) without materialising data
    noise_full, _ = xr.broadcast(noise_ping, ds_Sv["Sv"])

    # add TL, dB back
    Sv_noise = 10.0 * np.log10(noise_full) + tl
    Sv_noise = Sv_noise.assign_attrs(
        {
            "long_name": "Estimated background noise level",
            "units": "dB re 1 m-1",
            "comment": (
                "Computed with De Robertis & Higginbottom (2007) method: "
                f"{ping_num}-ping × {depth_bin_size} m rolling window."
            ),
        }
    )

    return Sv_noise
