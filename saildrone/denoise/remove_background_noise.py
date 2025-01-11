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
    Sv_noise = Sv_noise.clip(min=-120, max=-60)

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
    ds_Sv: xr.Dataset, ping_num: int, depth_bin_size: int, background_noise_max: float = None, depth_var: str = "depth"
) -> xr.DataArray:
    """
    Estimate background noise using a rolling window approach based on depth and pings.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        Dataset containing `Sv` (in dB) and `depth` [m].
    ping_num : int
        Number of pings to consider in each bin for noise estimation.
    depth_bin_size : float
        Depth bin size (in meters) for grouping data during noise estimation.
    background_noise_max : float, default None
        Maximum allowable background noise level (in dB). Values above this threshold will be capped.
    depth_var : str, default "depth"
        Variable representing the depth in the dataset.

    Returns
    -------
    Sv_noise : xr.DataArray
        DataArray with the estimated background noise level.

    Notes
    -----
    Based on De Robertis and Higginbottom (2007), ICES Journal of Marine Sciences.
    """
    # Transmission loss calculations
    spreading_loss = 20 * np.log10(ds_Sv[depth_var].clip(min=1))  # Avoid log(0)
    absorption_loss = 2 * ds_Sv["sound_absorption"] * ds_Sv[depth_var]
    transmission_loss = spreading_loss + absorption_loss

    # Convert Sv to linear domain and subtract transmission loss
    power_cal = 10 ** ((ds_Sv["Sv"] - transmission_loss) / 10)

    # Bin data by depth
    depth_bins = np.arange(
        ds_Sv[depth_var].min().item(), ds_Sv[depth_var].max().item() + depth_bin_size, depth_bin_size
    )
    power_cal_binned_depth = power_cal.groupby_bins(
        depth_var, bins=depth_bins, labels=np.arange(len(depth_bins) - 1)
    ).mean(dim=depth_var)

    # Coarsen along ping_time dimension
    power_cal_binned = power_cal_binned_depth.coarsen(
        ping_time=ping_num, boundary="trim"
    ).mean()

    # Compute noise as minimum binned power along the depth bins
    noise = power_cal_binned.min(dim="depth_bins", skipna=True)

    # Limit noise by maximum allowable background noise
    if background_noise_max is not None:
        noise = noise.clip(max=10 ** (background_noise_max / 10))

    # Upsample noise back to original ping_time and depth dimensions
    noise_resampled = noise.interp(ping_time=ds_Sv["ping_time"], method="nearest")
    noise_2d = noise_resampled.expand_dims(dim={depth_var: ds_Sv[depth_var]}).broadcast_like(ds_Sv["Sv"])

    # Add transmission loss back to noise
    Sv_noise = 10 * np.log10(noise_2d) + transmission_loss

    return Sv_noise


def _log2lin(data):
    return 10 ** (data / 10)


def _lin2log(data):
    return 10 * np.log10(data)