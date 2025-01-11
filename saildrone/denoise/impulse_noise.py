import numpy as np
import xarray as xr

from saildrone.utils.mask_transformation import downsample, upsample
from saildrone.utils import frequency_nominal_to_channel

RYAN_DEFAULT_PARAMS = {"thr": 10, "m": 5, "n": 1}
RYAN_ITERABLE_DEFAULT_PARAMS = {"thr": 10, "m": 5, "n": (1, 2)}
WANG_DEFAULT_PARAMS = {
    "thr": (-70, -40),
    "erode": [(3, 3)],
    "dilate": [(5, 5), (7, 7)],
    "median": [(7, 7)],
}


def get_impulse_noise_mask(
    source_Sv,
    parameters,
    desired_channel=None,
    desired_frequency=None,
    method="ryan",
) -> xr.DataArray:
    mask_map = {
        "ryan": _ryan
        # "wang": impulse_noise._wang,
    }

    if method not in mask_map.keys():
        raise ValueError(f"Unsupported method: {method}")

    if desired_channel is None:
        if desired_frequency is None:
            raise ValueError("Must specify either desired channel or desired frequency")
        else:
            desired_channel = frequency_nominal_to_channel(source_Sv, desired_frequency)

    impulse_mask = mask_map[method](source_Sv, desired_channel, parameters)
    noise_free_mask = ~impulse_mask

    return noise_free_mask


def _ryan(
    Sv_ds: xr.Dataset,
    desired_channel: str,
    parameters: dict,
) -> xr.DataArray:
    """
    Mask impulse noise following the two-sided comparison method described in:
    Ryan et al. (2015) ‘Reducing bias due to noise and attenuation in
    open-ocean echo integration data’, ICES Journal of Marine Science, 72: 2482–2493.

    Parameters
    ----------
        Sv_ds (xarray.Dataset): xr.DataArray with Sv data for multiple channels (dB).
        desired_channel (str): Name of the desired frequency channel.
        parameters (dict): Dictionary of parameters. Must contain the following:
            depth_bin (int/float): Vertical binning length (n samples or range).
            num_side_pings (int): Number of pings either side for comparisons.
            impulse_noise_threshold (int/float): User-defined threshold value (dB).

    Returns
    -------
        xarray.DataArray: xr.DataArray with IN mask.

    Notes
    -----
    In the original 'ryan' function (echopy), two masks are returned:
        - 'mask', where True values represent likely impulse noise, and
        - 'mask_', where True values represent valid samples for side comparison.

    When adapting for echopype, we must ensure the mask aligns with our data orientation.
    Hence, we transpose 'mask' and 'mask_' to match the shape of the data in 'Sv_ds'.

    Then, we create a combined mask using a bitwise AND operation between 'mask' and '~mask_'.

    """
    parameter_names = ("depth_bin", "num_side_pings", "impulse_noise_threshold")
    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError("Missing parameters - should be depth_bin, num_side_pings, impulse_noise_threshold, are" + str(parameters.keys()))

    m = parameters["depth_bin"]
    n = parameters["num_side_pings"]
    thr = parameters["impulse_noise_threshold"]

    # Select the desired frequency channel directly using 'sel'
    selected_channel_ds = Sv_ds.sel(channel=desired_channel)

    Sv = selected_channel_ds.Sv
    Sv_ = downsample(Sv, coordinates={"depth": m}, is_log=True)
    Sv_ = upsample(Sv_, Sv)

    # get valid sample mask
    mask = Sv_.isnull()

    # get IN mask
    forward = Sv_ - Sv_.shift(shifts={"ping_time": n}, fill_value=np.nan)
    backward = Sv_ - Sv_.shift(shifts={"ping_time": -n}, fill_value=np.nan)
    forward = forward.fillna(np.inf)
    backward = backward.fillna(np.inf)
    mask_in = (forward > thr) & (backward > thr)
    # add to the mask areas that have had data shifted out of range
    mask_in[0:n, :] = True
    mask_in[-n:, :] = True

    mask = mask | mask_in
    mask = mask.drop("channel")
    return mask

