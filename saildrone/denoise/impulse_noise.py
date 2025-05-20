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
    if desired_channel is None:
        if desired_frequency is None:
            raise ValueError("Must specify either desired channel or desired frequency")
        else:
            desired_channel = frequency_nominal_to_channel(source_Sv, desired_frequency)

    impulse_mask = impulse_noise_mask(source_Sv, desired_channel, parameters=parameters)
    noise_free_mask = ~impulse_mask

    return noise_free_mask


def impulse_noise_mask(
    Sv_ds: xr.Dataset,
    desired_channel: str,
    depth_dim="range_sample",
    parameters: dict = None
) -> xr.DataArray:
    """
    Ryan et al. (2015) Impulse-Noise filter implemented for lazy Dask arrays.

    Parameters
    ----------
    Sv_ds : xr.Dataset
        Multi-frequency dataset containing `Sv` in dB (dims: ping_time, depth, channel).
    desired_channel : str
        Channel (frequency) to process.
    depth_dim : str
        Depth dimension name (default: "range_sample").
    parameters : dict
        Dictionary of parameters:
            - depth_bin (int): Depth bin size for smoothing.
            - num_side_pings (int): Number of side pings for comparison.
            - impulse_noise_threshold (float): Threshold for impulse noise detection.

    Returns
    -------
    xr.DataArray (bool)
        True where data should be masked (original NaNs **or** detected IN).
    """
    parameter_names = ("depth_bin", "num_side_pings", "impulse_noise_threshold")
    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError("Missing parameters - should be depth_bin, num_side_pings, impulse_noise_threshold, are" + str(parameters.keys()))

    depth_bin, n, thr = (
        parameters["depth_bin"],
        parameters["num_side_pings"],
        parameters["impulse_noise_threshold"],
    )

    if depth_bin < 1 or n < 1 or thr <= 0:
        raise ValueError("depth_bin and num_side_pings must be ≥1; threshold > 0")

    # 1. Select channel and extract Sv (dB); drop singleton dim to keep array 2-D
    Sv = Sv_ds.sel(channel=desired_channel).Sv

    # Original invalid values
    invalid = Sv.isnull()

    # Ensure depth coordinate is ascending for interp
    if np.all(np.diff(Sv[depth_dim]) < 0):
        Sv = Sv.sortby(depth_dim)

    # 2. vertical block smoothing in the *linear* domain
    Sv_lin = 10 ** (Sv / 10)  # dB ➜ linear
    Sv_lin_blk = (
        Sv_lin.coarsen({depth_dim: depth_bin}, boundary="pad")
        .mean(skipna=True)
        .interp({depth_dim: Sv[depth_dim]}, method="nearest")  # restore grid
    )
    Sv_sm = 10 * np.log10(Sv_lin_blk)  # back to dB

    # 3.two-sided comparison along ping_time
    fwd = Sv_sm - Sv_sm.shift(ping_time=+n, fill_value=-np.inf)
    bwd = Sv_sm - Sv_sm.shift(ping_time=-n, fill_value=-np.inf)
    mask_in = (fwd > thr) & (bwd > thr)

    # 4. edge pings cannot be evaluated
    N = Sv.coords["ping_time"].size
    edge_vec = np.zeros(N, dtype=bool)
    edge_vec[:n] = True
    edge_vec[-n:] = True
    edge_mask = xr.DataArray(
        edge_vec,
        coords={"ping_time": Sv.ping_time},
        dims="ping_time",
    ).broadcast_like(mask_in)

    # 5. Combine and return
    return invalid | mask_in | edge_mask

