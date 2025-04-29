from functools import partial

import numpy as np
import xarray as xr

from saildrone.utils import frequency_nominal_to_channel


DEFAULT_RYAN_PARAMS = {
    "r0": 180,
    "r1": 280,
    "n": 30,
    "thr": -6,
    "start": 0,
    "chunks": {"ping_time": 100, "range_sample": 100},
}
DEFAULT_ARIZA_PARAMS = {"offset": 20, "thr": (-40, -35), "m": 20, "n": 50}


def get_attenuation_mask(
    ds,
    parameters,
    desired_channel=None,
    desired_frequency=None,
    method="ryan",
    chunks=None,
) -> xr.DataArray:
    mask_map = {
        "ryan": _ryan
    }
    if method not in mask_map.keys():
        raise ValueError(f"Unsupported method: {method}")

    if desired_channel is None:
        if desired_frequency is None:
            raise ValueError("Must specify either desired channel or desired frequency")
        else:
            desired_channel = frequency_nominal_to_channel(ds, desired_frequency)

    mask = mask_map[method](ds, desired_channel, parameters, chunks=chunks)
    return mask


def _ryan(ds: xr.DataArray, desired_channel: str, parameters=None, chunks=None):
    parameter_names = ("upper_limit_sl", "lower_limit_sl", "num_side_pings", "attenuation_signal_threshold")

    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError(
            "Missing parameters - should be upper_limit_sl, lower_limit_sl, num_side_pings, "
            "attenuation_signal_threshold are"
            + str(parameters.keys())
        )
    upper_limit_sl = parameters["upper_limit_sl"]
    lower_limit_sl = parameters["lower_limit_sl"]
    num_side_pings = parameters["num_side_pings"]
    attenuation_signal_threshold = parameters["attenuation_signal_threshold"]
    range_var = parameters["range_var"]

    channel_Sv = ds.sel(channel=desired_channel)
    Sv = channel_Sv["Sv"]
    range_values = channel_Sv[range_var]

    # Broadcast range_values to match Sv dimensions
    range_values = range_values.broadcast_like(Sv)

    if (upper_limit_sl > Sv[range_var].max()) or (lower_limit_sl < Sv[range_var].min()):
        return xr.zeros_like(Sv, dtype=bool)

    # Partial function for single-channel mask computation
    partial_echopy_attenuation_mask = partial(
        echopy_attenuated_signal_mask,
        upper_limit_sl=upper_limit_sl,
        lower_limit_sl=lower_limit_sl,
        num_side_pings=num_side_pings,
        attenuation_signal_threshold=attenuation_signal_threshold,
    )

    # Apply the function using xr.apply_ufunc
    attenuation_mask = xr.apply_ufunc(
        partial_echopy_attenuation_mask,
        Sv,
        range_values,
        input_core_dims=[["ping_time", range_var], ["ping_time", range_var]],
        output_core_dims=[["ping_time", range_var]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[bool],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    return attenuation_mask


def echopy_attenuated_signal_mask(
    Sv: np.ndarray,
    range_var: np.ndarray,
    upper_limit_sl: float,
    lower_limit_sl: float,
    num_side_pings: int,
    attenuation_signal_threshold: float,
) -> np.ndarray:
    """Single-channel attenuated signal mask computation from echopy."""
    attenuated_mask = np.zeros(Sv.shape, dtype=bool)

    for ping_time_idx in range(Sv.shape[0]):

        # Find indices for upper and lower SL limits
        up = np.argmin(abs(range_var[ping_time_idx, :] - upper_limit_sl))
        lw = np.argmin(abs(range_var[ping_time_idx, :] - lower_limit_sl))

        # if lw <= up or np.all(np.isnan(Sv[ping_time_idx, up:lw])):
        #     continue
        #
        # # Dynamically adjust block size for edge cases
        # block_start = max(0, ping_time_idx - num_side_pings)
        # block_end = min(Sv.shape[0], ping_time_idx + num_side_pings + 1)
        #
        # # Compute ping median and block median
        # pingmedian = _lin2log(np.nanmedian(_log2lin(Sv[ping_time_idx, up:lw])))
        # blockmedian = _lin2log(np.nanmedian(_log2lin(Sv[block_start:block_end, up:lw])))
        #
        # if (pingmedian - blockmedian) < attenuation_signal_threshold:
        #     attenuated_mask[ping_time_idx, :] = True

        # Mask when attenuation masking is feasible
        if not ((ping_time_idx - num_side_pings < 0) | (ping_time_idx + num_side_pings > Sv.shape[0] - 1) | np.all(np.isnan(Sv[ping_time_idx, up:lw]))):
            # Compare ping and block medians, and mask ping if difference greater than
            # threshold.
            pingmedian = _lin2log(np.nanmedian(_log2lin(Sv[ping_time_idx, up:lw])))
            blockmedian = _lin2log(np.nanmedian(_log2lin(
                        Sv[
                            (ping_time_idx - num_side_pings) : (ping_time_idx + num_side_pings),
                            up:lw,
                        ]
                    )))

            if (pingmedian - blockmedian) < attenuation_signal_threshold:
                attenuated_mask[ping_time_idx, :] = True

    return attenuated_mask


def _log2lin(data):
    return 10 ** (data / 10)


def _lin2log(data):
    return 10 * np.log10(data)
