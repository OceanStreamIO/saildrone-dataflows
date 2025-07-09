import traceback
import dask.array as da
import numpy as np
import xarray as xr


DEFAULT_RYAN_PARAMS = {
    "r0": 180,
    "r1": 280,
    "n": 30,
    "thr": -6,
    "start": 0,
    "chunks": {"ping_time": 100, "range_sample": 100},
}
PARAMETER_NAMES = ("upper_limit_sl", "lower_limit_sl", "num_side_pings", "threshold")


def attenuation_mask(
    ds_channel: xr.Dataset,
    parameters: dict
) -> xr.DataArray:
    if not all(name in parameters.keys() for name in PARAMETER_NAMES):
        raise ValueError(
            "Missing parameters – should be: " + str(PARAMETER_NAMES) + ", but the provided are: " + str(parameters.keys())
        )

    upper_limit_sl = parameters["upper_limit_sl"]
    lower_limit_sl = parameters["lower_limit_sl"]
    num_side_pings = parameters["num_side_pings"]
    attenuation_signal_threshold = parameters["threshold"]
    range_var = parameters["range_coord"]
    Sv = ds_channel["Sv"]
    range_values = ds_channel[range_var]
    vertical_dim = range_values.dims[0]

    if (upper_limit_sl > range_values.max()) or (lower_limit_sl < range_values.min()):
        empty = xr.zeros_like(Sv, dtype=bool)
        return empty, empty

    try:
        in_layer = (range_values >= upper_limit_sl) & (range_values <= lower_limit_sl)
        ping_median = Sv.where(in_layer).median(dim=vertical_dim, skipna=True)

        # rolling/block median across pings (centre-aligned)
        block_width = 2 * num_side_pings + 1

        reducer = da.nanmedian if Sv.chunks else np.nanmedian

        # xarray’s .rolling keeps Dask laziness:
        block_median = (
            ping_median.rolling(
                ping_time=block_width, center=True, min_periods=block_width
            )
            .reduce(reducer)
        )

        # difference in dB between ping & block medians
        diff_db = ping_median - block_median

        # flag entire ping when diff < threshold
        ping_flag = diff_db < -abs(attenuation_signal_threshold)

        # pings where the comparison could not be performed
        ping_unfeasible = ping_median.isnull() | block_median.isnull()

        # Broadcast to full cube
        mask_as = (ping_flag | ping_unfeasible).broadcast_like(Sv)
        mask_failed = ping_unfeasible.broadcast_like(Sv)

        return mask_as, mask_failed
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Error computing attenuated signal mask: {e}")


def _log2lin(data):
    return 10 ** (data / 10)


def _lin2log(data):
    return 10 * np.log10(data)
