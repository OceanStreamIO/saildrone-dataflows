import numpy as np
import xarray as xr
import dask.array as da

from functools import partial
from typing import Tuple, Dict


def _nanpct_keepdims(block: np.ndarray, *, q: float, axis=None):
    """
    NumPy helper: percentile over `axis`, but broadcast back so the result
    has the *same shape* as `block`.  This satisfies xarray.rolling.reduce.
    """

    return np.nanpercentile(block, q, axis=axis, keepdims=True)


def rolling_nanpercentile(arr, *, q: float, axis=None):
    """
    Reducer usable in xarray.rolling.reduce that works for both NumPy
    and Dask chunks, keeps dimensionality, and avoids the FutureWarning.
    """
    if isinstance(arr, da.Array):
        return da.map_blocks(
            _nanpct_keepdims,
            arr,
            dtype=arr.dtype,
            chunks=arr.chunks,
            q=q,
            axis=axis,
        )
    return _nanpct_keepdims(arr, q=q, axis=axis)


def transient_noise_mask(
    channel_dataset: xr.Dataset,
    params: Dict,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Ryan et al. (2015) transient‑noise filter for a single channel.

    Parameters
    ----------
    channel_dataset : Dataset with "Sv" (dB)
    params : dict
        range_coord            : vertical coordinate name (default "echo_range")
        ping_window            : half‑width (pings)      (default 5)
        range_window           : half‑width (samples)    (default 3)
        threshold              : dB above block statistic (default 6)
        exclude_above          : min range to apply (m)  (default 250)
        percentile             : percentile for block (default 15)
        min_pings / min_samples: override min_periods

    Returns
    -------
    mask_transient, mask_unfeasible : Boolean DataArrays
    """
    rng_var = params.get("range_coord", "echo_range")
    half_ping = params.get("ping_window", 5)
    half_range = params.get("range_window", 3)
    thr_db = params.get("threshold", 6.0)
    excl_above = params.get("exclude_above", 250.0)
    perc = params.get("percentile", 15)
    min_pings = params.get("min_pings", 2 * half_ping + 1)
    min_samples = params.get("min_samples", 2 * half_range + 1)

    Sv_db = channel_dataset["Sv"]
    range_values = channel_dataset[rng_var]
    ping_dim = "ping_time"
    range_dim = range_values.dims[0]

    block_ping = 2 * half_ping + 1
    block_range = 2 * half_range + 1
    min_periods = min_pings * min_samples

    # apply exclusion range mask
    in_zone = range_values >= excl_above

    # convert to linear & rolling block pXX
    Sv_lin = 10.0 ** (Sv_db / 10.0)

    rolled = Sv_lin.rolling(
        {ping_dim: block_ping, range_dim: block_range},
        center=True,
        min_periods=min_periods,
    )

    pct_func = partial(rolling_nanpercentile, q=perc)
    block_lin = rolled.reduce(pct_func, keep_attrs=True)

    # back to dB
    block_db = 10.0 * np.log10(block_lin)

    # compute masks
    diff_db = Sv_db - block_db
    mask_transient = (diff_db > thr_db) & in_zone
    mask_unfeasible = block_db.isnull() & in_zone

    mask_transient = mask_transient.where(in_zone, False)
    mask_unfeasible = mask_unfeasible.where(in_zone, False)

    return mask_transient, mask_unfeasible
