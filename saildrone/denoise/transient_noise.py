import numpy as np
import pandas as pd
import xarray as xr
import dask_image.ndfilters

from saildrone.utils import frequency_nominal_to_channel
from saildrone.utils.mask_transformation import rolling_median_block, line_to_square, dask_nanmedian


FIELDING_DEFAULT_PARAMS = {
    "r0": 200,
    "r1": 1000,
    "n": 5,
    "thr": [2, 0],
    "roff": 250,
    "jumps": 5,
    "maxts": -35,
    "start": 0
}


def _fielding(source_Sv: xr.DataArray, desired_channel: str, parameters: dict, chunks=None):
    """
    Mask transient noise with method proposed by Fielding et al (unpub.).

    A comparison is made ping by ping with respect to a block in a reference
    layer set at far range, where transient noise mostly occurs. If the ping
    median is greater than the block median by a user-defined threshold, the
    ping will be masked all the way up, until transient noise disappears, or
    until it gets the minimum range allowed by the user.

    Args:
        source_Sv (xr.DataArray): Sv array
        desired_channel (str): name of the channel to process
        parameters(dict): dict of parameters, containing:
            r0    (int  ): range below which transient noise is evaluated (m).
            r1    (int  ): range above which transient noise is evaluated (m).
            n     (int  ): n of preceding & subsequent pings defining the block.
            thr   (int  ): user-defined threshold for side-comparisons (dB).
            maxts (int  ): max transient noise permitted, prevents to interpret
                           seabed as transient noise (dB).
            jumps (int  ): height of vertical steps (m).
        chunks (dict, optional): Chunk sizes for dask array. Default is None.

    Returns:
        xarray.DataArray: xr.DataArray with mask indicating the presence of transient noise.
    """
    # Validate parameters
    parameter_names = ("r0", "r1", "n", "thr", "maxts", "jumps")
    missing_params = set(parameter_names) - set(parameters.keys())
    if missing_params:
        raise ValueError(f"Missing parameters: {missing_params}")

    r0, r1, n, thr, maxts, jumps = [parameters.get(p) for p in ["r0", "r1", "n", "thr", "maxts", "jumps"]]
    roff = parameters.get('roff', 250)
    start = parameters.get('start', 0)

    # Extract data
    Sv = source_Sv.sel(channel=desired_channel)['Sv']
    depth = Sv['depth']

    # Create the mask
    mask = xr.full_like(Sv, False, dtype=bool)

    print(f"Thresholds: thr[0]={thr[0]}, thr[1]={thr[1]}")

    for j in range(Sv.sizes['ping_time']):
        current_time = Sv.ping_time.values[j]
        time_slice = create_nearest_slice(source_Sv, current_time - pd.Timedelta(minutes=n),
                                          current_time + pd.Timedelta(minutes=n))

        ping = Sv.sel(ping_time=current_time)
        block = Sv.sel(ping_time=time_slice)

        ping_median = ping.sel(depth=slice(r0, r1)).median()
        block_median = block.sel(depth=slice(r0, r1)).median()
        ping_p75 = ping.sel(depth=slice(r0, r1)).quantile(0.75)

        if (ping_p75 < maxts) and ((ping_median - block_median) > thr[0]):
            depth_idx = int(np.abs(depth - r0).argmin().item())
            min_depth_idx = int(np.abs(depth - roff).argmin().item())

            while depth_idx > min_depth_idx:
                segment_ping = ping.sel(depth=slice(depth_idx, depth_idx + jumps))
                segment_block = block.sel(depth=slice(depth_idx, depth_idx + jumps))

                if (segment_ping.median() - segment_block.median()) < thr[1]:
                    break
                depth_idx -= jumps

            mask.loc[dict(ping_time=current_time, depth=slice(depth_idx, depth.max()))] = True

    return mask


def create_nearest_slice(ds, start_time, end_time):
    """Finds the nearest start and end times within the dataset for given timestamps."""
    nearest_start = ds.sel(ping_time=start_time, method='nearest').ping_time.values
    nearest_end = ds.sel(ping_time=end_time, method='nearest').ping_time.values
    return slice(nearest_start, nearest_end)


def get_transient_noise_mask(
    source_Sv: xr.Dataset,
    parameters: dict,
    desired_channel: str = None,
    desired_frequency: int = None,
    method: str = "ryan",
) -> xr.DataArray:
    """
    Create a transient noise mask.
    This method is based on:
    Ryan et al. (2015) ‘Reducing bias due to noise and attenuation in
        open-ocean echo integration data’, ICES Journal of Marine Science,
        72: 2482–2493.

    Parameters
    ----------
    source_Sv: xr.Dataset or str or pathlib.Path
        If a Dataset this value contains the Sv data to create a mask for,
        else it specifies the path to a zarr or netcdf file containing
        a Dataset. This input must correspond to a Dataset that has the
        coordinate ``channel`` and variables ``frequency_nominal`` and ``Sv``.
    desired_channel: str
        Name of the desired frequency channel.
    desired_frequency: int
        Desired frequency, in case the channel is not directly specified
    mask_type: str with either "ryan" or "fielding" based on
        the preferred method for signal attenuation mask generation
    Returns
    -------
    xr.DataArray
        A DataArray containing the mask for the Sv data. Regions satisfying the thresholding
        criteria are filled with ``True``, else the regions are filled with ``False``.

    Raises
    ------
    ValueError
        If neither ``ryan`` or ``fielding`` are given

    """
    mask_map = {
        "fielding": _fielding,
        "ryan": _ryan
    }

    if method not in mask_map.keys():
        raise ValueError(f"Unsupported method: {method}")

    if desired_channel is None:
        if desired_frequency is None:
            raise ValueError("Must specify either desired channel or desired frequency")

        desired_channel = frequency_nominal_to_channel(source_Sv, desired_frequency)

    mask = mask_map[method](source_Sv, desired_channel, parameters)
    return mask


def lin(variable):
    """
    Turn variable into the linear domain.

    Args:
        variable (float): array of elements to be transformed.

    Returns:
        float:array of elements transformed
    """

    lin = 10 ** (variable / 10)
    return lin


def _ryan(ds_Sv, channel, parameters, range_var='depth', chunks=None):
    Sv = ds_Sv.sel(channel=channel)['Sv']

    if chunks:
        ds_Sv = ds_Sv.chunk(chunks)

    depth_bin = parameters.get('depth_bin', 5)
    num_side_pings = parameters.get('num_side_pings', 20)
    thr = parameters.get('transient_noise_threshold', 2)
    exclude_above = parameters.get('exclude_above', 250)
    operation = parameters.get('operation', 'median')
    func = np.nanmean if operation == 'mean' else np.nanmedian

    # Calculate depth resolution and pooling size
    depth_resolution = np.nanmean(np.diff(ds_Sv[range_var].values))
    num_depth_indices = int(np.ceil(depth_bin / depth_resolution))
    pooling_size = [(2 * num_side_pings) + 1, (2 * num_depth_indices) + 1]
    print(f"Pooling size: {pooling_size}")

    # Apply pooling without masking
    pooled_Sv = dask_image.ndfilters.generic_filter(
        Sv.data,
        function=func,
        size=pooling_size,
        mode="reflect",
    )
    print(f"Raw pooled_Sv sample: {pooled_Sv[:10, :10]}")

    # Clamp pooled_Sv to avoid log10 issues
    pooled_Sv = np.maximum(pooled_Sv, 1e-10)
    pooled_Sv_log = 10 * np.log10(pooled_Sv)
    print(f"Pooled Sv log sample: {pooled_Sv_log[:10, :10]}")

    # Create final DataArray
    pooled_Sv_log = xr.DataArray(
        pooled_Sv_log,
        dims=["ping_time", "depth"],
        coords={"ping_time": Sv["ping_time"], "depth": Sv["depth"]},
    )

    # Compute transient noise mask
    transient_noise_mask = (Sv - pooled_Sv_log > thr)

    # Apply depth mask
    depth_mask = Sv[range_var] < exclude_above
    combined_mask = transient_noise_mask & ~depth_mask

    print(f"Sv sample: {Sv[:10, :10].values}")
    print(f"Sv - Pooled Sv log: {Sv[:10, :10].values - pooled_Sv_log[:10, :10].values}")
    print(f"Threshold: {thr}")
    print(f"Transient noise mask sample: {transient_noise_mask[:10, :10]}")

    return combined_mask



"""
def ____ryan(ds, desired_channel, parameters, chunks=None):
    print("DEBUG: Starting _ryan function")

    Sv = ds.sel(channel=desired_channel)['Sv']
    print(f"DEBUG: Initial dataset shape: {Sv.shape}")
    print(f"DEBUG: Initial dataset chunks: {Sv.chunks}")

    # Automatic chunking if none provided
    if not Sv.chunks:
        Sv = Sv.chunk({'ping_time': 500, 'depth': 500})
        print("DEBUG: Chunking applied to Sv:", Sv.chunks)

    # Extract parameters with defaults
    m = parameters.get('depth_bin', 5)
    n = parameters.get('num_side_pings', 20)
    thr = parameters.get('transient_noise_threshold', 2)
    excludeabove = parameters.get('exclude_above', 250)
    operation = parameters.get('operation', 'median')
    print(f"DEBUG: Parameters -> m: {m}, n: {n}, thr: {thr}, excludeabove: {excludeabove}, operation: {operation}")

    # Calculate offsets
    depth_bin_width = Sv.depth.diff('depth').mean().item()
    ioff = int(m / depth_bin_width)
    joff = n

    # Use map_overlap
    def sliding_window(block):
        return da.lib.stride_tricks.sliding_window_view(block, (2 * ioff + 1, 2 * joff + 1))

    Sv_rolled = Sv.data.map_overlap(
        sliding_window,
        depth=ioff,  # Overlap size for depth dimension
        ping_time=joff,  # Overlap size for ping_time dimension
        boundary="reflect",
        dtype=Sv.dtype,
    )

    # Define reduction operation
    def reduction_op(block):
        if operation == 'median':
            return np.nanmedian(block, axis=(-2, -1))
        elif operation == 'mean':
            return np.nanmean(block, axis=(-2, -1))
        elif 'percentile' in operation:
            q = int(operation.replace('percentile', ''))
            return np.nanpercentile(block, q=q, axis=(-2, -1))
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    # Apply reduction with map_blocks
    template = xr.DataArray(
        da.empty(Sv.shape, chunks=Sv.chunks, dtype=Sv.dtype),
        dims=Sv.dims,
        coords=Sv.coords,
    )
    Sv_reduced = xr.apply_ufunc(
        reduction_op,
        Sv_rolled,
        input_core_dims=[['window_depth', 'window_ping']],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[Sv.dtype],
    )

    # Mask computation
    mask = ((Sv - Sv_reduced) > thr) & (Sv.depth >= excludeabove)
    return mask
"""