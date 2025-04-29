import pathlib
import warnings
from pandas import Index

import dask.array as da
import numpy as np
import xarray as xr

from dask_image.ndfilters import convolve
from dask_image.ndmorph import binary_dilation, binary_erosion
from scipy.signal import medfilt

MAX_SV_DEFAULT_PARAMS = {"r0": 10, "r1": 1000, "roff": 0, "thr": (-40, -60)}
DELTA_SV_DEFAULT_PARAMS = {"r0": 10, "r1": 1000, "roff": 0, "thr": 20}
BLACKWELL_DEFAULT_PARAMS = {
    "r0": 10,
    "r1": 1000,
    "tSv": -75,
    "ttheta": 60,
    "tphi": 60,
    "wtheta": 28,
    "wphi": 52,
}
BLACKWELL_MOD_DEFAULT_PARAMS = {
    "r0": 10,
    "r1": 1000,
    "tSv": -75,
    "ttheta": 702,
    "tphi": 282,
    "wtheta": 28,
    "wphi": 52,
    "rlog": None,
    "tpi": None,
    "freq": None,
    "rank": 50,
}
EXPERIMENTAL_DEFAULT_PARAMS = {
    "r0": 10,
    "r1": 1000,
    "roff": 0,
    "thr": (-30, -70),
    "ns": 150,
    "n_dil": 3,
}
ARIZA_DEFAULT_PARAMS = {
    "r0": 10,
    "r1": 1300,
    "roff": 0,
    "thr": -40,
    "ec": 1,
    "ek": (3, 3),
    "dc": 3,
    "dk": (3, 3),
}

ARIZA_SPIKE_DEFAULT_PARAMS = {
    "r0": 10,
    "r1": 10000,
    "roff": 0,
    "thr": (-40, -40),
    "ec": 1,
    "ek": (3, 3),
    "dc": 3,
    "dk": (3, 3),
    "maximum_spike": 200,
}

ARIZA_EXPERIMENTAL_DEFAULT_PARAMS = {
    "r0": 10,
    "r1": 1000,
    "roff": 0,
    "thr": (-40, -70),
    "ec": 1,
    "ek": (3, 3),
    "dc": 3,
    "dk": (3, 3),
}


def get_seabed_mask_multichannel(ds: xr.Dataset, parameters: dict = None) -> xr.DataArray:
    channel_list = ds["channel"].values
    mask_list = []

    if parameters is None:
        parameters = BLACKWELL_DEFAULT_PARAMS

    for channel in channel_list:
        mask = max_sv_seabed_mask(
            ds,
            channel=channel,
            depth_min=parameters["r0"],
            depth_max=parameters["r1"],
            threshold_db=(-40.0, -60.0)
        )

        # mask = get_seabed_mask(
        #     source_Sv,
        #     parameters=parameters,
        #     desired_channel=channel,
        #     method=method,
        # )
        mask_list.append(mask)
    mask = create_multichannel_mask(mask_list, channel_list)
    return mask


def create_multichannel_mask(masks: [xr.Dataset], channels: [str]) -> xr.Dataset:
    if len(masks) != len(channels):
        raise ValueError("number of masks and of channels provided should be the same")

    for i in range(0, len(masks)):
        mask = masks[i]
        if "channel" in mask.dims:
            masks[i] = mask.isel(channel=0)

    result = xr.concat(
        masks, Index(channels, name="channel"), data_vars="all", coords="all", join="exact"
    )
    return result


def get_seabed_mask(source_Sv: xr.Dataset,
                    parameters: dict,
                    desired_channel: str = None,
                    desired_frequency: int = None,
                    method: str = "ariza"
                    ) -> xr.DataArray:
    """
    Create a mask based on the identified signal attenuations of Sv values at 38KHz.
    Parameters
    ----------
    source_Sv: xr.Dataset or str or pathlib.Path
        If a Dataset this value contains the Sv data to create a mask for,
        else it specifies the path to a zarr or netcdf file containing
        a Dataset. This input must correspond to a Dataset that has the
        coordinate ``channel`` and variables ``frequency_nominal`` and ``Sv``.
    desired_channel: str - channel to generate the mask for
    desired_frequency: int - desired frequency, in case the channel isn't directly specified
    method: str with either "ariza", "blackwell", based on the preferred method
        for seabed mask generation
    Returns
    -------
    xr.DataArray
        A DataArray containing the mask for the Sv data. Regions satisfying the thresholding
        criteria are filled with ``True``, else the regions are filled with ``False``.

    Raises
    ------
    ValueError
        If neither "ariza", "blackwell" are given

    Notes
    -----


    Examples
    --------

    """
    mask_map = {
        "ariza": _ariza,
        "blackwell": _blackwell
    }

    if method not in mask_map.keys():
        raise ValueError(f"Unsupported method: {method}")

    mask = mask_map[method](source_Sv, desired_channel, parameters)

    return mask


def _get_seabed_range(mask: xr.DataArray):
    """
    Given a seabed mask, returns the depth depth of the seabed

    Args:
        mask (xr.DataArray): seabed mask

    Returns:
        xr.DataArray: a ping_time-sized array containing the depth seabed depth,
        or max depth if no seabed is detected

    """
    seabed_depth = mask.argmax(dim="depth").compute()
    seabed_depth[seabed_depth == 0] = mask.depth.max().item()
    return seabed_depth


def _morpho(mask: xr.DataArray, operation: str, c: int, k: int):
    """
    Given a preexisting 1/0 mask, run erosion or dilation cycles on it to remove noise

    Args:
        mask (xr.DataArray): xr.DataArray with 1 and 0 data
        operation(str): dilation, erosion
        c (int): number of cycles.
        k (int): 2-elements tuple with vertical and horizontal dimensions
                      of the kernel.

    Returns:
        xr.DataArray: A DataArray containing the denoised mask.
            Regions satisfying the criteria are 1, others are 0
    """
    function_dict = {"dilation": binary_dilation, "erosion": binary_erosion}

    if c > 0:
        dask_mask = da.asarray(mask, allow_unknown_chunksizes=False)
        dask_mask.compute_chunk_sizes()
        dask_mask = function_dict[operation](
            dask_mask,
            structure=da.ones(shape=k, dtype=bool),
            iterations=c,
        ).compute()
        dask_mask = da.asarray(dask_mask, allow_unknown_chunksizes=False)
        dask_mask.compute()
        mask.values = dask_mask.compute()
    return mask


def _erode_dilate(mask: xr.DataArray, ec: int, ek: int, dc: int, dk: int):
    """
    Given a preexisting 1/0 mask, run erosion and dilation cycles on it to remove noise

    Args:
        mask (xr.DataArray): xr.DataArray with 1 and 0 data
        ec (int): number of erosion cycles.
        ek (int): 2-elements tuple with vertical and horizontal dimensions
                      of the erosion kernel.
        dc (int): number of dilation cycles.
        dk (int): 2-elements tuple with vertical and horizontal dimensions
                      of the dilation kernel.

    Returns:
        xr.DataArray: A DataArray containing the denoised mask.
            Regions satisfying the criteria are 1, others are 0
    """
    mask = _morpho(mask, "erosion", ec, ek)
    mask = _morpho(mask, "dilation", dc, dk)
    return mask


def _create_range_mask(Sv_ds: xr.DataArray, desired_channel: str, thr: int, r0: int, r1: int):
    """
    Return a raw threshold/range mask for a certain dataset and desired channel

    Args:
        Sv_ds (xr.DataArray): xr.DataArray with Sv data for multiple channels (dB)
        desired_channel(str): Name of the desired frequency channel
        r0 (int): minimum range below which the search will be performed (m).
        r1 (int): maximum range above which the search will be performed (m).
        thr (int): Sv threshold above which seabed might occur (dB).

    Returns:
        dict: a dict containing the mask and whether or not further processing is necessary
            mask (xr.DataArray): a basic range/threshold mask.
                                Regions satisfying the criteria are 1, others are 0
            ok (bool): should the mask be further processed  or is there no data to be found?

    """
    try:
        channel_Sv = Sv_ds.sel(channel=desired_channel)
        Sv = channel_Sv["Sv"]
        r = channel_Sv["echo_range"][0]

        # Early exit if out of bounds
        if (r0 > r.max()) or (r1 < r.min()):
            warnings.warn(
                "The searching range is outside the echosounder range. "
                "Returning default unmasked data."
            )
            mask = xr.DataArray(
                np.ones_like(Sv, dtype=bool),
                dims=("ping_time", "depth"),
                coords={"ping_time": Sv.ping_time, "depth": Sv.depth},
            )
            return {"mask": mask, "ok": False, "Sv": Sv, "range": r}

        # Lazily calculate bounding indices
        # up_idx = abs(r - r0).argmin(dim="depth")
        # lw_idx = abs(r - r1).argmin(dim="depth")

        # Create depth bounds directly from values instead of indices
        # upper_range = r.isel(depth=up_idx)
        # lower_range = r.isel(depth=lw_idx)

        base_mask = xr.where(Sv > thr, 1, 0).drop_vars("channel", errors="ignore")

        # Lazy mask
        echo_range = channel_Sv["echo_range"]
        range_filter = (echo_range >= r0) & (echo_range <= r1)
        masked = base_mask.where(range_filter, other=0)

        # Check if any signal survives — lazily
        # nonzero = masked.sum()

        # Return everything deferred
        return {
            "mask": masked,
            "ok": xr.apply_ufunc(lambda x: x > 0, masked.sum(), dask="allowed"),
            "Sv": Sv,
            "range": r
        }
    except Exception as e:
        print(e)
        raise ValueError("Error in creating range mask: " + str(e))


def _mask_down(mask: xr.DataArray):
    """
    Given a seabed mask, masks all signal under the detected seabed

    Args:
          mask (xr.DataArray): seabed mask

    Returns:
           xr.DataArray(mask with area under seabed masked)
    """
    seabed_depth = _get_seabed_range(mask)
    mask = (mask["depth"] <= seabed_depth).transpose()
    return mask


# move to utils and rewrite transient noise/fielding to use this once merged
def _erase_floating_zeros(mask: xr.DataArray):
    """
    Given a boolean mask, turns back to True any "floating" False values,
    e.g. not attached to the max range

    Args:
        mask: xr.DataArray - mask to remove floating values from

    Returns:
        xr.DataArray - mask with floating False values removed

    """
    flipped_mask = mask.isel(depth=slice(None, None, -1))
    flipped_mask["depth"] = mask["depth"]
    ft = len(flipped_mask.depth) - flipped_mask.argmax(dim="depth")

    first_true_indices = xr.DataArray(
        line_to_square(ft, mask, dim="depth").transpose(),
        dims=("ping_time", "depth"),
        coords={"ping_time": mask.ping_time, "depth": mask.depth},
    )

    indices = xr.DataArray(
        line_to_square(mask["depth"], mask, dim="ping_time"),
        dims=("ping_time", "depth"),
        coords={"ping_time": mask.ping_time, "depth": mask.depth},
    )
    spike_mask = mask.where(indices > first_true_indices, True)

    mask = spike_mask
    return mask


def _experimental_correction(mask: xr.DataArray, Sv: xr.DataArray, thr: int):
    """
    Given an existing seabed mask, the single-channel dataset it was created on
    and a secondary, lower threshold, it builds the mask up until the Sv falls below the threshold

    Args:
          mask (xr.DataArray): seabed mask
          Sv (xr.DataArray): single-channel Sv data the mask was build on
          thr (int): secondary threshold

    Returns:
          xr.DataArray: mask with secondary threshold correction applied

    """
    secondary_mask = xr.where(Sv < thr, 1, 0).drop("channel")
    secondary_mask.fillna(1)
    fill_mask = secondary_mask & mask
    spike_mask = _erase_floating_zeros(fill_mask)
    return spike_mask


def _cut_spikes(mask: xr.DataArray, maximum_spike: int):
    """
    In the Ariza seabed detecting method, large shoals can be falsely detected as
    seabed. Their appearance on the seabed mask is large vertical "spikes".
    We want to remove any such spikes from the dataset using maximum_spike
    as a control parameter.

    If this option is used, we also recommend applying the _experimental_correction,
    even if with the same threshold as the initial threshold, to fill up any
    imprecisions in the interpolated seabed

    Args:
        mask (xr.DataArray):
        maximum_spike(int): maximum height, in range samples, acceptable before
                            we start removing that data

    Returns:
        xr.DataArray: the corrected mask
    """
    int_mask = (~mask.copy()).astype(int)
    seabed = _get_seabed_range(int_mask)
    shifted_seabed = seabed.shift(ping_time=-1, fill_value=seabed[-1])
    spike = seabed - shifted_seabed
    spike_sign = xr.where(abs(spike) > maximum_spike, xr.where(spike > 0, 1, -1), 0)
    spike_cs = spike_sign.cumsum(dim="ping_time")
    # for i in spike:
    #     print(i.item())

    # mask spikes
    nan_mask = xr.where(spike_cs > 0, np.nan, xr.where(spike_sign == -1, np.nan, mask))

    # fill in with interpolated values from non-spikes
    mask_interpolated = nan_mask.interpolate_na(dim="ping_time", method="nearest").astype(bool)

    return mask_interpolated


def _ariza(Sv_ds: xr.DataArray, desired_channel: str, parameters: dict = ARIZA_DEFAULT_PARAMS):
    """
    Mask Sv above a threshold to get potential seabed features. These features
    are eroded first to get rid of fake seabeds (spikes, schools, etc.) and
    dilated afterwards to fill in seabed breaches. Seabed detection is coarser
    than other methods (it removes water nearby the seabed) but the seabed line
    never drops when a breach occurs. Suitable for pelagic assessments and
    reconmended for non-supervised processing.

    Args:
        Sv_ds (xr.DataArray): xr.DataArray with Sv data for multiple channels (dB)
        desired_channel(str): Name of the desired frequency channel
        parameters: parameter dict, should contain:
            r0 (int): minimum range below which the search will be performed (m).
            r1 (int): maximum range above which the search will be performed (m).
            roff (int): seabed range offset (m).
            thr (int): Sv threshold above which seabed might occur (dB).
                Can be a tuple, case in which a secondary experimental
                thresholding correction is applied
            ec (int): number of erosion cycles.
            ek (int): 2-elements tuple with vertical and horizontal dimensions
                      of the erosion kernel.
            dc (int): number of dilation cycles.
            dk (int): 2-elements tuple with vertical and horizontal dimensions
                      of the dilation kernel.
            maximum_spike(int): optional, if not None, used to determine the maximum
                    allowed height of the "spikes" potentially created by
                    dense shoals before masking them out. If used, applying
                    a secondary threshold correction is recommended


    Returns:
        xr.DataArray: A DataArray containing the mask for the Sv data.
            Regions satisfying the thresholding criteria are True, others are False
    """
    parameter_names = ["r0", "r1", "roff", "thr", "ec", "ek", "dc", "dk"]
    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError(
            "Missing parameters - should be: "
            + str(parameter_names)
            + ", are: "
            + str(parameters.keys())
        )
    r0 = parameters["r0"]
    r1 = parameters["r1"]
    thr = parameters["thr"]
    ec = parameters["ec"]
    ek = parameters["ek"]
    dc = parameters["dc"]
    dk = parameters["dk"]
    secondary_thr = None
    maximum_spike = None
    if "maximum_spike" in parameters.keys():
        maximum_spike = parameters["maximum_spike"]

    if isinstance(thr, int) is False:
        secondary_thr = thr[1]
        thr = thr[0]

    # create raw range and threshold mask, if no seabed is detected return empty
    raw = _create_range_mask(Sv_ds, desired_channel=desired_channel, thr=thr, r0=r0, r1=r1)
    mask = raw["mask"]
    if raw["ok"] is False:
        return mask

    # run erosion and dilation denoising cycles
    mask = _erode_dilate(mask, ec, ek, dc, dk)

    # mask areas under the detected seabed
    mask = _mask_down(mask)

    # apply spike correction
    if maximum_spike is not None:
        mask = _cut_spikes(mask, maximum_spike)

    # apply experimental correction, if specified
    if secondary_thr is not None:
        mask = _experimental_correction(mask, raw["Sv"], secondary_thr)

    return mask


def max_sv_seabed_mask(
    ds: xr.Dataset,
    channel: str = None,
    depth_min: float = 10.0,
    depth_max: float = 1000.0,
    range_offset: float = 0.0,
    threshold_db: tuple = (-40.0, -60.0),
    smooth_pings: int = 5,
    mean_len: int = 5
) -> xr.DataArray:
    """
    Detects seabed based on maximum Sv within a specified depth window for a given channel.
    Applies an upward trace until Sv falls below a secondary threshold, with optional smoothing.

    Parameters:
        ds (xr.Dataset): Dataset with variables Sv (in dB), dimensions: (channel, ping_time, depth)
        channel (str): Channel name to select from the 'channel' dimension
        depth_min (float): Minimum range (m) for search window
        depth_max (float): Maximum range (m) for search window
        range_offset (float): Range offset to subtract from final bottom pick (m)
        threshold_db (tuple): (primary_threshold, secondary_threshold) in dB
        smooth_pings (int): Window size (in pings) for moving median smoothing
        mean_len (int): Window size for rolling mean calculation

    Returns:
        xr.DataArray: Boolean mask (True = seabed and below), shape: (channel, ping_time, depth)
    """
    # guarantee depth is a single chunk
    Sv = ds["Sv"].sel(channel=channel).chunk({"depth": -1})
    depth = ds["depth"]

    win = Sv.sel(depth=slice(depth_min, depth_max))
    win = win.where(win.notnull().any("depth"))

    # 1) primary bottom pick (keeps NaNs)
    # idxmax works lazily & returns depth coordinate
    bottom_depth = win.where(win >= threshold_db[0]).idxmax("depth")
    bottom_depth = bottom_depth.where(~bottom_depth.isnull())
    bottom_depth = bottom_depth.assign_coords(ping_time=ds.ping_time)

    # 2) 5-sample linear-mean to test “step-up” threshold
    lin = 10.0 ** (Sv / 10.0)
    mean5 = lin.rolling(depth=mean_len, min_periods=1).mean()

    thresh_ok = (10 * np.log10(mean5) <= threshold_db[1])

    # 3) vectorised upward search
    def _step(depth_pick, ok_flags, mean_len):
        if np.isnan(depth_pick):
            return np.nan

        idx = np.searchsorted(depth.values, depth_pick)

        while idx >= mean_len and not ok_flags[idx - 1]:
            idx -= 1

        return float(idx)

    def _to_idx(depth_pick, shift, range_offset):
        """
        Convert depth coordinate to integer index minus shift & range-offset.
        Works on plain numpy scalars/arrays – safe for dask='parallelized'.
        """
        if np.isnan(depth_pick) or np.isnan(shift):
            return np.nan

        idx = np.searchsorted(depth.values, depth_pick)
        res = idx - shift - range_offset

        return float(res if res > 0 else 0.0)

    shift = xr.apply_ufunc(
        _step,
        bottom_depth,
        thresh_ok,
        kwargs=dict(mean_len=mean_len),
        input_core_dims=[[], ["depth"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    shift = shift.reindex(ping_time=ds.ping_time, fill_value=np.nan)

    bottom_idx = xr.apply_ufunc(
        _to_idx,
        bottom_depth,  # depth coordinate per ping (scalar/array)
        shift,  # shift returned by the first gufunc
        kwargs=dict(range_offset=range_offset),
        input_core_dims=[[], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    bottom_idx = bottom_idx.reindex(ping_time=ds.ping_time)

    # 4) optional ping-median smoothing
    if smooth_pings > 1:
        bottom_idx = (
            bottom_idx
            .rolling(ping_time=smooth_pings, center=True, min_periods=1)
            .median()
        )

    # 5) boolean mask by broadcasting (lazy)
    mask = depth >= bottom_idx
    mask = mask.transpose("ping_time", "depth")

    print(
        bottom_idx.sizes,
        mask.sizes,
        ds.ping_time.size
    )

    mask_da = xr.DataArray(
        mask,
        dims=("ping_time", "depth"),
        coords={"ping_time": ds.ping_time,
                "depth": depth},
        name="seabed_mask",
        attrs={"long_name": "bottom & below mask (True = seabed)"},
    ).expand_dims(channel=[channel])

    return mask_da


""" ///////////////////////////// BLACKWELL METHOD ////////////////////////////////// """


def _blackwell(Sv_ds: xr.DataArray, desired_channel: str, parameters: dict):
    """
    Detects and mask seabed using the split-beam angle and Sv, based in
    "Blackwell et al (2019), Aliased seabed detection in fisheries acoustic
    data". Complete article here: https://arxiv.org/abs/1904.10736

    Args:
        Sv_ds (xr.DataArray): xr.DataArray with Sv data for multiple channels (dB)
        desired_channel(str): Name of the desired frequency channel
        parameters: parameter dict, should contain:
            r0 (int): minimum range below which the search will be performed (m)
            r1 (int): maximum range above which the search will be performed (m)
            tSv (float): Sv threshold above which seabed is pre-selected (dB)
            ttheta (int): Theta threshold above which seabed is pre-selected (dB)
            tphi (int): Phi threshold above which seabed is pre-selected (dB)
            wtheta (int): window's size for mean square operation in Theta field
            wphi (int): window's size for mean square operation in Phi field

    Returns:
        xr.DataArray: A DataArray containing the mask for the Sv data.
            Regions satisfying the thresholding criteria are True, others are False
    """
    parameter_names = ["r0", "r1", "tSv", "ttheta", "tphi", "wtheta", "wphi"]
    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError(
            "Missing parameters - should be: "
            + str(parameter_names)
            + ", are: "
            + str(parameters.keys())
        )

    ttheta = parameters["ttheta"]
    tphi = parameters["tphi"]
    wtheta = parameters["wtheta"]
    wphi = parameters["wphi"]

    channel_Sv = Sv_ds.sel(channel=desired_channel)

    if not all(attr in channel_Sv.data_vars for attr in ["angle_alongship", "angle_athwartship"]):
        raise ValueError("Missing required angle variables: 'angle_alongship' and/or 'angle_athwartship'")

    theta = channel_Sv["angle_alongship"]
    phi = channel_Sv["angle_athwartship"]
    scaling_factor = (22.0 * 128.0) / 180.0  # Ensure float division
    theta = theta * scaling_factor
    phi = phi * scaling_factor

    dask_theta = da.asarray(theta, allow_unknown_chunksizes=False)
    dask_theta.compute_chunk_sizes()

    theta.values = convolve(
        dask_theta,
        weights=da.ones(shape=(wtheta, wtheta), dtype=float) / wtheta ** 2,
        mode="nearest",
    ).compute()

    dask_phi = da.asarray(phi, allow_unknown_chunksizes=False)
    dask_phi.compute_chunk_sizes()

    phi.values = convolve(
        dask_phi,
        weights=da.ones(shape=(wphi, wphi), dtype=float) / wphi ** 2,
        mode="nearest",
    ).compute()

    angle_mask = ~((theta > ttheta) | (phi > tphi))

    if angle_mask.all():
        warnings.warn(
            "No aliased seabed detected in Theta & Phi. "
            "A default mask with all True values is returned."
        )
        return angle_mask

    mask = theta > ttheta

    return mask


def line_to_square(one: xr.DataArray, two: xr.DataArray, dim: str):
    """
    Given a single dimension dataset and an example dataset with 2 dimensions,
    returns a two-dimensional dataset that is the single dimension dataset
    repeated as often as needed

    Args:
        one (xr.DataArray): data
        two (xr.DataArray): shape dataset
        dim (str): name of dimension to concat against

    Returns:
        xr.DataArray: the input dataset, with the same coords as dataset_size and
        the values repeated to fill it up
    """
    length = len(two[dim])
    array_list = [one for _ in range(0, length)]
    array = xr.concat(array_list, dim=dim)
    # return_data = xr.DataArray(data=array.values, dims=two.dims, coords=two.coords)
    return array.values


def frequency_nominal_to_channel(source_Sv, frequency_nominal: int):
    """
    Given a value for a nominal frequency, returns the channel associated with it
    """
    channels = source_Sv["frequency_nominal"].coords["channel"].values
    freqs = source_Sv["frequency_nominal"].values
    chan = channels[freqs == frequency_nominal]
    assert len(chan) == 1, "Frequency not uniquely identified"
    channel = chan[0]
    return channel
