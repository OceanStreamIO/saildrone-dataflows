import xarray as xr
from typing import Mapping, Dict, List, Tuple, Any, Hashable, Sequence, Union, Optional


def build_full_mask(
    ds: Union[xr.Dataset, str],
    stages: Mapping[str, Dict[str, Any]],
    var_name: str = "Sv",
    return_stage_masks: bool = False,
    mask_unfeasible: bool = False,
    pulse_length = None
) -> xr.DataArray:
    n_ch = ds.dims["channel"]

    stage_masks = {stage_name: [] for stage_name in stages}

    # build per-channel combined mask
    ch_masks: List[xr.DataArray] = []

    for ch in range(n_ch):
        ch_ds = ds.isel(channel=ch)
        stage_or = None
        reference = ds[var_name].isel(channel=ch)

        # iterate in declared order
        for stage_name, spec in stages.items():
            pars = _params_for_channel(spec["param_sets"], ch_ds, pulse_length)

            if pars is None:
                continue  # skip if no parameters for this channel

            fn = spec["fn"]
            res = fn(ch_ds, pars)
            # fn may return (mask_as, mask_unfeasible, …) or a single mask
            if isinstance(res, (tuple, list)):
                # if mask_unfeasible is True, we also get a mask for unfeasible pings
                if mask_unfeasible:
                    stage_mask = res[0] | res[1]
                else:
                    stage_mask = res[0]
            else:
                stage_mask = res

            # Ensure coordinates match reference for safe concat
            stage_mask = stage_mask.broadcast_like(ds[var_name].isel(channel=ch))
            stage_mask = stage_mask.reset_coords(drop=True)
            for cname in reference.coords:
                stage_mask = stage_mask.assign_coords({cname: reference.coords[cname]})

            # Expand to channel dimension
            ch_value = ds["channel"].values[ch]
            stage_mask = stage_mask.expand_dims(channel=[ch_value])
            stage_mask = stage_mask & ~reference.isnull()

            # Store for per-stage cubes: exactly one per channel per stage
            stage_masks[stage_name].append(stage_mask)

            # Combine all stage masks for this channel (OR logic)
            stage_or = stage_mask if stage_or is None else (stage_or | stage_mask)

        ch_masks.append(stage_or)

    full_mask = xr.concat(ch_masks, dim="channel")
    full_mask = full_mask.broadcast_like(ds[var_name])
    full_mask.name = "combined_mask"

    if not return_stage_masks:
        return full_mask

    # stitch per-stage lists into cubes
    stage_cubes = [
        (name, xr.concat(m_list, dim="channel").broadcast_like(ds[var_name]))
        for name, m_list in stage_masks.items()
    ]

    return full_mask, stage_cubes


def extract_channel_and_drop_pings(
    ds: xr.Dataset,
    channel: Union[int, float],
    *,
    var_name: str = "Sv",
    drop_threshold: float = 1.0,
    freq_coord: str = "frequency_nominal",
) -> xr.Dataset:
    """
    From a denoised multi‐channel Dataset (with mask already applied to `var_name`),
    extract exactly one channel (by numeric frequency or positional index),
    then drop any ping_time slices where the fraction of NaNs in that channel’s
    `var_name` is ≥ drop_threshold.

    Parameters
    ----------
    ds : xr.Dataset
        Must have dims ('channel','ping_time','depth') and data_var `var_name`.
        The mask has already been applied so bad samples are NaN.
        Also must have coord `frequency_nominal(channel)`.
    channel : int or float
        If equal to a value in ds[freq_coord], selects by frequency (leaving
        a length‐1 channel dim). Otherwise interpreted as a 0‐based index.
    var_name : str
        Name of the variable to inspect for NaNs (default "Sv").
    drop_threshold : float
        Fraction (0–1) of NaNs in a ping_time above which that ping is removed.
        e.g. 1.0 removes only fully‐NaN pings; 0.5 removes any ping with ≥50% NaNs.
    freq_coord : str
        Name of the per‐channel frequency coordinate (default "frequency_nominal").

    Returns
    -------
    xr.Dataset
        A copy of `ds` containing only the selected channel (singleton dim)
        with its `ping_time` axis pruned of pings that exceed the NaN threshold.
    """
    # --- 1) select the channel, preserving the 'channel' dim
    freqs = ds[freq_coord].values
    if (freqs == channel).any():
        ds_ch = ds.sel(channel=ds[freq_coord] == channel)
    else:
        ds_ch = ds.isel(channel=int(channel), drop=False)

    # 2) Compute fraction-NaN per ping_time
    depth_dim = ds_ch[var_name].dims[-1]  # "depth" or "echo_range"
    n_nan = ds_ch[var_name].isnull().sum(dim=depth_dim)  # (ping_time,)
    total = ds_ch[var_name].sizes[depth_dim]  # scalar
    frac_nan = n_nan / total  # DataArray

    if "channel" in frac_nan.dims:
        frac_nan = frac_nan.squeeze("channel", drop=True)

    keep_mask = (frac_nan < drop_threshold).compute().values
    ds_ch_clean = ds_ch.isel(ping_time=keep_mask)

    return ds_ch_clean


def apply_full_mask(
    ds: xr.Dataset,
    full_mask: xr.DataArray,
    *,
    var_name: str = "Sv"
) -> xr.Dataset:
    fm = full_mask.broadcast_like(ds[var_name])
    ds_out = ds.copy()

    rolling_dims = [d for d in fm.dims if d.startswith("_rolling_dim_")]
    if rolling_dims:
        indexer = {d: 0 for d in rolling_dims}
        fm = fm.isel(indexer, drop=True)

    ds_out[var_name] = ds[var_name].where(~fm)

    return ds_out


def _params_for_channel(param_sets: Mapping[Any, Any], ch_ds, pulse_length: str | None):
    """
    Return the parameter dictionary for this channel.

    Behavior:
    - If `param_sets` is a single dict (not per-frequency), return it unchanged.
    - Else pick the per-frequency entry (supports int/str keys like 38000/"38000").
    - If `pulse_length` is provided ("short_pulse" or "long_pulse") and that key
      exists in the per-frequency entry, return that sub-dict (may be None).
      Otherwise return the per-frequency entry as-is.
    """
    # single-dict case (no per-frequency mappings)
    if not any(isinstance(v, Mapping) for v in param_sets.values()):
        return param_sets

    # resolve frequency key
    freq = float(ch_ds["frequency_nominal"].compute().item())
    key_str = str(int(freq))

    if freq in param_sets:
        entry = param_sets[freq]
    elif key_str in param_sets:
        entry = param_sets[key_str]
    else:
        raise ValueError(f"{freq} Hz parameters missing")

    # optional pulse-length selection
    if pulse_length and isinstance(entry, Mapping):
        pl = pulse_length.lower()
        if pl.startswith("short"):
            subkey = "short_pulse"
        elif pl.startswith("long"):
            subkey = "long_pulse"
        else:
            subkey = None

        if subkey and (subkey in entry):
            return entry[subkey]  # may be None (explicitly disabled)

    return entry