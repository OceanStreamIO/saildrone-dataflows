import xarray as xr
from typing import Mapping, Dict, List, Tuple, Any, Hashable, Sequence, Union


def build_full_mask(
    ds: Union[xr.Dataset, str],
    stages: Mapping[str, Dict[str, Any]],
    var_name: str = "Sv",
    return_stage_masks: bool = False,
    mask_unfeasible: bool = False,
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
            fn = spec["fn"]
            pars = _params_for_channel(spec["param_sets"], ch_ds)

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


def apply_full_mask(
        ds: xr.Dataset,
        full_mask: xr.DataArray,
        *,
        var_name: str = "Sv",
        drop_pings: bool = False,
) -> xr.Dataset:
    ds_out = ds.copy()

    # sanity: broadcast mask to data shape if needed
    full_mask = full_mask.broadcast_like(ds_out[var_name])

    # option 1: NaN mask
    if not drop_pings:
        ds_out[var_name] = ds_out[var_name].where(~full_mask)
        return ds_out

    # option 2: drop pings
    vertical_dim = full_mask.dims[-1]  # depth | echo_range
    bad_ping = full_mask.all(dim=vertical_dim).all(dim="channel")

    # Dask arrays cannot index; materialise this tiny vector once
    good_idx = (~bad_ping).compute().values
    ds_out = ds_out.isel(ping_time=good_idx)

    return ds_out


def _params_for_channel(param_sets, ch_ds):
    """Return the parameter dictionary for this channel."""
    if not any(isinstance(v, Mapping) for v in param_sets.values()):
        return param_sets  # single-dict case

    freq = float(ch_ds["frequency_nominal"].compute().item())  # per-frequency case
    key_str = str(int(freq))

    if freq in param_sets:
        return param_sets[freq]

    if key_str in param_sets:
        return param_sets[key_str]

    raise ValueError(f"{freq} Hz parameters missing")
