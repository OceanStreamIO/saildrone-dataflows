import xarray as xr


def create_multichannel_mask(masks: [xr.Dataset], source_Ds: xr.Dataset) -> xr.Dataset:
    channel_list = source_Ds["channel"].values

    if len(masks) != len(channel_list):
        raise ValueError("number of masks and of channels provided should be the same")

    updated_masks = []
    for i in range(0, len(masks)):
        mask = masks[i]

        if "channel" not in mask.dims:
            mask = mask.expand_dims(dim={"channel": [channel_list[i]]})
        else:
            # Drop extra channels if present
            mask = mask.isel(channel=0).expand_dims(dim={"channel": [channel_list[i]]})

        updated_masks.append(mask)

    result = xr.concat(updated_masks, dim="channel", data_vars="all", coords="all", join="exact")
    return result
