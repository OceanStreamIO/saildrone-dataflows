import numpy as np
import xarray as xr
import dask.array as da
from typing import Union

def downsample(dataset, coordinates: {str: int}, operation: str = "mean", is_log: bool = False):
    """
    Given a dataset, downsamples it on the specified coordinates

    Args:
        dataset (xr.DataArray)  : the dataset to resample
        coordinates({str: int}  : a mapping of dimensions to the windows to use
        operation (str)         : the downsample operation to use
        is_log (bool)           : True if the data is logarithmic and should be
                                    converted to linear

    Returns:
        xr.DataArray            : the resampled dataset
    """
    operation_list = ["mean", "sum"]
    if operation not in operation_list:
        raise Exception("Operation not in approved list")
    for k in coordinates.keys():
        if k not in dataset.dims:
            raise Exception("Coordinate " + k + " not in dataset coordinates")
    if is_log:
        dataset = lin(dataset)

    if operation == "mean":
        dataset = dataset.coarsen(coordinates, boundary="pad").mean()
    elif operation == "sum":
        dataset = dataset.coarsen(coordinates, boundary="pad").sum()
    else:
        raise Exception("Operation not in approved list")

    if is_log:
        dataset = log(dataset)

    return dataset


def upsample(dataset: xr.DataArray, dataset_size: xr.DataArray):
    """
    Given a data dataset and an example dataset, upsamples the data dataset
    to the example dataset's dimensions by repeating values

    Args:
        dataset (xr.DataArray)      : data
        dataset_size (xr.DataArray) : dataset of the right size

    Returns
        xr.DataArray: the input dataset, with the same coords as dataset_size
        and the values repeated to fill it up.
    """

    interpolated = dataset.interp_like(dataset_size, method="nearest")
    return interpolated


def log(linear: xr.DataArray, parallelized=True) -> xr.DataArray:
    """
    Turn variable into the logarithmic domain. This function will return -999
    in the case of values less or equal to zero (undefined logarithm). -999 is
    the convention for empty water or vacant sample in fisheries acoustics.

    Args:
        variable (float): array of elements to be transformed.

    Returns:
        float: array of elements transformed
    """
    back_list = False
    back_single = False
    if not isinstance(linear, xr.DataArray):
        if isinstance(linear, list):
            linear = xr.DataArray(linear)
            back_list = True
        else:
            linear = xr.DataArray([linear])
            back_single = True

    if parallelized:
        db = xr.apply_ufunc(
            lambda x: 10 * np.log10(x),
            linear,
            dask="parallelized",
            vectorize=True,
            output_dtypes=[np.float64],
        )
    else:
        db = xr.apply_ufunc(lambda x: 10 * np.log10(x), linear)

    db = xr.where(db.isnull(), -999, db)
    db = xr.where(linear == 0, -999, db)

    if back_list:
        db = db.values
    if back_single:
        db = db.values[0]
    return db


def lin(db: xr.DataArray) -> xr.DataArray:
    """Convert decibel to linear scale, handling NaN values."""
    linear = xr.where(db.isnull(), np.nan, 10 ** (db / 10))
    return linear


def line_to_square(one: xr.DataArray, two: xr.DataArray, dim: str) -> np.ndarray:
    """
    Given a single dimension dataset and an example dataset with 2 dimensions,
    returns a two-dimensional dataset that is the single dimension dataset
    repeated as often as needed.

    Args:
        one (xr.DataArray): data
        two (xr.DataArray): shape dataset
        dim (str): name of dimension to concat against

    Returns:
        np.ndarray: The input dataset values repeated to match the shape of two
    """
    length = len(two[dim])
    
    if isinstance(one.data, da.Array):
        return da.repeat(one.data[..., np.newaxis], length, axis=-1)

    return np.repeat(one.values[..., np.newaxis], length, axis=-1)


def block_nanmedian(block: Union[np.ndarray, da.Array], i: int, n: int, axis: int) -> float:
    """
    Calculate block median efficiently for both numpy and dask arrays.
    
    Args:
        block: Input array (numpy or dask)
        i: Current index
        n: Window half size
        axis: Axis along which to take median
    
    Returns:
        float: Median value for the block
    """
    start = max(0, i - n)
    end = min(block.shape[axis], i + n + 1)
    
    if isinstance(block, da.Array):
        indices = da.arange(start, end, dtype=int)
        use_block = da.take(block, indices, axis)
        return float(da.nanmedian(use_block).compute())
    else:
        indices = np.arange(start, end)
        use_block = np.take(block, indices, axis=axis)
        return float(np.nanmedian(use_block))


def rolling_median_block(block: Union[np.ndarray, da.Array], window_half_size: int, axis: int) -> np.ndarray:
    """
    Applies a median block calculation as a rolling function across an axis.
    
    Args:
        block: Input array (numpy or dask)
        window_half_size: Size of the rolling window half width
        axis: Axis along which to apply rolling median
        
    Returns:
        np.ndarray: Array of rolling median values
    """
    shape = block.shape[axis]
    if isinstance(block, da.Array):
        # Process in chunks for better performance with dask
        chunk_size = block.chunks[axis][0]  # Get the chunk size along the axis
        results = []
        for i in range(0, shape, chunk_size):
            end = min(i + chunk_size, shape)
            chunk_results = [block_nanmedian(block, j, window_half_size, axis) 
                           for j in range(i, end)]
            results.extend(chunk_results)

        return np.array(results)
    
    return np.array([block_nanmedian(block, i, window_half_size, axis) for i in range(shape)])


def dask_nanmedian(array: Union[xr.DataArray, np.ndarray, da.Array], axis: int = None) -> da.Array:
    """
    Compute nanmedian for various array types, converting to dask array if needed.
    
    Args:
        array: Input array (xarray DataArray, numpy array, or dask array)
        axis: Axis along which to compute median
        
    Returns:
        da.Array: Dask array containing the median values
    """
    if isinstance(array, xr.DataArray):
        data = array.data
    else:
        data = array
        
    if isinstance(data, np.ndarray):
        data = da.from_array(data)
    elif not isinstance(data, da.Array):
        raise TypeError(f"Cannot convert type {type(data)} to dask array")
        
    return da.nanmedian(data, axis=axis)


def dask_nanmean(array: Union[xr.DataArray, np.ndarray, da.Array], axis: int = None) -> da.Array:
    """
    Compute nanmean for various array types, converting to dask array if needed.
    
    Args:
        array: Input array (xarray DataArray, numpy array, or dask array)
        axis: Axis along which to compute mean
        
    Returns:
        da.Array: Dask array containing the mean values
    """
    if isinstance(array, xr.DataArray):
        data = array.data
    else:
        data = array
        
    if isinstance(data, np.ndarray):
        data = da.from_array(data)
    elif not isinstance(data, da.Array):
        raise TypeError(f"Cannot convert type {type(data)} to dask array")
        
    return da.nanmean(data, axis=axis)
