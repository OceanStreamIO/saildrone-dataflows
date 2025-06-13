import xarray as xr


def fix_chunking(ds: xr.Dataset, *, tiny_limit = 10_000):
    """
    Harmonise Zarr‐chunk hints with current Dask chunking.

    • For variables with < `tiny_limit` elements:
        → compute to NumPy  (one scalar / small vector, no memory penalty)
    • For the rest: drop an incompatible `encoding["chunks"]`.
    """
    ds = ds.copy()

    for name, var in list(ds.variables.items()):
        # case A: tiny array → compute to NumPy
        if var.size <= tiny_limit:
            ds[name] = (var.dims, var.compute().data)
            ds[name].encoding.clear()
            continue

        # case B: larger array → keep lazy but fix the hint if it mismatches
        if "chunks" in var.encoding:
            dask_chunks = getattr(var.data, "chunks", None)

            # If the variable is NumPy *or* the hint disagrees → drop it.
            if (dask_chunks is None or
                    var.encoding["chunks"] != tuple(c[0] for c in dask_chunks)):
                var.encoding.pop("chunks", None)

    return ds


def get_variable_encoding(ds: xr.Dataset, compression_level):
    """Generate encoding dictionary for dataset variables."""
    encoding = {}
    for var in ds.data_vars:
        if ds[var].dtype.kind in {"U", "S", "O"}:  # String or object types
            # No compression or chunking for unsupported types
            encoding[var] = {}
        else:
            # Apply compression for numeric types
            encoding[var] = {
                "zlib": True,
                "complevel": compression_level,
            }
    return encoding
