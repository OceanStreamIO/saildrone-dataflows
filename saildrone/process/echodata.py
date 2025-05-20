from saildrone.store import open_converted as open_from_blobstorage


def open_echodata(source_path=None, container_name=None, zarr_path=None, chunks=None):
    if source_path is not None:
        from echopype.echodata.api import open_converted

        return open_converted(source_path, chunks=chunks)

    return open_from_blobstorage(zarr_path, container_name=container_name, chunks=chunks)
