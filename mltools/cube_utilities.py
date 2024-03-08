import itertools
from typing import Sequence, Tuple, Iterator, Dict
import rechunker
import numpy as np
import xarray as xr


def get_chunk_sizes(ds: xr.Dataset) -> Sequence[Tuple[str, int]]:
    """Determine maximum chunk sizes of all data variables of dataset *ds*.
    Helper function.
    """
    chunk_sizes = {}
    for var in ds.data_vars.values():
        if var.chunks:
            chunks = tuple(max(*c) if len(c) > 1 else c[0]
                           for c in var.chunks)
            for dim_name, chunk_size in zip(var.dims, chunks):
                chunk_sizes[dim_name] = max(chunk_size,
                                            chunk_sizes.get(dim_name, 0))
    return [(str(k), v) for k, v in chunk_sizes.items()]


def iter_data_var_blocks(ds: xr.Dataset,
                         block_sizes: Sequence[Tuple[str, int]] = None) \
        -> Iterator[Dict[str, np.ndarray]]:
    """Create an iterator that will provide all data blocks of all data
    variables of given dataset *ds*.

    The data blocks' order and shapes are predescribed
    by *block_sizes* argument, which is a seqence comprising
    dimension name and block size pairs. If *block_size is not given,
    the chunk sizes of data variables are used instead.
    """
    block_sizes = get_chunk_sizes(ds) if block_sizes is None else block_sizes
    dim_ranges = []
    for dim_name, chunk_size in block_sizes:
        dim_size = ds.dims[dim_name]
        dim_ranges.append(range(0, dim_size, chunk_size))
    for offsets in itertools.product(*dim_ranges):
        dim_slices = {block_size[0]: slice(offset, offset + block_size[1])
                      for block_size, offset in zip(block_sizes, offsets)}
        var_blocks = {}
        for var_name, var in ds.data_vars.items():
            indexers = {dim_name: dim_slice
                        for dim_name, dim_slice in dim_slices.items()
                        if dim_name in var.dims}
            var_blocks[var_name] = var.isel(indexers).values
        yield var_blocks


def rechunk_cube(source_cube: xr.DataArray, target_chunks: dict | tuple | list, target_path: str):
    """
    Rechunks an xarray DataArray to a new chunking scheme.

    Parameters:
    - source_cube: xr.DataArray
      The input DataArray that you want to rechunk.

    - target_chunks: dict | tuple | list
      The desired chunk sizes for the rechunking operation.
      If a dict, specify sizes for each named dimension, e.g., {'lon': 60, 'lat': 1, 'time': 101}.
      If a tuple or list, specify sizes by order, corresponding to the array's dimensions.

    - target_path: str
      The path where the rechunked DataArray should be stored, typically a path to a Zarr store.

    Returns:
    Nothing, but prints a message upon successful completion.
    """

    # Validate target_chunks input
    if not isinstance(target_chunks, (dict, tuple, list)):
        raise ValueError("target_chunks must be a dictionary, tuple, or list")

    # Create a rechunking plan
    rechunk_plan = rechunker.rechunk(source_cube, target_chunks, target_path, temp_store=None)

    # Execute the rechunking
    rechunk_plan.execute()

    print("Rechunking completed successfully.")
