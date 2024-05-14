import itertools
import rechunker
import numpy as np
import xarray as xr
from typing import Sequence, Tuple, Iterator, Dict


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


def calculate_total_chunks(ds: xr.Dataset) -> int:
    """Calculate the total number of chunks for the dataset based on maximum chunk sizes."""
    block_sizes = get_chunk_sizes(ds)

    total_chunks = np.prod([
        len(range(0, ds.dims[dim_name], size))
        for dim_name, size in block_sizes
    ])

    return total_chunks


def get_chunk_by_index(ds: xr.Dataset, index: int, block_sizes: Sequence[Tuple[str, int]] = None) -> Dict[
    str, np.ndarray]:
    """
    Returns a specific data chunk from an xarray.Dataset by index.

    Parameters:
    - ds: The xarray.Dataset from which to retrieve a chunk.
    - index: The linear index of the chunk to retrieve.
    - block_sizes: An optional sequence of tuples specifying the block size for each dimension.
                   Each tuple should contain a dimension name and a block size for that dimension.
                   If not provided, the function will attempt to use the dataset's chunk sizes.

    Returns:
    A dictionary where keys are variable names and values are the chunk data as numpy arrays.
    """
    if block_sizes is None:
        # Assume a function `get_chunk_sizes(ds)` to determine the block sizes from the dataset
        block_sizes = get_chunk_sizes(ds)

    # Calculate the total number of chunks along each dimension
    dim_chunks = [range(0, ds.dims[dim_name], size) for dim_name, size in block_sizes]
    total_chunks_per_dim = [len(list(chunks)) for chunks in dim_chunks]

    # Convert the linear index to a multi-dimensional index
    multi_dim_index = np.unravel_index(index, total_chunks_per_dim)

    # Calculate the slice for each dimension based on the multi-dimensional index
    dim_slices = {}
    for dim_idx, (dim_name, block_size) in enumerate(block_sizes):
        start = multi_dim_index[dim_idx] * block_size
        end = min(start + block_size, ds.dims[dim_name])
        dim_slices[dim_name] = slice(start, end)

    # Extract the chunk for each variable
    var_blocks = {}
    for var_name, var in ds.data_vars.items():
        # Determine the slices applicable to this variable
        indexers = {dim_name: dim_slice for dim_name, dim_slice in dim_slices.items() if dim_name in var.dims}
        # Extract the chunk using variable-specific indexers
        var_blocks[var_name] = var.isel(indexers).values

    return var_blocks


def rechunk_cube(source_cube: xr.DataArray, target_chunks: dict | tuple | list, target_path: str):
    """
    Rechunks an xarray DataArray to a new chunking scheme.

    Parameters:
    - source_cube: xr.DataArray
      The input DataArray that you want to rechunk.

    - target_chunks: dict | tuple | list
      The desired chunk sizes for the rechunking operation.
      If a dict, specify sizes for each named dimension, e.g., {'lon': 60, 'lat': 1, 'time': 100}.
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