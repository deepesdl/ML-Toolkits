import itertools
import rechunker
import numpy as np
import xarray as xr
import dask.array as da
from typing import Tuple, Iterator, Dict, List


def get_chunk_sizes(ds: xr.Dataset) -> List[Tuple[str, int]]:
    """
    Determine maximum chunk sizes of all data variables of the dataset.

    Args:
        ds (xr.Dataset): The xarray dataset.

    Returns:
        List[Tuple[str, int]]: A list of tuples where each tuple contains a dimension name and its maximum chunk size.
    """
    chunk_sizes = {}
    for var in ds.data_vars.values():
        if var.chunks:
            chunks = tuple(max(*c) if len(c) > 1 else c[0]
                           for c in var.chunks)
            for dim_name, chunk_size in zip(var.sizes, chunks):
                chunk_sizes[dim_name] = max(chunk_size,
                                            chunk_sizes.get(dim_name, 0))
    return [(str(k), v) for k, v in chunk_sizes.items()]


def iter_data_var_blocks(ds: xr.Dataset, block_size: List[Tuple[str, int]] = None) \
        -> Iterator[Dict[str, np.ndarray]]:
    """
    Create an iterator that provides all data blocks of all data variables of the given dataset.

    Args:
        ds (xr.Dataset): The xarray dataset.
        block_size (List[Tuple[str, int]]): A sequence comprising dimension name and block size pairs.
            If not given, the chunk sizes of data variables are used instead.

    Yields:
        Iterator[Dict[str, np.ndarray]]: An iterator of dictionaries where keys are variable names and values are data blocks as numpy arrays.
    """
    block_size = get_chunk_sizes(ds) if block_size is None else block_size
    dim_ranges = []
    for dim_name, chunk_size in block_size:
        dim_size = ds.sizes[dim_name]
        dim_ranges.append(range(0, dim_size, chunk_size))
    for offsets in itertools.product(*dim_ranges):
        dim_slices = {block_size[0]: slice(offset, offset + block_size[1])
                      for block_size, offset in zip(block_size, offsets)}
        var_blocks = {}
        for var_name, var in ds.data_vars.items():
            indexers = {dim_name: dim_slice
                        for dim_name, dim_slice in dim_slices.items()
                        if dim_name in var.sizes}
            var_blocks[var_name] = var.isel(indexers).values
        yield var_blocks


def calculate_total_chunks(ds: xr.Dataset, block_size: List[Tuple[str, int]] = None) -> int:
    """
    Calculate the total number of chunks for the dataset based on maximum chunk sizes.

    Args:
        ds (xr.Dataset): The xarray dataset.
        block_size (List[Tuple[str, int]]): A sequence of tuples specifying the block size for each dimension.
            If not provided, the function will use the dataset's chunk sizes.

    Returns:
        int: The total number of chunks.
    """
    default_block_sizes = get_chunk_sizes(ds)

    if block_size is not None:
        # Replace the sizes which are not None
        for i, (dim, size) in enumerate(block_size):
            if size is not None:
                default_block_sizes[i] = (dim, size)

    block_size = default_block_sizes

    total_chunks = np.prod([
        len(range(0, ds.sizes[dim_name], size))
        for dim_name, size in block_size
    ])

    return total_chunks


def get_chunk_by_index(ds: xr.Dataset, index: int, block_size: List[Tuple[str, int]] = None) -> Dict[
    str, np.ndarray]:
    """
    Returns a specific data chunk from an xarray.Dataset by index.

    Args:
        ds (xr.Dataset): The xarray.Dataset from which to retrieve a chunk.
        index (int): The linear index of the chunk to retrieve.
        block_size (List[Tuple[str, int]]): An optional sequence of tuples specifying the block size for each dimension.
            Each tuple should contain a dimension name and a block size for that dimension.
            If not provided, the function will attempt to use the dataset's chunk sizes.

    Returns:
        Dict[str, np.ndarray]: A dictionary where keys are variable names and values are the chunk data as numpy arrays.
    """
    # Get the default chunk sizes from the dataset
    default_block_sizes = get_chunk_sizes(ds)

    if block_size is not None:
        # Replace the sizes which are not None
        for i, (dim, size) in enumerate(block_size):
            if size is not None:
                default_block_sizes[i] = (dim, size)

    block_size = default_block_sizes

    # Calculate the total number of chunks along each dimension
    dim_chunks = [range(0, ds.sizes[dim_name], size) for dim_name, size in block_size]
    total_chunks_per_dim = [len(list(chunks)) for chunks in dim_chunks]

    # Convert the linear index to a multi-dimensional index
    multi_dim_index = np.unravel_index(index, total_chunks_per_dim)

    # Calculate the slice for each dimension based on the multi-dimensional index
    dim_slices = {}
    for dim_idx, (dim_name, block_size) in enumerate(block_size):
        start = multi_dim_index[dim_idx] * block_size
        end = min(start + block_size, ds.sizes[dim_name])
        dim_slices[dim_name] = slice(start, end)

    # Extract the chunk for each variable
    var_blocks = {}
    for var_name, var in ds.data_vars.items():
        # Determine the slices applicable to this variable
        indexers = {dim_name: dim_slice for dim_name, dim_slice in dim_slices.items() if dim_name in var.sizes}
        # Extract the chunk using variable-specific indexers
        var_blocks[var_name] = var.isel(indexers).values

    return var_blocks


def rechunk_cube(source_cube: xr.DataArray, target_chunks: Dict[str, int] | Tuple[int] | List[int], target_path: str):
    """
    Rechunks an xarray DataArray to a new chunking scheme.

    Args:
        source_cube (xr.DataArray): The input DataArray that you want to rechunk.
        target_chunks (Dict | Tuple | List): The desired chunk sizes for the rechunking operation.
            If a dict, specify sizes for each named dimension, e.g., {'lon': 60, 'lat': 1, 'time': 100}.
            If a tuple or list, specify sizes by order, corresponding to the array's dimensions.
        target_path (str): The path where the rechunked DataArray should be stored, typically a path to a Zarr store.

    Returns:
        None: Prints a message upon successful completion.
    """
    # Validate target_chunks input
    if not isinstance(target_chunks, (dict, tuple, list)):
        raise ValueError("target_chunks must be a dictionary, tuple, or list")

    # Create a rechunking plan
    rechunk_plan = rechunker.rechunk(source_cube, target_chunks, target_path, temp_store=None)

    # Execute the rechunking
    rechunk_plan.execute()

    print("Rechunking completed successfully.")


def split_chunk(chunk: Dict[str, np.ndarray], sample_size: List[Tuple[str, int]] = None,
                overlap: List[Tuple[str, float]] = None) -> Dict[str, np.ndarray]:
    """
    Split a chunk into points based on provided indices.

    Args:
        chunk (Dict[str, np.ndarray]): The chunk to split.
        sample_size (List[Tuple[str, int]]): Sizes of the samples to be extracted from the chunk along each dimension.
                                             Each tuple contains the dimension name and the size along that dimension.
                                             If None the chunk is split into points.
        overlap (List[Tuple[str, float]]): Overlap for overlapping samples while chunk splitting.
                                             Each tuple contains the dimension name and the overlap fraction along that dimension.
                                             floats between 0 and 1.

    Returns:
        Dict[str, np.ndarray]: A dictionary where keys are variable names and values are the extracted points as numpy arrays.
    """

    if sample_size is None: return {x: chunk[x].ravel() for x in chunk.keys()}
    else: cf = {x: chunk[x] for x in chunk.keys()}

    # Extract the step sizes from the sample_size
    step_sizes = [step for _, step in sample_size]

    # Handle overlaps
    if overlap:
        overlap_steps = [
            int(step * overlap_frac) if step > 1 else 0
            for (_, step), (_, overlap_frac) in zip(sample_size, overlap)
        ]
    else:
        overlap_steps = [0] * len(step_sizes)

    # Extract the shapes of the arrays in the chunk
    shape = next(iter(cf.values())).shape  # Assuming all arrays have the same shape

    # Calculate the number of splits for each dimension
    num_splits = [
        (shape[i] - step_sizes[i]) // (step_sizes[i] - overlap_steps[i]) + 1
        for i in range(len(step_sizes))
    ]

    # Calculate the total number of resulting points
    total_points = np.prod(num_splits)

    # Initialize the result dictionary with the expected shape
    result = {}
    for key in cf.keys():
        if step_sizes[0] == 1:
            result[key] = np.zeros((total_points, *step_sizes[1:]), dtype=cf[key].dtype)
        else:
            result[key] = np.zeros((total_points, *step_sizes), dtype=cf[key].dtype)

    # Iterate through all possible splits
    point_idx = 0

    for time_idx in range(0, shape[0], step_sizes[0] - overlap_steps[0]):
        if time_idx + step_sizes[0] > shape[0]: continue
        for lat_idx in range(0, shape[1], step_sizes[1] - overlap_steps[1]):
            if lat_idx + step_sizes[1] > shape[1]: continue
            for lon_idx in range(0, shape[2], step_sizes[2] - overlap_steps[2]):
                if lon_idx + step_sizes[2] > shape[2]: continue
                for key in cf.keys():
                    result[key][point_idx] = cf[key][time_idx:time_idx + step_sizes[0],
                                             lat_idx:lat_idx + step_sizes[1],
                                             lon_idx:lon_idx + step_sizes[2]]
                point_idx += 1

    return result


def assign_dims(data: Dict[str, da.Array|xr.DataArray], dims: Tuple) -> Dict[str, xr.DataArray]:
    """
    Assign dimensions to each variable in the dataset based on provided dimension names.

    Args:
        data (Dict[str, da.Array | xr.DataArray]): A dictionary where keys are variable names and values are dask arrays or xarray DataArrays.
        dims (Tuple): A tuple of dimension names to assign to the DataArrays.

    Returns:
        Dict[str, xr.DataArray]: A dictionary where keys are variable names and values are xarray DataArrays with assigned dimensions.
    """
    result = {}
    for var, dask_array in data.items():
        if len(dims) >= dask_array.ndim:
            result[var] = xr.DataArray(dask_array, dims=dims[:dask_array.ndim])
    return result


def get_dim_range(cube: xr.DataArray, dim: str):
    """
    Calculates the dimension range of an xr.DataArray.

    Args:
        cube (xr.DataArray): The input data cube.
        dim (str): The dimension name.

    Returns:
        tuple: The minimum and maximum values of the dimension.
    """
    try:
        if np.issubdtype(cube[dim].dtype, np.datetime64):
            min_val = np.datetime_as_string(cube[dim].values.min(), unit='D')
            max_val = np.datetime_as_string(cube[dim].values.max(), unit='D')
        else:
            min_val = cube[dim].values.min()
            max_val = cube[dim].values.max()
    except:
        min_val = cube[dim].values.min()
        max_val = cube[dim].values.max()
    return min_val, max_val