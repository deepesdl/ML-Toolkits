import random
import warnings
import xarray as xr
import dask.array as da
from typing import Tuple, List
from ml4xcube.cube_utilities import get_chunk_sizes
warnings.filterwarnings('ignore')


def assign_rand_split(ds: xr.Dataset, split: float = 0.8) -> xr.Dataset:
    """
    Assign random split using random sampling.

    Args:
        ds (xr.Dataset): The xarray dataset to which the random split will be assigned.
        split (float): The proportion of the dataset to be used for training (default is 0.8).

    Returns:
        xr.Dataset: The dataset with an additional 'split' variable indicating the random split.
    """
    seed = 32  # Consistent seed for reproducibility
    random.seed(seed)

    # Fetch the chunk sizes using a predefined method that returns a list of tuples (dimension, chunk size)
    chunk_sizes = dict(get_chunk_sizes(ds))  # Assuming this returns [(dim, size), ...]

    # Create a Dask array that generates random numbers for each data point, following the dataset's chunking structure
    random_split = da.random.random(size=tuple(ds.dims[dim] for dim in ds.dims),
                                    chunks=tuple(chunk_sizes[dim] for dim in ds.dims if dim in chunk_sizes)) < split

    # Convert boolean values to floats (1.0 for True, 0.0 for False)
    random_split = random_split.astype(float)

    # Assign this array to the dataset with the variable name 'split'
    return ds.assign(split=(list(ds.dims), random_split))


def assign_block_split(ds: xr.Dataset, block_size: List[Tuple[str, int]] = None, split: float = 0.8) -> xr.Dataset:
    """
    Assign blocks of data to training or testing sets based on a specified split ratio.

    This function uniquely assigns a block of data to either a training or testing set using a random process.
    The random seed for the assignment is generated using a Cantor pairing function on the block's indices.

    Args:
        ds (xr.Dataset): The input dataset.
        block_size (List[Tuple[str, int]]): List of tuples specifying the dimensions and their respective sizes.
                                            If None, chunk sizes are inferred from the dataset.
        split (float): The fraction of data to assign to the training set. The remainder will be assigned to the testing set.

    Returns:
        xr.Dataset: The input dataset with an additional variable 'split' that indicates whether each block belongs to the training (True) or testing (False) set.
    """
    if block_size is None:
        block_size = get_chunk_sizes(ds)

    def role_dice(x):
        return x + (random.random() < split)

    def block_rand(x):
        block_ind_array = da.zeros(
            (list(x.sizes.values())), chunks=([v for k, v in block_size])
        )
        mapped = block_ind_array.map_blocks(role_dice)
        return ("time", "lat", "lon"), mapped

    return ds.assign({"split": block_rand})








