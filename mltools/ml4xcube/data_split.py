import random
import warnings
import xarray as xr
import dask.array as da
from typing import Tuple, List
from ml4xcube.cube_utilities import get_chunk_sizes
warnings.filterwarnings('ignore')


# random sampling

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

    # Ensure the random array matches the dimensions and chunk sizes of the input dataset
    dimensions = list(ds.sizes)  # Gets the dimensions from the dataset
    chunk_sizes = {dim: ds.chunks[dim] for dim in ds.sizes}  # Assumes the dataset is chunked

    # Generate a random array with the same shape and chunking as the dataset
    random_split = da.random.random(size=tuple(ds.sizes[dim] for dim in dimensions), chunks=tuple(chunk_sizes[dim] for dim in dimensions)) < split

    # Convert boolean values to floats
    random_split = random_split.astype(float)

    # Assign the new data array to the dataset under the variable name 'split'
    return ds.assign(split=(dimensions, random_split))
      

# block sampling

def cantor_pairing(x: int, y: int) -> int:
    """
    Unique assignment of a pair (x,y) to a natural number using the Cantor pairing function.

    Args:
        x (int): The first integer.
        y (int): The second integer.

    Returns:
        int: The unique natural number assigned to the pair (x, y).
    """
    return int((x + y) * (x + y + 1) / 2 + y)


# for random seed generation
def cantor_tuple(index_list: List[int]) -> int:
    """
    Unique assignment of a pair (x,y) to a natural number using the Cantor pairing function.

    Args:
        x (int): The first integer.
        y (int): The second integer.

    Returns:
        int: The unique natural number assigned to the pair (x, y).
    """
    t = index_list[0]
    for x in index_list[1:]:
        t = cantor_pairing(t, x)
    return t


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

    def role_dice(x, block_id=None):
        if block_id is not None:
            seed=cantor_tuple(block_id)
            random.seed(seed)
        return x + (random.random() < split)

    def block_rand(x):
        block_ind_array = da.zeros(
            (list(x.sizes.values())), chunks=([v for k, v in block_size])
        )
        mapped = block_ind_array.map_blocks(role_dice)
        return ("time", "lat", "lon"), mapped

    return ds.assign({"split": block_rand})








