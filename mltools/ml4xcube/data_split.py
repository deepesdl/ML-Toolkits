import random
import warnings
import xarray as xr
import dask.array as da
from typing import Optional, Tuple, List
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
    dimensions = list(ds.dims)  # Gets the dimensions from the dataset
    chunk_sizes = {dim: ds.chunks[dim] for dim in ds.dims}  # Assumes the dataset is chunked

    # Generate a random array with the same shape and chunking as the dataset
    random_split = da.random.random(size=tuple(ds.dims[dim] for dim in dimensions), chunks=tuple(chunk_sizes[dim] for dim in dimensions)) < split

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


def assign_block_split(ds: xr.Dataset, block_size: Optional[List[Tuple[str, int]]] = None, split: float = 0.8) -> xr.Dataset:
    """
    Unique assignment of a tuple to a natural number, generalization of Cantor pairing to tuples.

    Args:
        index_list (List[int]): A list of integers representing the tuple.

    Returns:
        int: The unique natural number assigned to the tuple.
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
            (list(x.dims.values())), chunks=([v for k, v in block_size])
        )
        mapped = block_ind_array.map_blocks(role_dice)
        return ("time", "lat", "lon"), mapped

    return ds.assign({"split": block_rand})








