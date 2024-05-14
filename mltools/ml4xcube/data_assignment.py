import random
import warnings
import xarray as xr
import dask.array as da
from typing import (Sequence, Tuple)
from ml4xcube.cube_utilities import get_chunk_sizes
warnings.filterwarnings('ignore')


def assign_rand_split(ds: xr.Dataset, split: float = 0.8):
    """Assign random split using random sampling."""
    seed = 32  # Consistent seed for reproducibility
    random.seed(seed)

    # Ensure the random array matches the dimensions and chunk sizes of the input dataset
    dimensions = tuple(ds.dims)  # Gets the dimensions from the dataset
    chunk_sizes = tuple(ds.chunks[dim][0] for dim in ds.dims)  # Assumes the dataset is chunked

    # Generate a random array with the same shape and chunking as the dataset
    random_split = da.random.random(size=tuple(ds.dims[dim] for dim in dimensions), chunks=chunk_sizes) < split

    # Assign the new data array to the dataset under the variable name 'split'
    return ds.assign(split=(dimensions, random_split))
      

### dask block sampling

def cantor_pairing(x: int, y: int):
    """unique assignment of a pair (x,y) to a natural number, bijectiv """
    return int((x + y) * (x + y + 1) / 2 + y)


# for random seed generation
def cantor_tuple(index_list: list):
    """unique assignment of a tuple to a natural number, generalization of cantor pairing to tuples"""
    t = index_list[0]
    for x in index_list[1:]:
        t = cantor_pairing(t, x)
    return t


def assign_block_split(ds: xr.Dataset, block_size: Sequence[Tuple[str, int]] = None, split: float = 0.8):
    """Block sampling: add a variable "split" to xarray x, that contains blocks filled with 0 or 1 with frequency split
    Usage:
    xds = xr.open_zarr("***.zarr")
    cube_with_split = assign_block_split(xds, block_size=[("time", 10), ("lat", 20), ("lon", 20)], split=0.5)"""
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
  

       







