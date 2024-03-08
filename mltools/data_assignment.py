import xarray as xr
import dask.array as da
import random
from typing import (Sequence, Tuple)
import warnings
from MLTools.cube_utilities import get_chunk_sizes

warnings.filterwarnings('ignore')


def rand(x: xr.Dataset):
    """assign random split"""
    da.random.seed(32)
    res = ("time", "lat", "lon"), da.random.random((list(x.dims.values())), chunks=([v for k,v in get_chunk_sizes(x)])) < 0.8
    print(res)
    return res
      
    
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


def assign_split(x: xr.Dataset, block_size: Sequence[Tuple[str, int]] = None, split: float = 0.8):
    """Block sampling: add a variable "split" to xarray x, that contains blocks filled with 0 or 1 with frequency split
    Usage:
    xds = xr.open_zarr("***.zarr")
    cube_with_split = assign_split(xds, block_size=[("time", 10), ("lat", 20), ("lon", 20)], split=0.5)"""
    if block_size is None:
        block_size = get_chunk_sizes(x)

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

    return x.assign({"split": block_rand})
  

       







