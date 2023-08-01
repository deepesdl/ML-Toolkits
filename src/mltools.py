# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
# 
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import xarray as xr
import dask.array as da
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import itertools
from typing import (Callable, Dict, Iterator,
                    Sequence, Tuple, TypeVar, Union)

import warnings
warnings.filterwarnings('ignore')



def getRange(ds: xr.Dataset, var: str):
    """returns min and max values of variable var of xarray ds"""
    x = ds[var].min().values
    y = ds[var].max().values
    return [x,y]

def getStatistics(ds: xr.Dataset, var: str):
    """returns mean and std values of variable var of xarray ds"""
    x = ds[var].mean().values
    y = ds[var].std().values
    return [x,y]

def normalize(x: np.ndarray, xmin: float, xmax: float):
    """min-max feature scaling of x, shift values to range [0,1]"""
    return (x - xmin ) / (xmax - xmin)


def standardize(x: np.ndarray, xmean: float, xstd: float):
    """transforms distribution to mean 0 and variance 1"""
    return (x - xmean ) / xstd

def undo_normalizing(x: np.ndarray, xmin: float, xmax: float):
    """inverse operation of normalization"""
    return x*(xmax - xmin) + xmin

def undo_standardizing(x: np.ndarray, xmean: float, xstd: float):
    """inverse operation of standardization"""
    return x * xstd + xmean


def rand(x: xr.Dataset):
    """assign random split"""
    da.random.seed(32)
    return ("time", "lat", "lon"), da.random.random((list(x.dims.values())), chunks=([v for k,v in get_chunk_sizes(x)])) < 0.8
      
    
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
  

       
### pytorch training

def train_one_epoch(epoch_index: int, training_loader: torch.utils.data.DataLoader, model: torch.nn.Module, loss_fn: torch.nn.modules.loss, optimizer: torch.optim, device:str):
    """pytorch model training, training of one epoch"""
    running_loss = 0.
    last_loss = 0.
    train_pred = np.empty(0)
    for i, data in enumerate(training_loader):
        inputs, labels = data
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        X = inputs.unsqueeze(1)
        X = X.to(device)
        model = model.to(device)
        outputs = model(X)
        train_pred = np.append(train_pred, outputs.cpu().detach().numpy().ravel())

        l = labels.unsqueeze(1)
        l = l.to(device)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, l)

        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            running_loss = 0.

    return model, train_pred, last_loss


def test(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, loss_fn: torch.nn.modules.loss, device: str):
    """pytorch model testing"""
    test_pred = np.empty(0)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.unsqueeze(1)
            y = y.unsqueeze(1)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_pred= np.append(test_pred, pred.cpu().detach().numpy().ravel())
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    return test_pred, test_loss
    
### chunk iteration
    
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


M = TypeVar('M')
R = TypeVar('R')



