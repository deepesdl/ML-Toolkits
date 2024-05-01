from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import xarray as xr
import sys
sys.path.append('../mltools')
from mltools.cube_utilities import get_chunk_by_index, calculate_total_chunks
from mltools.preprocessing import apply_filter, drop_nan_values
from typing import Callable
import torch


class XrDataset():
    def __init__(self, ds: xr.Dataset, num_chunks: int, rand_chunk: bool = True,
                 drop_nan: bool = True, strict_nan: bool = False,
                 filter_var: str = 'land_mask', patience: int = 20):
        """
        Initialize xarray dataset.

        Args:
            ds (xr.Dataset): The input dataset.
            num_chunks (int): The number of unique chunks to process.
            rand_chunk (bool): If true, chunks are chosen randomly.
            drop_nan (bool): If true, NaN values are dropped.
            strict_nan (bool): If true, discard the entire chunk if any NaN is found in any variable.
            filter_var (str): The variable to use for filtering. Defaults to 'land_mask'.
                              If None, no filtering is applied.
            patience (int): The number of consecutive iterations without a valid chunk before stopping.

        Returns:
            list: A list of processed data chunks.
        """
        self.ds = ds
        self.rand_chunk = rand_chunk
        self.drop_nan = drop_nan
        self.strict_nan = strict_nan
        self.filter_var = filter_var
        self.patience = patience
        self.total_chunks = calculate_total_chunks(self.ds)

        if self.total_chunks >= num_chunks:
            self.num_chunks = num_chunks
        else:
            self.num_chunks = self.total_chunks

        self.chunks = self.get_chunks()
        self.dataset = self.concatenate_chunks()

    def get_dataset(self):
        return self.dataset

    def concatenate_chunks(self):
        concatenated_chunks = {}

        # Get the keys of the first dictionary in self.chunks
        keys = list(self.chunks[0].keys())

        # Loop over the keys and concatenate the arrays along the time dimension
        for key in keys:
            concatenated_chunks[key] = np.concatenate([chunk[key] for chunk in self.chunks], axis=0)

        return concatenated_chunks

    def preprocess_chunk(self, chunk):
        # Flatten the data and select only land values, then drop NaN values
        cf = {x: chunk[x].ravel() for x in chunk.keys()}

        # Apply filtering based on the specified variable, if provided
        cft = apply_filter(cf, self.filter_var)

        valid_chunk = True

        if self.drop_nan:
            vars = list(cft.keys())
            cft = drop_nan_values(cft, vars)
            valid_chunk = all(np.nan_to_num(cft[var]).sum() > 0 for var in cf)
            if self.strict_nan:
                valid_chunk = any(np.nan_to_num(cft[var]).sum() > 0 for var in cf)

        return cft, valid_chunk

    def get_chunks(self):
        """
        Retrieve specific chunks of data from a dataset.
        Returns:
            list: A list of processed data chunks.
        """

        chunks_idx = list()
        chunks_list = []
        chunk_index = 0
        no_valid_chunk_count = 0
        iterations = 0

        # Process chunks until 3 unique chunks have been processed
        while len(chunks_idx) < self.num_chunks:
            iterations += 1

            if no_valid_chunk_count >= self.patience:
                print("Patience threshold reached, returning collected chunks.")
                break

            if self.rand_chunk:
                chunk_index = random.randint(0, self.total_chunks - 1)  # Select a random chunk index

            if chunk_index in chunks_idx:
                continue  # Skip if this chunk has already been processed

            # Retrieve the chunk by its index
            chunk = get_chunk_by_index(self.ds, chunk_index)

            cft, valid_chunk = self.preprocess_chunk(chunk)

            if valid_chunk:
                chunks_idx.append(chunk_index)
                chunks_list.append(cft)
                no_valid_chunk_count = 0  # reset the patience counter after finding a valid chunk
            else:
                no_valid_chunk_count += 1  # increment the patience counter if no valid chunk is found

            chunk_index += 1
        return chunks_list


class LargeScaleXrDataset(Dataset):
    def __init__(self, xr_dataset, num_chunks, rand_chunk=True, drop_nan=True, filter_var='land_mask'):
        """
        Initialize the dataset to manage large datasets efficiently.

        Args:
            xr_dataset (xr.Dataset): The xarray dataset.
            num_chunks (int): Number of chunks to process dynamically.
            rand_chunk (bool): Whether to select chunks randomly.
            drop_nan (bool): Whether to drop NaN values.
            filter_var (str): Filtering variable name, default 'land_mask'.
        """
        self.ds = xr_dataset
        self.num_chunks = num_chunks
        self.rand_chunk = rand_chunk
        self.drop_nan = drop_nan
        self.filter_var = filter_var
        self.total_chunks = calculate_total_chunks(xr_dataset)
        if self.total_chunks >= num_chunks:
            self.chunk_indices = random.sample(range(self.total_chunks), num_chunks)
        else:
            self.chunk_indices = range(self.total_chunks)

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        chunk_index = self.chunk_indices[idx]

        chunk = get_chunk_by_index(self.ds, chunk_index)

        # Process the chunk
        cf = {x: chunk[x].ravel() for x in chunk.keys()}
        cft = apply_filter(cf, self.filter_var)

        if self.drop_nan:
            cft = drop_nan_values(cft, list(cft.keys()))

        return cft  # Return the processed chunk



def prepare_dataloader(dataset: Dataset, batch_size: int, callback_fn: Callable = None, num_workers: int = 0, parallel: bool = False, shuffle = True) -> DataLoader:
    """
    Prepares a DataLoader.

    Parameters:
    - dataset: The dataset from which to load the data.
    - batch_size: How many samples per batch to load.
    - callback_fn: A function used to collate data into batches.
    - num_workers: How many subprocesses to use for data loading.
    - parallel: Specifies if distributed training is performed.

    Returns:
    A DataLoader object.
    """
    sampler = None
    if parallel:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,  # Conditionally use pin_memory
        shuffle=shuffle,
        collate_fn=callback_fn,
        sampler=sampler
    )
