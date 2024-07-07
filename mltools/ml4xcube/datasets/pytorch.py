import torch
import random
import numpy as np
import xarray as xr
from ml4xcube.cube_utilities import split_chunk
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Callable, Dict
from ml4xcube.cube_utilities import calculate_total_chunks, get_chunk_by_index
from ml4xcube.preprocessing import apply_filter, drop_nan_values, fill_masked_data


class LargeScaleXrDataset(Dataset):
    def __init__(self, xr_dataset: xr.Dataset, rand_chunk: bool = True, drop_nan: bool = True,
                 chunk_indices: List[int] = None, drop_nan_masked: bool = False, use_filter: bool = True,
                 drop_sample: bool = False, fill_method: str = None, const: float = None,
                 filter_var: str = 'land_mask', num_chunks: int = None,
                 block_sizes: List[Tuple[str, int]] = None,
                 sample_size: List[Tuple[str, int]] = None,
                 overlap: List[Tuple[str, int]] = None):
        """
        Initialize the dataset to manage large datasets efficiently.

        Attributes:
            ds (xr.Dataset): The xarray dataset.
            rand_chunk (bool): Whether to select chunks randomly.
            drop_nan (bool): Whether to drop NaN values.
            drop_nan_masked (bool): If true, NaN values are dropped using the mask specified by filter_var.
            use_filter (bool): If true, apply the filter based on the specified filter_var.
            drop_sample (bool): If true, drop the entire subarray if any value in the subarray does not belong to the mask (False).
            fill_method (str): Method to fill masked data, if any.
            const (float): Constant value to use for filling masked data, if needed.
            filter_var (str): Filtering variable name.
            num_chunks (int): Number of chunks to process dynamically.
            block_sizes (List[Tuple[str, int]]): Block sizes for considered blocks (of (sub-)chunks).
            sample_size (List[Tuple[str, int]]): Sample size for chunk splitting.
            overlap (List[Tuple[str, int]]): Overlap for overlapping samples due to chunk splitting.
            total_chunks (int): Total number of chunks in the dataset.
            chunk_indices (List[int]): List of indices of chunks to load.
        """
        self.ds = xr_dataset
        self.rand_chunk = rand_chunk
        self.drop_nan = drop_nan
        self.drop_nan_masked = drop_nan_masked
        self.use_filter = use_filter
        self.drop_sample = drop_sample
        self.fill_method = fill_method
        self.const = const
        self.filter_var = filter_var
        self.num_chunks = num_chunks
        self.block_sizes = block_sizes
        self.sample_size = sample_size
        self.overlap = overlap

        self.total_chunks = int(calculate_total_chunks(xr_dataset, self.block_sizes))
        if not chunk_indices is None:
            self.chunk_indices = chunk_indices
        elif num_chunks is not None and self.total_chunks >= num_chunks:
            self.chunk_indices = random.sample(range(self.total_chunks), num_chunks)
        else:
            self.chunk_indices = range(self.total_chunks)

    def __len__(self) -> int:
        """
        Return the total number of chunks.

        Returns:
            int: Number of chunks.
        """
        return len(self.chunk_indices)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Retrieve a chunk by its index and preprocess it.

        Args:
            idx (int): Index of the chunk to retrieve.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the processed chunk.
        """
        chunk_index = self.chunk_indices[idx]

        chunk = get_chunk_by_index(self.ds, chunk_index)

        # Process the chunk
        cf = split_chunk(chunk, self.sample_size, overlap=self.overlap)

        if self.use_filter:
            cft = apply_filter(cf, self.filter_var, self.drop_sample)
        else:
            cft = cf

        if self.drop_nan:
            vars = list(cft.keys())
            if self.drop_nan_masked:
                cft = drop_nan_values(cft, vars, self.filter_var)
            else:
                cft = drop_nan_values(cft, vars)

        valid_chunk = all(np.nan_to_num(cft[var]).sum() > 0 for var in cf)

        if valid_chunk:
            vars = [var for var in cft.keys() if var != 'split' and var != self.filter_var]
            if self.fill_method is not None:
                cft = fill_masked_data(cft, vars, self.fill_method, self.const)

        return cft  # Return the processed chunk


def prepare_dataloader(dataset: Dataset, batch_size: int = 1, callback_fn: Callable = None, num_workers: int = 0, parallel: bool = False, shuffle = True, drop_last=True) -> DataLoader:
    """
    Prepares a DataLoader.

    Args:
        dataset (Dataset): The PyTorch dataset from which to load the data.
        batch_size (int): How many samples per batch to load. Defaults to 1.
        callback_fn (Callable): A function used to collate data into batches. Defaults to None.
        num_workers (int): How many subprocesses to use for data loading. Defaults to 0.
        parallel (bool): Specifies if distributed training is performed. Defaults to False.
        shuffle (bool): Whether to shuffle the data at every epoch. Defaults to True.
        drop_last (bool): Whether to drop the last incomplete batch, if the dataset size is not divisible by the batch size. Defaults to True.

    Returns:
        DataLoader: A DataLoader object.
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
        sampler=sampler,
        drop_last=drop_last
    )
