import torch
import random
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Callable, Dict, Union
from ml4xcube.datasets.chunk_processing import process_chunk
from ml4xcube.utils import calculate_total_chunks, get_chunk_by_index


class PTXrDataset(Dataset):
    def __init__(self, ds: xr.Dataset, rand_chunk: bool = True, drop_nan: str = 'auto', drop_sample: bool = False,
                 chunk_indices: List[int] = None, apply_mask: bool = True, fill_method: str = None,
                 const: float = None, filter_var: str = 'filter_mask', num_chunks: int = None, callback = None,
                 block_sizes: List[Tuple[str, int]] = None, sample_size: List[Tuple[str, int]] = None,
                 overlap: List[Tuple[str, float]] = None, process_chunks: bool = False):
        """
        Initializes a PyTorch-compatible dataset for efficiently managing and processing large xarray datasets,
        with support for chunking, filtering, and preprocessing.

        Args:
            ds (xr.Dataset): The input xarray dataset to process.
            rand_chunk (bool): If True, selects chunks randomly when no chunk indices are provided. Defaults to True.
            drop_nan (str): Specifies how to handle NaN values in the data. Defaults to 'auto'.
                - 'auto': Drop the entire sample if any NaN values are present.
                - 'if_all_nan': Drop the sample if all values are NaN.
                - 'masked': Drop the subarray if valid values according to the mask are NaN.
            drop_sample (bool): If True, drops the entire subarray if any value in the subarray does not match the mask. Defaults to False.
            chunk_indices (List[int]): List of specific chunk indices to process. If None, chunks are selected randomly or sequentially. Defaults to None.
            apply_mask (bool): If True, applies a filter based on the specified `filter_var` to mask invalid data. Defaults to True.
            fill_method (str): The method used to fill masked data, such as 'ffill' (forward fill) or 'bfill' (backward fill). Defaults to None.
            const (float): A constant value to fill masked data if no fill method is provided. Defaults to None.
            filter_var (str): The variable used to filter the data, typically a mask (e.g., 'land_mask'). Defaults to 'land_mask'.
            num_chunks (int): The number of chunks to process. If None, all chunks will be processed. Defaults to None.
            callback (Callable): A function to apply additional processing to each chunk after initial preprocessing. Defaults to None.
            block_sizes (List[Tuple[str, int]]): Block sizes for splitting the dataset into chunks, defined as a list of tuples (dimension name, block size). Defaults to None.
            sample_size (List[Tuple[str, int]]): The size of samples to extract from chunks, defined as a list of tuples (dimension name, sample size). Defaults to None.
            overlap (List[Tuple[str, float]]): The overlap between samples, defined as a list of tuples (dimension name, overlap fraction). Defaults to None.
            process_chunks (bool,): If True, preprocesses each chunk (e.g., filtering, filling) before returning it. Defaults to False.

        Attributes:
            total_chunks (int): The total number of chunks in the dataset.
        """
        self.ds = ds
        self.rand_chunk = rand_chunk
        self.drop_nan = drop_nan
        self.apply_mask = apply_mask
        self.drop_sample = drop_sample
        self.fill_method = fill_method
        self.const = const
        self.filter_var = filter_var
        self.num_chunks = num_chunks
        self.callback = callback
        self.block_sizes = block_sizes
        self.sample_size = sample_size
        self.overlap = overlap
        self.process_chunks = process_chunks

        self.total_chunks = int(calculate_total_chunks(ds, self.block_sizes))
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

        if self.process_chunks:
            chunk, _ = process_chunk(chunk, self.apply_mask, self.drop_sample, self.filter_var, self.sample_size,
                                     self.overlap, self.fill_method, self.const, self.drop_nan)

        if self.callback:
            chunk = self.callback(chunk)

        return chunk  # Return the processed chunk


def prep_dataloader(
    train_ds: Dataset, test_ds: Dataset = None, batch_size: int = 1, callback: Callable = None, num_workers: int = 0,
    parallel: bool = False, shuffle = True, drop_last=True) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """
    Prepares a DataLoader for training and optionally testing.

    Args:
        train_ds (Dataset): The PyTorch dataset from which to load training data.
        test_ds (Dataset, optional): The PyTorch dataset from which to load test data. If None, only the training DataLoader is returned. Defaults to None.
        batch_size (int): How many samples per batch to load. Defaults to 1.
        callback (Callable, optional): A function used to collate data into batches. Defaults to None.
        num_workers (int): How many subprocesses to use for data loading. Defaults to 0 (single process).
        parallel (bool): Specifies if distributed training is performed. When True, a DistributedSampler is used for training, and shuffling is disabled. Defaults to False.
        shuffle (bool): Whether to shuffle the data at every epoch. Defaults to True. Automatically set to False when `parallel` is True.
        drop_last (bool): Whether to drop the last incomplete batch, if the dataset size is not divisible by the batch size. Defaults to True.

        Returns:
        Union[DataLoader, Tuple[DataLoader, DataLoader]]:
            If `test_ds` is None: Returns a single DataLoader object for the training dataset.
            If `test_ds` is provided: Returns a tuple containing two DataLoader objects, one for the training dataset and one for the test dataset.

    """
    sampler = None
    if parallel:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(train_ds)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,  # Conditionally use pin_memory
        shuffle=shuffle,
        collate_fn=callback,
        sampler=sampler,
        drop_last=drop_last
    )

    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,  # Conditionally use pin_memory
            shuffle=shuffle,
            collate_fn=callback,
            sampler=sampler,
            drop_last=drop_last
        )
        return train_loader, test_loader

    return train_loader
