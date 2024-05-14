import torch
import random
from typing import Callable
from torch.utils.data import Dataset, DataLoader

from ml4xcube.cube_utilities import calculate_total_chunks, get_chunk_by_index
from ml4xcube.preprocessing import apply_filter, drop_nan_values


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
    - dataset: The pytorch dataset from which to load the data.
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
