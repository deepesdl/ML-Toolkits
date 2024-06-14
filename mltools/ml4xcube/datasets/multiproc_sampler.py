import zarr
import math
import random
import numpy as np
import xarray as xr
from multiprocessing import Pool
from typing import Tuple, Optional, Dict, List
from ml4xcube.cube_utilities import split_chunk
from ml4xcube.preprocessing import apply_filter, drop_nan_values
from ml4xcube.cube_utilities import get_chunk_by_index, calculate_total_chunks


def worker_preprocess_chunk(args):
    chunk, point_indices, overlap, filter_var, callback_fn, data_split = args
    if point_indices is not None:
        cf = {x: chunk[x] for x in chunk.keys()}
        cf = split_chunk(cf, point_indices, overlap=overlap)
    else:
        cf = {x: chunk[x].ravel() for x in chunk.keys()}

    # Apply filtering based on the specified variable, if provided
    if not filter_var is None:
        cft = apply_filter(cf, filter_var)
    else:
        cft = cf

    vars = list(cft.keys())
    cft = drop_nan_values(cft, vars)
    valid_chunk = all(np.nan_to_num(cft[var]).sum() > 0 for var in cf)

    if valid_chunk:
        if callback_fn:
            cft = callback_fn(cft)

        num_samples = cft[list(cft.keys())[0]].shape[0]
        indices = list(range(num_samples))
        random.shuffle(indices)
        split_idx = int(num_samples * data_split)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        return cft, train_indices, test_indices
    return None, None, None


class MultiProcSampler():
    def __init__(self, ds: xr.Dataset, filter_var: str = 'land_mask',
                 data_fraq: float = 0.03, data_split: float = 0.8,
                 nproc: int = 4, callback_fn = None,
                 block_sizes: Optional[Dict[str, Optional[int]]] = None,
                 point_indices: Optional[List[Tuple[str, int]]] = None,
                 overlap: Optional[List[Tuple[str, int]]] = None):
        """
        Initialize the TrainTestSampler with the given dataset and parameters.

        Args:
            ds (xr.Dataset): The input dataset.
            filter_var (str): The variable to use for filtering. Defaults to 'land_mask'.
                               If None, no filtering is applied.
            data_fraq (float): The fraction of data to process.
            data_split (float): The fraction of data to use for training. The rest is used for testing.
            nproc (int): Number of processes to use for parallel processing.
            callback_fn (function): Optional callback function to apply to each chunk after preprocessing.
            block_sizes (Optional[Dict[str, Optional[int]]]): Optional dictionary specifying the block sizes for each dimension.
            point_indices (Optional[List[Tuple[str, int]]]): List of tuples specifying the dimensions and their respective step sizes.
            overlap (Optional[List[Tuple[str, int]]]): List of tuples specifying the dimensions and their respective overlap sizes.

        Returns:
            None
        """
        self.ds = ds
        self.filter_var = filter_var
        self.data_fraq = data_fraq
        self.data_split = data_split
        self.block_sizes = block_sizes
        self.point_indices = point_indices
        self.overlap = overlap
        self.total_chunks = int(calculate_total_chunks(self.ds, self.block_sizes) * self.data_fraq)
        self.chunks = None
        self.nproc = nproc
        self.callback_fn = callback_fn
        self.train_store = zarr.open('train_cube.zarr')
        self.test_store = zarr.open('test_cube.zarr')
        self.chunk_size = tuple(dim[1] for dim in self.point_indices) if self.point_indices else (1000,)
        self.create_cubes()

    def store_chunks(self, processed_chunks):
        """
        Store the processed chunks into the training and testing Zarr cubes.

        Args:
            processed_chunks (List[Tuple[Dict[str, np.ndarray], List[int], List[int]]]):
                List of tuples containing the processed chunk, training indices, and testing indices.

        Returns:
            None
        """
        for chunk, train_indices, test_indices in processed_chunks:
            print(train_indices, test_indices)

            if chunk is None:
                continue

            for var in chunk.keys():
                train_data = chunk[var][train_indices]
                test_data = chunk[var][test_indices]

                print(f"Processing variable: {var}")
                print(f"Train data shape: {train_data.shape}")
                print(f"Test data shape: {test_data.shape}")

                if var not in self.train_store:
                    self.train_store.create_dataset(var, data=train_data, chunks=self.chunk_size, append_dim=0)
                    self.test_store.create_dataset(var, data=test_data, chunks=self.chunk_size, append_dim=0)
                else:
                    self.train_store[var].append(train_data)
                    self.test_store[var].append(test_data)

    def create_cubes(self):
        """
        Create the training and testing cubes by processing chunks of data from the dataset.

        This method retrieves specific chunks of data, preprocesses them in parallel, and stores the results in Zarr cubes.

        Returns:
            None
        """
        chunk_indices = list(range(self.total_chunks))
        chunks = [get_chunk_by_index(self.ds, idx) for idx in chunk_indices]
        with Pool(processes=self.nproc) as pool:
            for i in range(0, self.total_chunks, math.ceil(self.nproc*0.2)):
                batch_chunks = chunks[i:i + math.ceil(self.nproc*0.2)]
                processed_chunks = pool.map(worker_preprocess_chunk, [
                    (chunk, self.point_indices, self.overlap, self.filter_var, self.callback_fn, self.data_split)
                    for chunk in batch_chunks
                ])
                self.store_chunks(processed_chunks)

    def get_datasets(self):
        """
        Retrieve the training and testing Zarr datasets.

        Returns:
            Tuple[zarr.Group, zarr.Group]: The training and testing Zarr datasets.
        """
        return self.train_store, self.test_store
