import zarr
import random
import numpy as np
import xarray as xr
import dask.array as da
from multiprocessing import Pool
from typing import Tuple, Optional, Dict, List
from ml4xcube.cube_utilities import split_chunk, assign_dims
from ml4xcube.preprocessing import apply_filter, drop_nan_values
from ml4xcube.cube_utilities import get_chunk_by_index, calculate_total_chunks


def process_chunk(chunk, filter_var, point_indices, overlap, callback_fn):
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
    return cft, valid_chunk

def worker_preprocess_chunk(args):
    chunk, point_indices, overlap, filter_var, callback_fn, data_split = args

    train_chunk = {var: np.empty(0) for var in chunk.keys() if var != 'split'}
    test_chunk = {var: np.empty(0) for var in chunk.keys() if var != 'split'}

    # Split the data based on the 'split' attribute
    if 'split' in chunk:
        train_mask, test_mask = chunk['split'] == True, chunk['split'] == False

        train_cf = {var: np.ma.masked_where(~train_mask, chunk[var]) for var in chunk if var != 'split'}
        test_cf = {var: np.ma.masked_where(~test_mask, chunk[var]) for var in chunk if var != 'split'}

        train_cft, test_cft = None, None
        valid_train, valid_test = False, False

        if train_mask.any():
            train_cft, valid_train = process_chunk(train_cf, filter_var, point_indices, overlap, callback_fn)
        if test_mask.any():
            test_cft, valid_test = process_chunk(test_cf, filter_var, point_indices, overlap, callback_fn)

        if valid_train:
            train_chunk = train_cft
        if valid_test:
            test_chunk = test_cft
    else:
        cft, valid_train = process_chunk(chunk, filter_var, point_indices, overlap, callback_fn)

        if valid_train:
            num_samples = cft[list(cft.keys())[0]].shape[0]
            indices = list(range(num_samples))
            random.shuffle(indices)
            split_idx = int(num_samples * data_split)
            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]

            for var in chunk.keys():
                train_chunk[var] = cft[var][train_indices]
                test_chunk[var] = cft[var][test_indices]
    return train_chunk, test_chunk


class MultiProcSampler():
    def __init__(self, ds: xr.Dataset, filter_var: str = 'land_mask',
                 chunk_size: Tuple = None, train_cube: str = 'train_cube.zarr',
                 test_cube: str = 'test_cube.zarr', split: int = 0.8,
                 nproc: int = 4, data_fraq: float = 1.0, rand_chunk: bool = False,
                 array_dims: Tuple[str, Optional[str], Optional[str]] = ('samples',),
                 data_split: float = 0.8, chunk_batch: int = None, callback_fn = None,
                 block_size: Optional[List[Tuple[str, int]]] = None,
                 point_indices: Optional[List[Tuple[str, int]]] = None,
                 overlap: Optional[List[Tuple[str, int]]] = None):
        """
        Initialize the TrainTestSampler with the given dataset and parameters.

        Args:
            ds (xr.Dataset): The input dataset.
            filter_var (str): The variable to use for filtering. Defaults to 'land_mask'.
                               If None, no filtering is applied.
            data_fraq (float): The fraction of data to process.
            rand_chunk (bool): Whether to select chunks randomly.
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
        self.block_size = block_size
        self.point_indices = point_indices
        self.overlap = overlap
        self.split = split
        self.total_chunks = int(calculate_total_chunks(self.ds, self.block_size))
        self.num_chunks = int(self.total_chunks * self.data_fraq)
        self.chunks = None
        self.rand_chunk = rand_chunk
        self.nproc = min(nproc, self.num_chunks)
        self.array_dims = array_dims
        self.chunk_batch = chunk_batch if chunk_batch is not None else self.nproc
        self.callback_fn = callback_fn
        self.train_store = zarr.open(train_cube)
        self.test_store = zarr.open(test_cube)
        self.chunk_size = chunk_size
        if chunk_size is None:
            self.chunk_size = tuple(dim[1] for dim in self.point_indices) if self.point_indices else (1, )
        self.total_chunks = calculate_total_chunks(ds)
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
        for train_data, test_data in processed_chunks:

            if train_data is None and test_data is None:
                continue

            for var in self.ds.keys():
                if var == 'split' or var == 'land_mask': continue

                print(f"Processing variable: {var}")
                print(f"Train data samples: {train_data[var].shape[0]}")
                print(f"Test data samples: {test_data[var].shape[0]}")

                if train_data[var].shape[0] > 0:
                    train_var_data = train_data[var]
                    if var not in self.train_store:
                        self.train_store.create_dataset(var, data=train_var_data, shape=train_var_data.shape, dtype=train_var_data.dtype,
                                                        chunks=self.chunk_size, append_dim=0)
                    else:
                        self.train_store[var].append(train_var_data)

                if test_data[var].shape[0] > 0:
                    test_var_data = test_data[var]
                    if var not in self.test_store:
                        self.test_store.create_dataset(var, data=test_var_data, shape=test_var_data.shape, dtype=test_var_data.dtype,
                                                       chunks=self.chunk_size, append_dim=0)
                    else:
                        self.test_store[var].append(test_var_data)

    def create_cubes(self):
        """
        Create the training and testing cubes by processing chunks of data from the dataset.

        This method retrieves specific chunks of data, preprocesses them in parallel, and stores the results in Zarr cubes.

        Returns:
            None
        """
        if self.rand_chunk:
            chunk_indices = random.sample(range(self.total_chunks), self.num_chunks)
        else:
            chunk_indices = list(range(self.num_chunks))

        with Pool(processes=self.nproc) as pool:
            for i in range(0, self.num_chunks, self.chunk_batch):
                batch_indices = chunk_indices[i:i + self.chunk_batch]
                batch_chunks = [get_chunk_by_index(self.ds, idx, block_size=self.block_size) for idx in batch_indices]
                processed_chunks = pool.map(worker_preprocess_chunk, [
                    (chunk, self.point_indices, self.overlap, self.filter_var, self.callback_fn, self.data_split)
                    for chunk in batch_chunks
                ])
                self.store_chunks(processed_chunks)

    def get_datasets(self) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Retrieve the training and testing datasets as xarray Datasets.

        Returns:
            Tuple[xr.Dataset, xr.Dataset]: The training and testing xarray Datasets.
        """
        # Convert Zarr stores to Dask arrays and then to xarray Datasets
        self.train_data = {var: da.from_zarr(self.train_store[var]) for var in self.train_store.array_keys()}
        self.test_data = {var: da.from_zarr(self.test_store[var]) for var in self.test_store.array_keys()}

        # Assign dimensions using the assign_dims function
        self.train_data = assign_dims(self.train_data, self.array_dims)
        self.test_data = assign_dims(self.test_data, self.array_dims)

        train_ds = xr.Dataset(self.train_data)
        test_ds = xr.Dataset(self.test_data)

        return train_ds, test_ds
