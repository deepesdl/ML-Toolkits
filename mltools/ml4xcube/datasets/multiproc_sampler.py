import zarr
import random
import numpy as np
import xarray as xr
import dask.array as da
from multiprocessing import Pool
from typing import Tuple, Optional, Dict, List, Callable
from ml4xcube.cube_utilities import split_chunk, assign_dims
from ml4xcube.cube_utilities import get_chunk_by_index, calculate_total_chunks
from ml4xcube.preprocessing import apply_filter, drop_nan_values, fill_masked_data


def process_chunk(chunk: Dict[str, np.ndarray], use_filter: bool, drop_sample: bool, filter_var: Optional[str],
                  sample_size: Optional[List[Tuple[str, int]]], overlap: Optional[List[Tuple[str, int]]], callback_fn: Optional[Callable],
                  drop_nan_masked: bool, fill_method: Optional[str], const: Optional[float]
                  ) -> Tuple[Dict[str, np.ndarray], bool]:
    """
    Process a single chunk of data.

    Args:
        chunk (Dict[str, np.ndarray]): A dictionary containing the data chunk to preprocess.
        use_filter (bool): If true, apply the filter based on the specified filter_var.
        drop_sample (bool): If true, drop the entire subarray if any value in the subarray does not belong to the mask (False).
        filter_var (Optional[str]): The variable to use for filtering.
        sample_size (Optional[List[Tuple[str, int]]]): Sizes of the samples to be extracted from the chunk along each dimension.
                                                       Each tuple contains the dimension name and the size along that dimension.
        overlap (Optional[List[Tuple[str, int]]]): Overlap for overlapping samples due to chunk splitting.
                                                   Each tuple contains the dimension name and the overlap fraction along that dimension.
        callback_fn (Optional[Callable]): Optional callback function to apply to each chunk after preprocessing.
        drop_nan_masked (bool): If true, NaN values are dropped using the mask specified by filter_var.
        fill_method (Optional[str]): Method to fill masked data, if any.
        const (float): Constant value to use for filling masked data, if needed.

    Returns:
        Tuple[Dict[str, np.ndarray], bool]: A tuple containing the preprocessed chunk and a boolean indicating if the chunk is valid.
    """
    cf = split_chunk(chunk, sample_size=sample_size, overlap=overlap)

    # Apply filtering based on the specified variable, if provided
    if use_filter:
        cft = apply_filter(cf, filter_var, drop_sample)
    else:
        cft = cf

    vars = list(cft.keys())
    if drop_nan_masked:
        cft = drop_nan_values(cft, vars, filter_var)
    else:
        cft = drop_nan_values(cft, vars)

    valid_chunk = all(np.nan_to_num(cft[var]).sum() > 0 for var in cf)

    if valid_chunk:
        vars = [var for var in cft.keys() if var != 'split' and var != filter_var]
        if fill_method is not None:
            cft = fill_masked_data(cft, vars, fill_method, const)
        if callback_fn:
            cft = callback_fn(cft)
    return cft, valid_chunk


def worker_preprocess_chunk(args: Tuple) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Worker function to preprocess a chunk in parallel.

    Args:
        args (Tuple): A tuple containing the arguments needed for processing the chunk.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: A tuple containing the preprocessed training and testing chunks.
    """
    chunk, use_filter, drop_sample, point_indices, overlap, filter_var, callback_fn, data_split, drop_nan_masked, fill_method, const = args

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
            train_cft, valid_train = process_chunk(train_cf, use_filter, drop_sample, filter_var, point_indices, overlap, callback_fn, drop_nan_masked, fill_method, const)
        if test_mask.any():
            test_cft, valid_test = process_chunk(test_cf, use_filter, drop_sample, filter_var, point_indices, overlap, callback_fn, drop_nan_masked, fill_method, const)

        if valid_train:
            train_chunk = train_cft
        if valid_test:
            test_chunk = test_cft
    else:
        cft, valid_train = process_chunk(chunk, use_filter, drop_sample, filter_var, point_indices, overlap, callback_fn, drop_nan_masked, fill_method, const)

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
    def __init__(self, ds: xr.Dataset, rand_chunk: bool = False, drop_nan_masked: bool = False,
                 data_fraq: float = 1.0, nproc: int = 4, use_filter: bool = True,
                 drop_sample: bool = True, fill_method: str = None, const: float = None,
                 filter_var: str = 'land_mask', chunk_size: Tuple[int] = None,
                 train_cube: str = 'train_cube.zarr', test_cube: str = 'test_cube.zarr',
                 array_dims: Tuple[str, Optional[str], Optional[str]] = ('samples',),
                 data_split: float = 0.8, chunk_batch: int = None, callback_fn: Callable = None,
                 block_size: List[Tuple[str, int]] = None,
                 sample_size: List[Tuple[str, int]] = None,
                 overlap: List[Tuple[str, int]] = None):
        """
        Initialize the MultiProcSampler with the given dataset and parameters.

        Attributes:
            ds (xr.Dataset): The input dataset.
            rand_chunk (bool): Whether to select chunks randomly.
            drop_nan_masked (bool): If true, NaN values are dropped using the mask specified by filter_var.
            data_fraq (float): The fraction of data to process.
            nproc (int): Number of processes to use for parallel processing.
            use_filter (bool): If true, apply the filter based on the specified filter_var.
            drop_sample (bool): If true, drop the entire subarray if any value in the subarray does not belong to the mask (False).
            fill_method (str): Method to fill masked data, if any.
            const (Optional[float]): Constant value to use for filling masked data, if needed.
            filter_var (Optional[str]): The variable to use for filtering.
            chunk_size (Tuple[int]): The size of chunks to process.
            train_store (zarr.Group): Zarr store for training data.
            test_store (zarr.Group): Zarr store for testing data.
            array_dims (Tuple[str, Optional[str], Optional[str]]): Tuple specifying the dimensions of the arrays.
            data_split (float): The fraction of data to use for training.
            chunk_batch (int): Number of chunks to process in each batch.
            callback_fn (function): Optional callback function to apply to each chunk after preprocessing.
            block_size (List[Tuple[str, int]]): Optional list specifying the block sizes for each dimension.
            sample_size (List[Tuple[str, int]]): List of tuples specifying the dimensions and their respective sizes.
            overlap (List[Tuple[str, int]]): List of tuples specifying the dimensions and their respective overlap fractions.
            total_chunks (int): Total number of chunks in the dataset.
            num_chunks (int): Number of chunks to process.
        """
        self.ds = ds
        self.rand_chunk = rand_chunk
        self.drop_nan_masked = drop_nan_masked
        self.use_filter = use_filter
        self.drop_sample = drop_sample
        self.fill_method = fill_method
        self.const = const
        self.filter_var = filter_var
        self.chunk_size = chunk_size
        self.train_store = zarr.open(train_cube)
        self.test_store = zarr.open(test_cube)
        self.array_dims = array_dims
        self.data_split = data_split
        self.chunk_batch = chunk_batch if chunk_batch is not None else self.nproc
        self.callback_fn = callback_fn
        self.block_size = block_size
        self.total_chunks = int(calculate_total_chunks(self.ds, self.block_size))
        self.sample_size = sample_size
        self.overlap = overlap
        self.data_fraq = data_fraq
        self.num_chunks = int(self.total_chunks * self.data_fraq)
        self.nproc = min(nproc, self.num_chunks)
        if chunk_size is None:
            self.chunk_size = tuple(dim[1] for dim in self.sample_size) if self.sample_size else (1,)
        self.total_chunks = calculate_total_chunks(ds)
        self.create_cubes()

    def store_chunks(self, processed_chunks) -> None:
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
                if var == 'split': continue

                print(f"Processing variable: {var}")
                print(f"Train data samples: {train_data[var].shape[0]}")
                print(f"Test data samples: {test_data[var].shape[0]}")

                append_dim = None
                if self.sample_size is not None: append_dim = 0
                if train_data[var].shape[0] > 0:
                    train_var_data = train_data[var]
                    if var not in self.train_store:
                        self.train_store.create_dataset(var, data=train_var_data, shape=train_var_data.shape,
                                                        dtype=train_var_data.dtype, chunks=self.chunk_size, append_dim=append_dim)
                    else:
                        self.train_store[var].append(train_var_data)

                if test_data[var].shape[0] > 0:
                    test_var_data = test_data[var]
                    if var not in self.test_store:
                        self.test_store.create_dataset(var, data=test_var_data, shape=test_var_data.shape,
                                                       dtype=test_var_data.dtype, chunks=self.chunk_size, append_dim=append_dim)
                    else:
                        self.test_store[var].append(test_var_data)

    def create_cubes(self) -> None:
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
                    (chunk, self.use_filter, self.drop_sample, self.sample_size, self.overlap, self.filter_var, self.callback_fn, self.data_split, self.drop_nan_masked, self.fill_method, self.const)
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