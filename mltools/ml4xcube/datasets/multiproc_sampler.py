import os
import zarr
import random
import numpy as np
import xarray as xr
import dask.array as da
from multiprocessing import Pool
from typing import Tuple, Optional, Dict, List, Callable
from ml4xcube.preprocessing import get_statistics, get_range, normalize, standardize
from ml4xcube.datasets.chunk_processing import process_chunk
from ml4xcube.utils import get_chunk_by_index, calculate_total_chunks


def scale_dataset(dataset, scale_params, scale_fn) -> None:
    """
    Applies scaling to the given dataset based on the specified scaling function.
    The function directly modifies and returns the input dataset with the applied scaling,
    but since the dataset is mutable, the changes will also be reflected in the original reference.


    Args:
        dataset (Dict[str, np.ndarray]): The dataset to be scaled, represented as a dictionary where keys are variable names and values are numpy arrays.
        scale_params (Dict[str, Any]): The scaling parameters to use. These parameters depend on the scaling function:
            For 'standardize', scale_params typically include mean and standard deviation.
            For 'normalize', scale_params typically include minimum and maximum values.
        scale_fn (str): The scaling function to apply to the dataset. Possible values are:
            'standardize': Standardizes the dataset by subtracting the mean and dividing by the standard deviation.
            'normalize': Normalizes the dataset by scaling the values to a specific range (usually 0 to 1).

    Returns:
        Dict[str, np.ndarray]: The scaled dataset. The dataset is returned after being scaled according to the chosen method.
    """
    if scale_fn == 'standardize':
        dataset = standardize(dataset, scale_params)
    elif scale_fn == 'normalize':
        dataset = normalize(dataset, scale_params)
    return dataset


def worker_preprocess_chunk(args: Tuple) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Worker function to preprocess a chunk in parallel.

    Args:
        args (Tuple): A tuple containing the arguments needed for processing the chunk.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: A tuple containing the preprocessed training and testing chunks.
    """
    (chunk, use_filter, drop_sample, point_indices, overlap, filter_var, callback_fn, data_split, drop_nan,
     fill_method, const, scale_fn, scale_params) = args

    train_chunk = {var: np.empty(0) for var in chunk.keys() if var != 'split'}
    test_chunk = {var: np.empty(0) for var in chunk.keys() if var != 'split'}

    # Split the data based on the 'split' attribute
    if 'split' in chunk:
        train_mask, test_mask = chunk['split'] == True, chunk['split'] == False

        train_cf = {var: np.ma.masked_where(~train_mask, chunk[var]).filled(np.nan) for var in chunk if var != 'split'}
        test_cf = {var: np.ma.masked_where(~test_mask, chunk[var]).filled(np.nan) for var in chunk if var != 'split'}

        train_cft, test_cft = None, None
        valid_train, valid_test = False, False

        if train_mask.any():
            train_cft, valid_train = process_chunk(train_cf, use_filter, drop_sample, filter_var, point_indices, overlap, fill_method, const, drop_nan)
        if test_mask.any():
            test_cft, valid_test = process_chunk(test_cf, use_filter, drop_sample, filter_var, point_indices, overlap, fill_method, const, drop_nan)

        if valid_train:
            if callback_fn: train_cft = callback_fn(train_cft)
            if scale_fn: train_cft = scale_dataset(train_cft, scale_params, scale_fn)
            train_chunk = train_cft
        if valid_test:
            if callback_fn: test_cft = callback_fn(test_cft)
            if scale_fn: test_cft = scale_dataset(test_cft, scale_params, scale_fn)
            test_chunk = test_cft
    else:
        cft, valid_train = process_chunk(chunk, use_filter, drop_sample, filter_var, point_indices, overlap, fill_method, const, drop_nan)

        if valid_train:
            if callback_fn: cft = callback_fn(cft)
            if scale_fn: cft = scale_dataset(cft, scale_params, scale_fn)

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


class MultiProcSampler:
    def __init__(
        self, ds: xr.Dataset, rand_chunk: bool = False, data_fraq: float = 1.0, nproc: int = 4,
        apply_mask: bool = True, drop_sample: bool = True, fill_method: str = None, const: float = None,
        filter_var: str = 'filter_mask', chunk_size: Tuple[int, ...] = None, train_cube: str = 'train_cube.zarr',
        test_cube: str = 'test_cube.zarr', drop_nan: str = 'auto', array_dims: Tuple[str, ...] = None,
        data_split: float = 0.8, chunk_batch: int = None, callback: Callable = None,
        block_size: List[Tuple[str, int]] = None, sample_size: List[Tuple[str, int]] = None,
        overlap: List[Tuple[str, float]] = None, scale_fn: str = 'standardize'
    ):
        """
        Initializes the MultiProcSampler with the given dataset and parameters. This class handles the parallel processing
        of data chunks, applying preprocessing and scaling, and storing the processed chunks in Zarr format for training
        and testing.

        Args:
            ds (xr.Dataset): The input xarray dataset to process.
            rand_chunk (bool): Whether to select chunks randomly for processing. Defaults to False.
            data_fraq (float): The fraction of the dataset to process, where 1.0 means the entire dataset. Defaults to 1.0.
            nproc (int): The number of processes to use for parallel processing. Defaults to 4.
            apply_mask (bool): Whether to apply a mask based on the `filter_var`. Defaults to True.
            drop_sample (bool): Whether to drop an entire subarray if any value does not belong to the mask. Defaults to True.
            fill_method (str): The method to fill masked data, if any (e.g., 'ffill', 'bfill'). Defaults to None.
            const (float): A constant value to fill masked data if `fill_method` is not provided. Defaults to None.
            filter_var (str): The variable name to use for filtering invalid data (e.g., 'land_mask'). Defaults to 'land_mask'.
            chunk_size (Tuple[int]): The size of chunks to generate for training and testing data. Defaults to None, which will automatically determine chunk sizes.
            train_cube (str): The file path for the Zarr store to hold training data. Defaults to 'train_cube.zarr'.
            test_cube (str): The file path for the Zarr store to hold testing data. Defaults to 'test_cube.zarr'.
            drop_nan (str): Defines how to handle missing values:
                If 'auto', drop the entire sample if any NaN values are present.
                If 'if_all_nan', drop the sample if all values are NaN.
                If 'masked', drop the subarray if valid values according to the mask are NaN. Defaults to 'auto'.
            array_dims (Tuple[str, ...]): The dimensions of the arrays in the dataset (e.g., 'samples', 'time'). Defaults to ('samples',).
            data_split (float): The fraction of data to use for training, with the remaining used for testing. Defaults to 0.8 (80% training, 20% testing).
            chunk_batch (int): The number of chunks to process in each batch. Defaults to the number of processes (`nproc`).
            callback (Callable): An optional callback function to apply additional processing to each chunk. Defaults to None.
            block_size (List[Tuple[str, int]]): The block sizes for processing chunks, defined as a list of tuples (dimension name, block size). Defaults to None.
            sample_size (List[Tuple[str, int]]): The size of samples to extract from chunks, defined as a list of tuples (dimension name, sample size). Defaults to None.
            overlap (List[Tuple[str, float]]): The overlap between samples, defined as a list of tuples (dimension name, overlap fraction). Defaults to None.
            scale_fn (str): The scaling function to apply to the dataset.
                If 'standardize', standardization of the data is conducted.
                If 'normalize', the data is normalized.

        Attributes:
            total_chunks (int): The total number of chunks in the dataset, calculated based on the block size.
            num_chunks (int): The number of chunks to process, determined by the data fraction (`data_fraq`).
            train_cube (zarr.Group): The Zarr store for training data.
            test_cube (zarr.Group): The Zarr store for testing data.
            scale_params (Dict[str, Any]): The parameters used for scaling the dataset, determined by the scaling function (`scale_fn`).
            remainder_data_train (Dict[str, np.ndarray]): Stores remainder training data that couldn't be appended due to size not being a multiple of chunk_size[0].
            remainder_data_test (Dict[str, np.ndarray]): Stores remainder testing data that couldn't be appended due to size not being a multiple of chunk_size[0].
        """
        self.ds = ds
        self.rand_chunk = rand_chunk
        self.apply_mask = apply_mask
        self.drop_sample = drop_sample
        self.fill_method = fill_method
        self.const = const
        self.filter_var = filter_var
        self.train_cube = train_cube
        self.test_cube = test_cube
        self.drop_nan = drop_nan

        if array_dims is None:
            sample_dims = set(dim[0] for dim in sample_size)
            extra_dims = [dim for dim in array_dims if
                          dim not in sample_dims]
            self.array_dims = tuple(extra_dims) + tuple(dim for dim in array_dims if dim in sample_dims)
        else:
            self.array_dims = array_dims

        self.data_split = data_split
        self.callback = callback
        self.block_size = block_size
        self.total_chunks = int(calculate_total_chunks(self.ds, self.block_size))
        self.sample_size = sample_size
        self.overlap = overlap
        self.data_fraq = data_fraq
        self.num_chunks = int(self.total_chunks * self.data_fraq)
        self.nproc = min(nproc, self.num_chunks)
        self.chunk_batch = chunk_batch if chunk_batch is not None else self.nproc
        self.scale_fn = scale_fn
        self.scale_params = None
        self.set_scale_params()
        self.chunk_size = chunk_size
        if chunk_size is None: self.chunk_size = 'auto'

        self.remainder_data_train = {var: np.empty(0) for var in self.ds.keys() if var != 'split'}
        self.remainder_data_test = {var: np.empty(0) for var in self.ds.keys() if var != 'split'}

        self.total_chunks = calculate_total_chunks(ds)
        self.create_cubes()

    def set_scale_params(self) -> None:
        """
        Sets the parameters for scaling the dataset based on the selected scaling function (`scale_fn`).
            If 'standardize', computes mean and standard deviation for standardization.
            If 'normalize', computes the range (min and max values) for normalization.

        Returns:
            None
        """
        if self.scale_fn == 'standardize':
            self.scale_params = get_statistics(self.ds)
        elif self.scale_fn == 'normalize':
            self.scale_params = get_range(self.ds)

    def store_chunks(self, processed_chunks) -> None:
        """
        Store the processed chunks into the training and testing Zarr cubes.

        Args:
            processed_chunks (List[Tuple[Dict[str, np.ndarray], List[int], List[int]]]):
                List of tuples containing the processed chunk, training indices, and testing indices.

        Returns:
            None
        """
        chunk_dim_size = self.chunk_size[0]

        for train_data, test_data in processed_chunks:

            if train_data is None and test_data is None:
                continue

            train_data_arrays = {}
            test_data_arrays = {}

            for var in self.ds.keys():
                if var == 'split' or var == self.filter_var: continue

                print(f"Processing variable: {var}")
                print(f"Train data samples: {train_data[var].shape}")
                print(f"Test data samples: {test_data[var].shape}")

                # Concatenate remainder data if it exists
                if self.remainder_data_train[var].size > 0:
                    train_data[var] = np.concatenate([self.remainder_data_train[var], train_data[var]], axis=0)
                    self.remainder_data_train[var] = np.empty(0)  # Clear the remainder once concatenated

                if self.remainder_data_test[var].size > 0:
                    test_data[var] = np.concatenate([self.remainder_data_test[var], test_data[var]], axis=0)
                    self.remainder_data_test[var] = np.empty(0)  # Clear the remainder once concatenated

                # Calculate how much data can be appended (multiple of chunk_dim_size)
                train_size = train_data[var].shape[0]
                test_size = test_data[var].shape[0]

                # Determine the number of full chunks that can be stored
                num_train_chunks = train_size // chunk_dim_size
                num_test_chunks = test_size // chunk_dim_size

                # Compute the remainder
                train_remainder_size = train_size % chunk_dim_size
                test_remainder_size = test_size % chunk_dim_size

                if num_train_chunks > 0:
                    train_var_data = da.from_array(train_data[var][:num_train_chunks * chunk_dim_size],
                                                   chunks=self.chunk_size)
                    train_data_arrays[var] = (self.array_dims, train_var_data)

                if num_test_chunks > 0:
                    test_var_data = da.from_array(test_data[var][:num_test_chunks * chunk_dim_size],
                                                  chunks=self.chunk_size)
                    test_data_arrays[var] = (self.array_dims, test_var_data)

                # Store the remainder data for future concatenation
                if train_remainder_size > 0:
                    self.remainder_data_train[var] = train_data[var][-train_remainder_size:]
                    print('train remainder: ', self.remainder_data_train[var].shape)

                if test_remainder_size > 0:
                    self.remainder_data_test[var] = test_data[var][-test_remainder_size:]
                    print('test remainder: ', self.remainder_data_test[var].shape)

            if train_data_arrays:
                train_ds = xr.Dataset(train_data_arrays)
                # Append data to the Zarr store with dimension names
                if not os.path.exists(self.train_cube):
                    train_ds.to_zarr(self.train_cube)
                else:
                    train_ds.to_zarr(self.train_cube, mode='a', append_dim=self.array_dims[0])

            if test_data_arrays:
                test_ds = xr.Dataset(test_data_arrays)
                # Append data to the Zarr store with dimension names
                if not os.path.exists(self.test_cube):
                    test_ds.to_zarr(self.test_cube)
                else:
                    test_ds.to_zarr(self.test_cube, mode='a', append_dim=self.array_dims[0])

    def create_cubes(self) -> None:
        """
        Create the training and testing cubes by processing chunks of data from the train_ds.

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
                    (chunk, self.apply_mask, self.drop_sample, self.sample_size, self.overlap, self.filter_var, self.callback, self.data_split, self.drop_nan, self.fill_method, self.const, self.scale_fn, self.scale_params)
                    for chunk in batch_chunks
                ])
                self.store_chunks(processed_chunks)

    def get_datasets(self) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Retrieve the training and testing datasets as xarray Datasets.

        Returns:
            Tuple[xr.Dataset, xr.Dataset]: The training and testing xarray Datasets.
        """
        train_ds = xr.open_zarr(self.train_cube)
        test_ds = xr.open_zarr(self.test_cube)
        return train_ds, test_ds

