import random
import warnings
import numpy as np
import xarray as xr
import dask.array as da
from typing import Tuple, List, Union, Dict
from ml4xcube.utils import get_chunk_sizes, calculate_total_chunks
from ml4xcube.preprocessing import drop_nan_values
warnings.filterwarnings('ignore')


def assign_rand_split(ds: xr.Dataset, split: float = 0.8, seed: int = 32) -> xr.Dataset:
    """
    Assign a random split using NumPy for reproducibility.

    Args:
        ds (xr.Dataset): The xarray dataset to which the random split will be assigned.
        split (float): The proportion of the dataset to be used for training (default is 0.8).
        seed (int): The seed for random number generation to ensure reproducibility.

    Returns:
        xr.Dataset: The dataset with an additional 'split' variable indicating the random split.
    """
    if seed is not None:
        np.random.seed(seed)

    # Fetch the dataset's shape
    shape = tuple(ds.dims[dim] for dim in ds.dims)

    # Generate random values using NumPy
    random_split = (np.random.rand(*shape) < split).astype(float)

    # Convert to a DataArray and assign to the dataset
    split_da = xr.DataArray(random_split, coords=ds.coords, dims=ds.dims)
    ds = ds.assign(split=split_da)
    return ds


def assign_block_split(ds: xr.Dataset, block_size: list = None, split: float = 0.8, seed: int = 32) -> xr.Dataset:
    """
    Assign blocks of data to training or testing sets based on a specified split ratio.

    Args:
        ds (xr.Dataset): The input dataset.
        block_size (list of tuples, optional): List of tuples specifying the dimensions and their respective block sizes.
                                               If None, block sizes are set to the chunk sizes or full dimension sizes.
        split (float): The fraction of data to assign to the training set. The remainder will be assigned to the testing set.
        seed (int): The seed for random number generation to ensure reproducibility.

    Returns:
        xr.Dataset: The input dataset with an additional variable 'split' that indicates whether each block belongs
                    to the training (1) or testing (0) set.
    """
    print(block_size)
    if seed is not None:
        np.random.seed(seed)

    # If block_size is None, get chunk sizes or use full dimension sizes
    if block_size is None:
        block_size = get_chunk_sizes(ds)

    # Calculate the number of blocks in each dimension
    total_blocks = calculate_total_chunks(ds, block_size)
    random_numbers = (np.random.rand(total_blocks) < split).astype(float)

    # Initialize an array to hold the split assignments
    dims = list(ds.dims)
    sizes = [ds.sizes[dim] for dim in dims]
    block_sizes = [dict(block_size)[dim] for dim in dims]
    num_blocks = [int(np.ceil(size / bsize)) for size, bsize in zip(sizes, block_sizes)]
    split_array = np.empty(sizes, dtype=float)

    # Create index ranges for each dimension based on block sizes
    block_edges = []
    for size, bsize in zip(sizes, block_sizes):
        edges = np.arange(0, size + bsize, bsize)
        edges[-1] = size  # Ensure the last edge matches the size exactly
        block_edges.append(edges)

    # Iterate over all combinations of blocks and assign values from random_numbers
    index = 0
    for block_indices in np.ndindex(*num_blocks):
        # Build slices for each dimension
        slices = tuple(
            slice(block_edges[dim][block_idx], block_edges[dim][block_idx + 1])
            for dim, block_idx in enumerate(block_indices)
        )

        # Assign the value from random_numbers to the entire block
        split_array[slices] = random_numbers[index]
        index += 1

    # Add the split_array to the dataset
    split_da = xr.DataArray(split_array, coords=ds.coords, dims=dims)
    ds = ds.assign(split=split_da)
    return ds


def create_split(
        data: Union[xr.Dataset, Dict[str, np.ndarray]], to_pred: Union[List[str], str] = None,
        exclude_vars: List[str] = list(), feature_vars: List[str] = None, stack_axis: int = -1,
        filter_var: str = 'filter_mask'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a train-test split for the given feature variables and target variables using the 'split' variable.

    Args:
        data (Union[xr.Dataset, Dict[str, np.ndarray]]): The xarray train_ds or dictionary of variables.
        to_pred (Union[List[str], str]): List of target variable names.
        exclude_vars (str): Variable names to exclude from the features.
        feature_vars (List[str]): List of feature variable names. If None, will be determined automatically.
        stack_axis (int): Axis along which to stack the feature and target variables (default is -1).
        filter_var (str): Name of the variable used for masking or filtering (default is 'filter_mask').
            This is used to automatically exclude filtered data from the training and the test set.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training and testing sets for features and targets.
    """
    if isinstance(data, xr.Dataset):
        data_vars = list(data.data_vars)
    elif isinstance(data, dict):
        data_vars = list(data.keys())
    else:
        raise TypeError("Input data must be an xarray.Dataset or a dictionary.")

    train_mask, test_mask = data['split'] == True, data['split'] == False

    train_data = {var: np.ma.masked_where(~train_mask, data[var]).filled(np.nan) for var in data if var != 'split'}
    test_data  = {var: np.ma.masked_where(~test_mask, data[var]).filled(np.nan) for var in data if var != 'split'}

    train_data = drop_nan_values(train_data, data_vars, 'if_all_nan', filter_var)
    test_data  = drop_nan_values(test_data, data_vars, 'if_all_nan', filter_var)

    if isinstance(to_pred, str): to_pred = [to_pred]

    for var in to_pred + ['split'] + exclude_vars:
        if var not in data_vars:
            raise ValueError(f"Variable '{var}' not found in the data.")

    # Determine feature variables if not provided
    if feature_vars is None:
        feature_vars = [var for var in data_vars if
                        var not in to_pred + ['split', filter_var] + exclude_vars]

    # Stack feature variables along the specified axis
    X_train = np.stack([train_data[var] for var in feature_vars], axis=stack_axis)
    X_test  = np.stack([test_data[var] for var in feature_vars], axis=stack_axis)

    # Stack target variables along the specified axis
    y_train = np.stack([train_data[var] for var in to_pred], axis=stack_axis)
    y_test  = np.stack([test_data[var] for var in to_pred], axis=stack_axis)

    return X_train, X_test, y_train, y_test





