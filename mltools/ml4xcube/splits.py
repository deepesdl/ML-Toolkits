import random
import warnings
import numpy as np
import xarray as xr
import dask.array as da
from typing import Tuple, List, Union, Dict
from ml4xcube.utils import get_chunk_sizes
from ml4xcube.preprocessing import drop_nan_values
warnings.filterwarnings('ignore')


def assign_rand_split(ds: xr.Dataset, split: float = 0.8) -> xr.Dataset:
    """
    Assign random split using random sampling.

    Args:
        ds (xr.Dataset): The xarray train_ds to which the random split will be assigned.
        split (float): The proportion of the train_ds to be used for training (default is 0.8).

    Returns:
        xr.Dataset: The train_ds with an additional 'split' variable indicating the random split.
    """
    seed = 32  # Consistent seed for reproducibility
    random.seed(seed)

    # Fetch the chunk sizes using a predefined method that returns a list of tuples (dimension, chunk size)
    chunk_sizes = dict(get_chunk_sizes(ds))  # Assuming this returns [(dim, size), ...]

    # Create a Dask array that generates random numbers for each data point, following the train_ds's chunking structure
    random_split = da.random.random(size=tuple(ds.dims[dim] for dim in ds.dims),
                                    chunks=tuple(chunk_sizes[dim] for dim in ds.dims if dim in chunk_sizes)) < split

    # Convert boolean values to floats (1.0 for True, 0.0 for False)
    random_split = random_split.astype(float)

    # Assign this array to the train_ds with the variable name 'split'
    return ds.assign(split=(list(ds.dims), random_split))


def assign_block_split(ds: xr.Dataset, block_size: List[Tuple[str, int]] = None, split: float = 0.8) -> xr.Dataset:
    """
    Assign blocks of data to training or testing sets based on a specified split ratio.

    This function uniquely assigns a block of data to either a training or testing set using a random process.
    The random seed for the assignment is generated using a Cantor pairing function on the block's indices.

    Args:
        ds (xr.Dataset): The input train_ds.
        block_size (List[Tuple[str, int]]): List of tuples specifying the dimensions and their respective sizes.
                                            If None, chunk sizes are inferred from the train_ds.
        split (float): The fraction of data to assign to the training set. The remainder will be assigned to the testing set.

    Returns:
        xr.Dataset: The input train_ds with an additional variable 'split' that indicates whether each block belongs to the training (True) or testing (False) set.
    """
    if block_size is None:
        block_size = get_chunk_sizes(ds)

    def role_dice(x):
        return x + (random.random() < split)

    def block_rand(x):
        block_ind_array = da.zeros(
            (list(x.sizes.values())), chunks=([v for k, v in block_size])
        )
        mapped = block_ind_array.map_blocks(role_dice)
        return list(ds.dims), mapped

    return ds.assign({"split": block_rand})


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





