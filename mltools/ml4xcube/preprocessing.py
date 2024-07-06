import numpy as np
import xarray as xr
from typing import Dict, List


def apply_filter(ds: Dict[str, np.ndarray], filter_var: str, drop_sample: bool = False) -> Dict[str, np.ndarray]:
    """
    Apply a filter to the dataset. If drop_sample is True and any value in a subarray does not belong to the mask (False),
    drop the entire subarray. If drop_sample is False, set all values to NaN which do not belong to the mask (False).
    For lists of points, keep the current behavior of dropping single values.

    Args:
        ds (Dict[str, np.ndarray]): The dataset to filter. It should be a dictionary where keys are variable names and values are numpy arrays.
        filter_var (str): The variable name to use as the filter mask.
        drop_sample (bool): Boolean flag to determine whether to drop the entire subarray or set values to NaN.

    Returns:
        Dict[str, np.ndarray]: The filtered dataset.
    """
    if filter_var and filter_var in ds:
        valid_mask_lists = list()
        filter_mask = ds[filter_var]
        for key, value in ds.items():
            if key == filter_var or key == 'split': continue
            if value.ndim == 1:  # List of points
                valid_mask = ~np.isnan(value)
                valid_mask_lists.append(valid_mask)
            elif value.ndim in (2, 3, 4):
                if drop_sample:
                    axes_to_check = tuple(range(1, value.ndim))
                    valid_mask = np.all(filter_mask, axis=axes_to_check)
                    valid_mask_lists.append(valid_mask)
                else:
                    value_copy = value.copy()
                    value_copy[~filter_mask] = np.nan
                    ds[key] = value_copy
            else:
                return ds

        if len(valid_mask_lists) > 0:
            valid_mask = np.all(valid_mask_lists, axis=0)
            ds = {x: ds[x][valid_mask] for x in ds.keys()}

    return ds


def drop_nan_values(ds: Dict[str, np.ndarray], vars: list, filter_var: str = None) -> Dict[str, np.ndarray]:
    """
    Drop NaN values from the dataset. If any value in a subarray is NaN, drop the entire subarray.
    For lists of points, drop single NaN values. If filter_var is defined, it will use this mask
    to determine validity of subarrays.

    Args:
        ds (Dict[str, np.ndarray]): The dataset to filter. It should be a dictionary where keys are variable names and values are numpy arrays.
        vars (list): The variables to check for NaN values.
        filter_var (str): The name of the mask variable in the dataset. If None, drop the entire subarray based on NaN values alone.

    Returns:
        Dict[str, np.ndarray]: The filtered dataset.
    """
    mask_values = None
    if filter_var is not None and filter_var in ds:
        mask_values = ds[filter_var]
    valid_mask_lists = list()

    for var in vars:
        if var == 'split' or var == filter_var: continue
        if var in ds:
            value = ds[var]
            if value.ndim == 1:  # List of points
                # Create a mask where NaN values are marked as invalid
                valid_mask = ~np.isnan(value)
                valid_mask_lists.append(valid_mask)
            elif value.ndim in (2, 3, 4):  # Multi-dimensional arrays
                axes_to_check = tuple(range(1, value.ndim))
                if len(axes_to_check) == 1: axes_to_check = 1
                if mask_values is not None:
                    # Create a mask based on filter_var
                    valid_mask = np.any(ds[filter_var], axis=axes_to_check)
                    valid_mask_lists.append(valid_mask)
                    # Create a NaN mask
                    nan_mask = np.where(np.isnan(ds[var]), False, True)
                    valid_mask = np.any(nan_mask, axis=axes_to_check)
                    valid_mask_lists.append(valid_mask)
                    # Combine the NaN mask and the logical NOT of mask_values
                    lm = ~ds[filter_var]
                    nan_val = nan_mask | lm
                    # Final validation mask combining NaN mask and filter_var
                    validation_mask = np.where(nan_val, True, False)
                    valid_mask = np.all(validation_mask, axis=axes_to_check)

                    valid_mask_lists.append(valid_mask)
                else:
                    # If no filter_var, just check for NaNs
                    valid_mask_lists.append(~np.isnan(value).any(axis=axes_to_check))
    # Combine all masks using logical AND
    valid_mask = np.all(valid_mask_lists, axis=0)
    # Filter the dataset
    ds = {x: ds[x][valid_mask] for x in ds.keys()}

    return ds


def fill_masked_data(ds: Dict[str, np.ndarray], vars: List[str], method: str = 'mean', const: float | str | bool = None) -> Dict[str, np.ndarray]:
    """
    Fill NaN values in the dataset. If method is 'mean', fill NaNs with the mean value of the non-NaN values.
    If method is 'noise', fill NaNs with random noise within the range of the non-NaN values.
    If method is 'constant', fill NaNs with the specified constant value.

    Args:
        ds (Dict[str, np.ndarray]): The dataset to fill. It should be a dictionary where keys are variable names and values are numpy arrays.
        vars (List[str]): The variables to fill NaN values for.
        method (str): The method to use for filling NaN values. Options are 'mean', 'noise', or 'constant'.
        constant_value (float | str | bool): The constant value to use for filling NaN values when method is 'constant'.

    Returns:
        Dict[str, np.ndarray]: The dataset with NaN values filled.
    """
    for var in vars:
        if var in ds:
            value = ds[var]
            if np.isnan(value).any():
                if method == 'mean':
                    mean_value = np.nanmean(value)
                    ds[var] = np.where(np.isnan(value), mean_value, value)
                elif method == 'noise':
                    non_nan_values = value[~np.isnan(value)]
                    min_value = np.min(non_nan_values)
                    max_value = np.max(non_nan_values)
                    random_noise = np.random.uniform(min_value, max_value, size=value.shape)
                    ds[var] = np.where(np.isnan(value), random_noise, value)
                elif method == 'constant':
                    if const is not None:
                        ds[var] = np.where(np.isnan(value), const, value)
                    else:
                        raise ValueError("Constant value must be provided when method is 'constant'")
    return ds


def get_range(ds: xr.Dataset, var: str) -> List[float]:
    """
    Returns min and max values of the variable `var` of xarray dataset `ds`.

    Args:
        ds (xr.Dataset): The xarray dataset.
        var (str): The variable name to get the range for.

    Returns:
        List[float]: A list containing the min and max values.
    """
    data_var = ds[var]
    if isinstance(data_var, np.ndarray):
        min = np.nanmin(data_var)
        max = np.nanmax(data_var)
    else:
        min = data_var.min().values
        max = data_var.max().values
    return [min, max]


def get_statistics(ds: xr.Dataset, var: str) -> List[float]:
    """
    Returns mean and std values of the variable `var` of xarray dataset `ds`.

    Args:
        ds (xr.Dataset): The xarray dataset.
        var (str): The variable name to get the statistics for.

    Returns:
        List[float]: A list containing the mean and std values.
    """
    data_var = ds[var]
    if isinstance(data_var, np.ndarray):
        mean = np.nanmean(data_var)
        std = np.nanstd(data_var)
    else:
        mean = data_var.mean().compute().values.item()
        std = data_var.std().compute().values.item()

    return [mean, std]


def normalize(x: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    """
    Perform min-max feature scaling of `x`, shifting values to range [0,1].

    Args:
        x (np.ndarray): The array to normalize.
        xmin (float): The minimum value of the range.
        xmax (float): The maximum value of the range.

    Returns:
        np.ndarray: The normalized array.
    """
    return (x - xmin) / (xmax - xmin)


def standardize(x: np.ndarray, xmean: float, xstd: float) -> np.ndarray:
    """
    Transforms the distribution to mean 0 and variance 1.

    Args:
        x (np.ndarray): The array to standardize.
        xmean (float): The mean value for standardization.
        xstd (float): The standard deviation value for standardization.

    Returns:
        np.ndarray: The standardized array.
    """
    return (x - xmean) / xstd
