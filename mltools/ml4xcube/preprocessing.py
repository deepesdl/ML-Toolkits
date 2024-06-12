import numpy as np
import xarray as xr
from typing import Dict


def apply_filter(ds, filter_var):
    """
    Apply a filter to the dataset. If any value in a subarray does not belong to the mask (False),
    drop the entire subarray. For lists of points, keep the current behavior of dropping single values.

    Parameters:
    - ds: The dataset to filter. It should be a dictionary where keys are variable names and values are numpy arrays.
    - filter_var: The variable name to use as the filter mask.

    Returns:
    The filtered dataset.
    """
    if filter_var and filter_var in ds:
        filter_mask = ds[filter_var]
        filtered_ds = {}

        for key, value in ds.items():
            if value.ndim == 1:  # List of points
                filtered_ds[key] = value[filter_mask == True]
                #filtered_ds = {x: ds[x][filter_mask == True] for x in ds.keys()}
            elif value.ndim in (3, 4):
                axes_to_check = tuple(range(1, value.ndim))
                valid_subarray_mask = np.all(filter_mask, axis=axes_to_check)
                filtered_ds[key] = value[valid_subarray_mask]
            else:
                filtered_ds[key] = value
            """elif value.ndim == 3:  # 2D tensor (batch, lat, lon)
                valid_subarray_mask = np.all(filter_mask, axis=(1, 2))  # Check if entire (lat, lon) subarray is valid
                filtered_ds[key] = value[valid_subarray_mask]
            elif value.ndim == 4:  # 3D tensor
                valid_subarray_mask = np.all(filter_mask, axis=(1, 2, 3))  # Check if entire subarray is valid
                filtered_ds[key] = value[valid_subarray_mask]"""

    else:
        filtered_ds = ds

    return filtered_ds


def drop_nan_values(ds: Dict[str, np.ndarray], vars: list) -> Dict[str, np.ndarray]:
    """
    Drop NaN values from the dataset. If any value in a subarray is NaN,
    drop the entire subarray. For lists of points, drop single NaN values.

    Parameters:
    - ds: The dataset to filter. It should be a dictionary where keys are variable names and values are numpy arrays.
    - vars: The variables to check for NaN values.

    Returns:
    The filtered dataset.
    """
    for var in vars:
        if var in ds:
            value = ds[var]
            if value.ndim == 1:  # List of points
                valid_mask = ~np.isnan(value)
                ds = {x: ds[x][valid_mask] for x in ds.keys()}
            elif value.ndim in (3, 4):  # 2D or 3D tensor
                axes_to_check = tuple(range(1, value.ndim))
                valid_subarray_mask = ~np.isnan(value).any(axis=axes_to_check)
                ds = {x: ds[x][valid_subarray_mask] for x in ds.keys()}
    return ds


def get_range(ds: xr.Dataset, var: str):
    """returns min and max values of variable var of xarray ds"""
    data_var = ds[var]
    if isinstance(data_var, np.ndarray):
        min = data_var.min().item()
        max = data_var.max().item()
    else:
        min = data_var.min().values
        max = data_var.max().values
    return [min, max]


def get_statistics(ds: xr.Dataset, var: str):
    """returns mean and std values of variable var of xarray ds"""
    data_var = ds[var]
    if isinstance(data_var, np.ndarray):
        mean = data_var.mean().item()
        std = data_var.std().item()
    else:
        mean = data_var.mean().compute().values.item()
        std = data_var.std().compute().values.item()

    return [mean, std]


def normalize(x: np.ndarray, xmin: float, xmax: float):
    """min-max feature scaling of x, shift values to range [0,1]"""
    return (x - xmin) / (xmax - xmin)


def standardize(x: np.ndarray, xmean: float, xstd: float):
    """transforms distribution to mean 0 and variance 1"""
    return (x - xmean) / xstd
