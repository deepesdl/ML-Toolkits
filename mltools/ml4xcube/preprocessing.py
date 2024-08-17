import numpy as np
import xarray as xr
import dask.array as da
from typing import Dict, List, Union
from ml4xcube.utils import get_chunk_sizes


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


def drop_nan_values(ds: Dict[str, np.ndarray], vars: List[str], mode: str = 'auto', filter_var: str = 'filter_mask') -> Dict[str, np.ndarray]:
    """
    Drop NaN values from the dataset. If any value in a subarray is NaN, drop the entire subarray.
    For lists of points, drop single NaN values. If filter_var is defined, it will use this mask
    to determine validity of a sample.

    Args:
        ds (Dict[str, np.ndarray]): The dataset to filter. It should be a dictionary where keys are variable names and values are numpy arrays.
        vars (List[str]): The variables to check for NaN values.
        mode (str): Defines the means by which areas with missing values are dropped
            If 'auto', drop the entire sample if any NaN is contained.
            If 'if_all_nan', drop the sample if entirely NaN.
            If 'masked', drop the entire subarray if valid values according to mask are NaN.
        filter_var (str): The name of the mask variable in the dataset. Required if mode is 'masked'.

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
                if mode == 'masked' and mask_values is not None:
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
                elif mode == 'if_all_nan':
                    valid_mask_lists.append(~np.isnan(value).all(axis=axes_to_check))
                else:
                    # If no filter_var, just check for NaNs
                    valid_mask_lists.append(~np.isnan(value).any(axis=axes_to_check))
    # Combine all masks using logical AND
    valid_mask = np.all(valid_mask_lists, axis=0)
    # Filter the dataset
    ds = {x: ds[x][valid_mask] for x in ds.keys()}

    return ds


def fill_nan_values(ds: Union[Dict[str, np.ndarray], xr.Dataset], vars: List[str], method: str = 'mean', const: float | str | bool = None) -> Union[Dict[str, np.ndarray], xr.Dataset]:
    """
    Fill NaN values in the dataset.

    Args:
        ds (Union[Dict[str, np.ndarray], xr.Dataset]): The dataset to fill.
        vars (List[str]): The variables to fill NaN values for.
        method (str): The method to use for filling NaN values.
            If 'sample_mean', fill NaNs with the sample mean value.
            If 'mean', fill NaNs with the mean value of the non-NaN values.
            If 'noise', fill NaNs with random noise within the range of the non-NaN values.
            If 'constant', fill NaNs with the specified constant value.
        const (float | str | bool): The constant value to use for filling when method is 'constant'.

    Returns:
        Dict[str, np.ndarray]: The dataset with NaN values filled.
    """
    for var in vars:
        if isinstance(ds, dict):
            if var in ds:
                value = ds[var]
                if np.isnan(value).any():
                    if method == 'mean':
                        mean_value = np.nanmean(value)
                        ds[var] = np.where(np.isnan(value), mean_value, value)
                    elif method == 'sample_mean':
                        axes_to_check = tuple(range(1, value.ndim))
                        if len(axes_to_check) == 1: axes_to_check = 1
                        mean_value = np.nanmean(value, axis=axes_to_check, keepdims=True)
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


def get_range(ds: Union[xr.Dataset, Dict[str, np.ndarray]], exclude_vars:List[str] = list()) -> Dict[str, List[float]]:
    """
    Returns min and max values for all variables in the xarray dataset or dictionary.

    Args:
        ds (Union[xr.Dataset, Dict[str, np.ndarray]]): The xarray dataset or dictionary of variables.
        exclude_vars (List[str]): Variable name to exclude from calculation, such as a mask variable (e.g., 'land_mask').

    Returns:
        Dict[str, List[float]]: A dictionary containing the min and max values for each variable.
    """
    data_vars = ds.data_vars if isinstance(ds, xr.Dataset) else ds.keys()
    range_dict = {}
    for var in data_vars:
        if var == 'split' or var == 'filter_mask' or var in exclude_vars: continue
        data_var = ds[var]
        min_val = np.nanmin(data_var) if isinstance(data_var, np.ndarray) else data_var.min().values.item()
        max_val = np.nanmax(data_var) if isinstance(data_var, np.ndarray) else data_var.max().values.item()
        range_dict[var] = [min_val, max_val]
    return range_dict


def get_statistics(ds: Union[xr.Dataset, Dict[str, np.ndarray]], exclude_vars:List[str] = list()) -> Dict[str, List[float]]:
    """
    Returns mean and std values for all variables in the xarray dataset or dictionary.

    Args:
        ds (Union[xr.Dataset, Dict[str, np.ndarray]]): The xarray dataset or dictionary of variables.
        exclude_vars (str): Variable name to exclude from calculation, such as a mask variable (e.g., 'land_mask').

    Returns:
        Dict[str, List[float]]: A dictionary containing the mean and std values for each variable.
    """
    data_vars = ds.data_vars if isinstance(ds, xr.Dataset) else ds.keys()
    stats_dict = {}
    for var in data_vars:
        if var == 'split' or var == 'filter_mask' or var in exclude_vars: continue
        data_var = ds[var]
        mean_val = np.nanmean(data_var) if isinstance(data_var, np.ndarray) else data_var.mean().values.item()
        std_val = np.nanstd(data_var) if isinstance(data_var, np.ndarray) else data_var.std().values.item()
        stats_dict[var] = [mean_val, std_val]
    return stats_dict


def normalize(ds: Union[xr.Dataset, Dict[str, np.ndarray]], range_dict: Dict[str, List[float]], filter_var: str = None) -> Union[xr.Dataset, Dict[str, np.ndarray]]:
    """
    Normalize all variables in the dataset or dictionary using the provided range dictionary.

    Args:
        ds (Union[xr.Dataset, Dict[str, np.ndarray]]): The xarray dataset  or dictionary to normalize.
        range_dict (Dict[str, List[float]]): Dictionary with min and max values for each variable.
        filter_var (str): Variable name to exclude from normalization, such as a mask variable (e.g., 'land_mask').

    Returns:
        Union[xr.Dataset, Dict[str, np.ndarray]]: The normalized dataset or dictionary.
    """
    normalized_ds = ds.copy()
    data_vars = normalized_ds.data_vars if isinstance(normalized_ds, xr.Dataset) else normalized_ds.keys()
    for var in data_vars:
        if var == 'split' or var == filter_var: continue
        if var in range_dict:
            xmin, xmax = range_dict[var]
            if xmax != xmin:
                normalized_data = (ds[var] - xmin) / (xmax - xmin)
                if isinstance(normalized_ds, xr.Dataset):
                    normalized_ds[var] = normalized_data
                else:
                    normalized_ds[var] = normalized_data
            else:
                normalized_ds[var] = ds[var]  # If xmin == xmax, normalization isn't possible
    return normalized_ds


def standardize(ds: Union[xr.Dataset, Dict[str, np.ndarray]], stats_dict: Dict[str, List[float]], filter_var: str = None) -> Union[xr.Dataset, Dict[str, np.ndarray]]:
    """
    Standardize all variables in the dataset or dictionary using the provided statistics dictionary.

    Args:
        ds (Union[xr.Dataset, Dict[str, np.ndarray]]): The xarray dataset or dictionary to standardize.
        stats_dict (Dict[str, List[float]]): Dictionary with mean and std values for each variable.
        filter_var (str): Variable name to exclude from standardization, such as a mask variable (e.g., 'land_mask').

    Returns:
        Union[xr.Dataset, Dict[str, np.ndarray]]: The standardized dataset or dictionary.
    """
    standardized_ds = ds.copy()
    data_vars = standardized_ds.data_vars if isinstance(standardized_ds, xr.Dataset) else standardized_ds.keys()
    for var in data_vars:
        if var == 'split' or var == filter_var: continue
        if var in stats_dict:
            xmean, xstd = stats_dict[var]
            if xstd != 0:
                standardized_data = (ds[var] - xmean) / xstd
                if isinstance(standardized_ds, xr.Dataset):
                    standardized_ds[var] = standardized_data
                else:
                    standardized_ds[var] = standardized_data
            else:
                standardized_ds[var] = ds[var] - xmean  # If xstd == 0, standardization isn't possible
    return standardized_ds


def assign_mask(ds: xr.Dataset, mask: da.Array, mask_name: str = None, stack_dim: str = 'time') -> xr.Dataset:
    """
    Assign a mask to a dataset, expanding it along a specified dimension if provided.

    Args:
        ds (xr.Dataset): The dataset to which the mask will be assigned.
        mask (da.Array): The mask array to be assigned to the dataset.
        mask_name (str): The name to be used for the mask variable in the dataset. filter_mask if None
        stack_dim (str): The dimension along which to expand the mask. If None, the mask is not expanded. Default is None.

    Returns:
        xr.Dataset: The dataset with the mask assigned.

    Raises:
        ValueError: If the specified stack dimension is not present in the dataset dimensions.

    Notes:
        - The function validates that the specified stack dimension is present in the dataset.
        - If a stack dimension is specified, the mask is expanded along this dimension.
        - The mask is rechunked to align with the dataset's chunk sizes for the common dimensions.
    """
    # Validate that the stack_dim is a dimension in the dataset
    if stack_dim and stack_dim not in ds.dims:
        raise ValueError(f"The specified stack dimension '{stack_dim}' is not present in the dataset dimensions.")

    cube_dims = list(ds.dims)

    # If a stacking dimension is specified and it exists in the dataset
    if stack_dim:
        # Calculate the axis index for the specified stack dimension
        stack_dim_index = cube_dims.index(stack_dim)

        # Expand the mask across the specified dimension
        mask = da.stack([mask for _ in range(ds.sizes[stack_dim])], axis=stack_dim_index)

    # Rechunk the mask to align with the dataset's chunk sizes for the common dimensions
    chunk_sizes=([v for k, v in get_chunk_sizes(ds)])
    mask = mask.rechunk(chunks=chunk_sizes)

    # Set mask name
    if mask_name is None: mask_name = 'filter_mask'

    # Assign the mask to the dataset using the provided mask name
    xds = ds.assign({
        mask_name: (cube_dims, mask)
    })

    return xds