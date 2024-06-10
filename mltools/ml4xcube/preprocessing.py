import numpy as np
import xarray as xr


def apply_filter(ds, filter_var):
    if filter_var and filter_var in ds:
        filter_mask = ds[filter_var]
        filtered_ds = {x: ds[x][filter_mask == True] for x in ds.keys()}
    else:
        filtered_ds = ds
    return filtered_ds


def drop_nan_values(ds, vars):
    for var in vars:
        ds = {x: ds[x][~np.isnan(ds[var])] for x in vars}
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
