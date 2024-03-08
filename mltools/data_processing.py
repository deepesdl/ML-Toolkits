import numpy as np
import xarray as xr


def getStatistics(ds: xr.Dataset, var: str):
    """returns mean and std values of variable var of xarray ds"""
    x = ds[var].mean().values
    y = ds[var].std().values
    return [x,y]


def normalize(x: np.ndarray, xmin: float, xmax: float):
    """min-max feature scaling of x, shift values to range [0,1]"""
    return (x - xmin ) / (xmax - xmin)


def standardize(x: np.ndarray, xmean: float, xstd: float):
    """transforms distribution to mean 0 and variance 1"""
    return (x - xmean ) / xstd


def undo_normalizing(x: np.ndarray, xmin: float, xmax: float):
    """inverse operation of normalization"""
    return x*(xmax - xmin) + xmin


def undo_standardizing(x: np.ndarray, xmean: float, xstd: float):
    """inverse operation of standardization"""
    return x * xstd + xmean


def getRange(ds: xr.Dataset, var: str):
    """returns min and max values of variable var of xarray ds"""
    x = ds[var].min().values
    y = ds[var].max().values
    return [x,y]
