import numpy as np


def undo_normalizing(x: np.ndarray, xmin: float, xmax: float):
    """inverse operation of normalization"""
    return x*(xmax - xmin) + xmin


def undo_standardizing(x: np.ndarray, xmean: float, xstd: float):
    """inverse operation of standardization"""
    return x * xstd + xmean
