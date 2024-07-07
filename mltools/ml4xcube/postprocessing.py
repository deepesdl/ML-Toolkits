import numpy as np


def undo_normalizing(x: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    """
    Perform the inverse operation of normalization.

    Args:
        x (np.ndarray): The normalized array.
        xmin (float): The minimum value used for the original normalization.
        xmax (float): The maximum value used for the original normalization.

    Returns:
        np.ndarray: The denormalized array.
    """
    return x*(xmax - xmin) + xmin


def undo_standardizing(x: np.ndarray, xmean: float, xstd: float) -> np.ndarray:
    """
    Perform the inverse operation of standardization.

    Args:
        x (np.ndarray): The standardized array.
        xmean (float): The mean value used for the original standardization.
        xstd (float): The standard deviation value used for the original standardization.

    Returns:
        np.ndarray: The destandardized array.
    """
    return x * xstd + xmean
