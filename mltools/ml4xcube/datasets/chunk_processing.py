import numpy as np
from typing import Tuple, Dict, List
from ml4xcube.utils import split_chunk
from ml4xcube.preprocessing import apply_filter, drop_nan_values, fill_nan_values


def process_chunk(chunk: Dict[str, np.ndarray], use_filter: bool, drop_sample: bool, filter_var: str,
                  sample_size: List[Tuple[str, int]], overlap: List[Tuple[str, int]], fill_method: str,
                  const: float, mode: str = False) -> Tuple[Dict[str, np.ndarray], bool]:
    """
    Process a single chunk of data.

    Args:
        chunk (Dict[str, np.ndarray]): A dictionary containing the data chunk to preprocess.
        use_filter (bool): If true, apply the filter based on the specified filter_var.
        drop_sample (bool): If true, drop the entire subarray if any value in the subarray does not belong to the mask (False).
        filter_var (str): The variable to use for filtering.
        sample_size (List[Tuple[str, int]]): Sizes of the samples to be extracted from the chunk along each dimension.
                                                       Each tuple contains the dimension name and the size along that dimension.
        overlap (List[Tuple[str, int]]): Overlap for overlapping samples due to chunk splitting.
                                                   Each tuple contains the dimension name and the overlap fraction along that dimension.
        drop_nan_masked (bool): If true, NaN values are dropped using the mask specified by filter_var.
        fill_method (str): Method to fill masked data, if any.
        const (float): Constant value to use for filling masked data, if needed.
        all_nan_drop (bool)

    Returns:
        Tuple[Dict[str, np.ndarray], bool]: A tuple containing the preprocessed chunk and a boolean indicating if the chunk is valid.
    """
    cf = split_chunk(chunk, sample_size=sample_size, overlap=overlap)

    # Apply filtering based on the specified variable, if provided
    if use_filter:
        cft = apply_filter(cf, filter_var, drop_sample)
    else:
        cft = cf

    vars = list(cft.keys())
    cft = drop_nan_values(cft, vars, mode=mode, filter_var=filter_var)


    if fill_method is not None:
        vars = [var for var in cft.keys() if var != 'split' and var != filter_var]
        cft = fill_nan_values(cft, vars, fill_method, const)

    valid_chunk = all(not np.isnan(cft[var]).any() for var in cf)

    return cft, valid_chunk