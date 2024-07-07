import random
import numpy as np
import xarray as xr
from typing import Tuple, List, Dict
from ml4xcube.preprocessing import apply_filter, drop_nan_values, fill_masked_data
from ml4xcube.cube_utilities import get_chunk_by_index, calculate_total_chunks, split_chunk


class XrDataset():
    def __init__(self, ds: xr.Dataset, chunk_indices: list = None,
                 rand_chunk: bool = True, drop_nan: bool = True,
                 strict_nan: bool = False, drop_nan_masked: bool = False,
                 use_filter: bool = True, drop_sample: bool = False,
                 fill_method: str = None, const: float = None,
                 filter_var: str = 'land_mask', patience: int = 500,
                 block_size: List[Tuple[str, int]] = None,
                 sample_size: List[Tuple[str, int]] = None,
                 overlap: List[Tuple[str, int]] = None,
                 num_chunks: int = None):
        """
        Creates a dataset of processed data chunks.

        Attributes:
            ds (xr.Dataset): The input dataset.
            chunk_indices (List[int]): List of indices specifying which chunks to process.
            rand_chunk (bool): If true, chunks are chosen randomly.
            drop_nan (bool): If true, NaN values are dropped.
            strict_nan (bool): If true, discard the entire chunk if any NaN is found in any variable.
            drop_nan_masked (bool): If true, NaN values are dropped using the mask specified by filter_var.
            use_filter (bool): If true, apply the filter based on the specified filter_var.
            drop_sample (bool): If true, drop the entire subarray if any value in the subarray does not belong to the mask (False).
            fill_method (str): Method to fill masked data, if any.
            const (float): Constant value to use for filling masked data, if needed.
            filter_var (str): The variable to use for filtering.
            patience (int): The number of consecutive iterations without a valid chunk before stopping.
            block_size (List[Tuple[str, int]]): Block sizes for considered blocks (of (sub-)chunks).
            sample_size (List[Tuple[str, int]]): Sample size for chunk splitting.
            overlap (List[Tuple[str, int]]): Overlap for overlapping samples due to chunk splitting.
            total_chunks (int): Total number of chunks in the dataset.
            num_chunks (int): The number of unique chunks to process.
            chunks (List[Dict[str, np.ndarray]]): List of processed data chunks.
            dataset (Dict[str, np.ndarray]): Concatenated dataset from the processed chunks.
        """
        self.ds = ds
        self.chunk_indices = chunk_indices
        self.rand_chunk = rand_chunk
        self.drop_nan = drop_nan
        self.strict_nan = strict_nan
        self.drop_nan_masked = drop_nan_masked
        self.use_filter = use_filter
        self.drop_sample = drop_sample
        self.fill_method = fill_method
        self.const = const
        self.filter_var = filter_var
        self.patience = patience
        self.block_size = block_size
        self.sample_size = sample_size
        self.overlap = overlap
        self.total_chunks = calculate_total_chunks(self.ds, self.block_size)

        if not self.chunk_indices is None:
            self.num_chunks = len(self.chunk_indices)
        elif num_chunks is not None and self.total_chunks >= num_chunks:
            self.num_chunks = num_chunks
        else:
            self.num_chunks = self.total_chunks

        self.chunks = None
        self.chunks = self.get_chunks()
        self.dataset = self.concatenate_chunks()

    def get_dataset(self) -> Dict[str, np.ndarray]:
        """
        Get the processed dataset.

        Returns:
            Dict[str, np.ndarray]: Concatenated dataset from the processed chunks.
        """
        return self.dataset

    def concatenate_chunks(self) -> Dict[str, np.ndarray]:
        """
        Concatenate the chunks along the time dimension.

        Returns:
            Dict[str, np.ndarray]: A dictionary of concatenated data chunks.
        """
        concatenated_chunks = {}

        # Get the keys of the first dictionary in self.chunks
        keys = list(self.chunks[0].keys())

        # Loop over the keys and concatenate the arrays along the time dimension
        for key in keys:
            concatenated_chunks[key] = np.concatenate([chunk[key] for chunk in self.chunks], axis=0)

        print(concatenated_chunks)

        return concatenated_chunks

    def preprocess_chunk(self, chunk: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], bool]:
        """
        Preprocess a single chunk of data.

        Args:
            chunk (Dict[str, np.ndarray]): A dictionary containing the data chunk to preprocess.

        Returns:
            Tuple[Dict[str, np.ndarray], bool]: A tuple containing the preprocessed chunk and a boolean indicating if the chunk is valid.
        """
        # Split a chunk into samples
        cf = split_chunk(chunk, self.sample_size, overlap=self.overlap)

        # Apply filtering based on the specified variable, if provided
        if self.use_filter:
            cft = apply_filter(cf, self.filter_var, self.drop_sample)
        else:
            cft = cf

        valid_chunk = True

        if self.drop_nan:
            vars = list(cft.keys())
            if self.drop_nan_masked:
                cft = drop_nan_values(cft, vars, self.filter_var)
            else:
                cft = drop_nan_values(cft, vars)
            valid_chunk = all(np.nan_to_num(cft[var]).sum() > 0 for var in cf)
            if self.strict_nan:
                valid_chunk = any(np.nan_to_num(cft[var]).sum() > 0 for var in cf)

        if valid_chunk:
            if self.fill_method is not None:
                vars = [var for var in cft.keys() if var != 'split' and var != self.filter_var]
                cft = fill_masked_data(cft, vars, self.fill_method, self.const)

        return cft, valid_chunk

    def get_chunks(self) -> List[Dict[str, np.ndarray]]:
        """
        Retrieve specific chunks of data from a dataset.

        Returns:
            List[Dict[str, np.ndarray]]: A list of processed data chunks.
        """
        chunks_idx = list()
        chunks_list = []
        chunk_index = 0
        no_valid_chunk_count = 0
        iterations = 0

        # Process chunks until 3 unique chunks have been processed
        while len(chunks_idx) < self.num_chunks:
            iterations += 1

            if no_valid_chunk_count >= self.patience:
                print("Patience threshold reached, returning collected chunks.")
                break

            if self.rand_chunk and self.chunk_indices is None:
                chunk_index = random.randint(0, self.total_chunks - 1)  # Select a random chunk index

            if chunk_index in chunks_idx:
                continue  # Skip if this chunk has already been processed

            # Retrieve the chunk by its index
            if self.chunk_indices is None:
                chunk = get_chunk_by_index(self.ds, chunk_index)
            else:
                chunk = get_chunk_by_index(self.ds, self.chunk_indices[chunk_index])

            cft, valid_chunk = self.preprocess_chunk(chunk)

            if valid_chunk:
                chunks_idx.append(chunk_index)
                chunks_list.append(cft)
                no_valid_chunk_count = 0  # reset the patience counter after finding a valid chunk
            else:
                no_valid_chunk_count += 1  # increment the patience counter if no valid chunk is found

            chunk_index += 1

        return chunks_list




