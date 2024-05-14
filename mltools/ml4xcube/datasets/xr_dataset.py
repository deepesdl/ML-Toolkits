import random
import numpy as np
import xarray as xr
from ml4xcube.preprocessing import apply_filter, drop_nan_values
from ml4xcube.cube_utilities import get_chunk_by_index, calculate_total_chunks


class XrDataset():
    def __init__(self, ds: xr.Dataset, num_chunks: int, rand_chunk: bool = True,
                 drop_nan: bool = True, strict_nan: bool = False,
                 filter_var: str = 'land_mask', patience: int = 500):
        """
        Initialize xarray dataset.

        Args:
            ds (xr.Dataset): The input dataset.
            num_chunks (int): The number of unique chunks to process.
            rand_chunk (bool): If true, chunks are chosen randomly.
            drop_nan (bool): If true, NaN values are dropped.
            strict_nan (bool): If true, discard the entire chunk if any NaN is found in any variable.
            filter_var (str): The variable to use for filtering. Defaults to 'land_mask'.
                              If None, no filtering is applied.
            patience (int): The number of consecutive iterations without a valid chunk before stopping.

        Returns:
            list: A list of processed data chunks.
        """
        self.ds = ds
        self.rand_chunk = rand_chunk
        self.drop_nan = drop_nan
        self.strict_nan = strict_nan
        self.filter_var = filter_var
        self.patience = patience
        self.total_chunks = calculate_total_chunks(self.ds)

        if self.total_chunks >= num_chunks:
            self.num_chunks = num_chunks
        else:
            self.num_chunks = self.total_chunks

        self.chunks = self.get_chunks()
        self.dataset = self.concatenate_chunks()

    def get_dataset(self):
        return self.dataset

    def concatenate_chunks(self):
        concatenated_chunks = {}

        # Get the keys of the first dictionary in self.chunks
        keys = list(self.chunks[0].keys())

        # Loop over the keys and concatenate the arrays along the time dimension
        for key in keys:
            concatenated_chunks[key] = np.concatenate([chunk[key] for chunk in self.chunks], axis=0)

        return concatenated_chunks

    def preprocess_chunk(self, chunk):
        # Flatten the data and select only land values, then drop NaN values
        cf = {x: chunk[x].ravel() for x in chunk.keys()}

        # Apply filtering based on the specified variable, if provided
        cft = apply_filter(cf, self.filter_var)

        valid_chunk = True

        if self.drop_nan:
            vars = list(cft.keys())
            cft = drop_nan_values(cft, vars)
            valid_chunk = all(np.nan_to_num(cft[var]).sum() > 0 for var in cf)
            if self.strict_nan:
                valid_chunk = any(np.nan_to_num(cft[var]).sum() > 0 for var in cf)

        return cft, valid_chunk

    def get_chunks(self):
        """
        Retrieve specific chunks of data from a dataset.
        Returns:
            list: A list of processed data chunks.
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

            if self.rand_chunk:
                chunk_index = random.randint(0, self.total_chunks - 1)  # Select a random chunk index

            if chunk_index in chunks_idx:
                continue  # Skip if this chunk has already been processed

            # Retrieve the chunk by its index
            chunk = get_chunk_by_index(self.ds, chunk_index)

            cft, valid_chunk = self.preprocess_chunk(chunk)

            if valid_chunk:
                chunks_idx.append(chunk_index)
                chunks_list.append(cft)
                no_valid_chunk_count = 0  # reset the patience counter after finding a valid chunk
            else:
                no_valid_chunk_count += 1  # increment the patience counter if no valid chunk is found

            chunk_index += 1
        return chunks_list


