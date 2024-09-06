import random
import numpy as np
import xarray as xr
from ml4xcube.splits import create_split
from typing import Tuple, List, Dict, Union, Callable
from ml4xcube.datasets.chunk_processing import process_chunk
from ml4xcube.utils import get_chunk_by_index, calculate_total_chunks
from ml4xcube.preprocessing import get_statistics, standardize, get_range, normalize


class XrDataset:
    def __init__(
        self, ds: xr.Dataset, chunk_indices: List[int] = None, rand_chunk: bool = True, drop_nan: str = 'auto',
        apply_mask: bool = True, drop_sample: bool = False, fill_method: str = None, const: float = None,
        filter_var: str = 'filter_mask', patience: int = 500, block_size: List[Tuple[str, int]] = None,
        sample_size: List[Tuple[str, int]] = None, overlap: List[Tuple[str, int]] = None, callback: Callable = None,
        num_chunks: int = None, to_pred: Union[str, List[str]] = None, scale_fn: str = 'standardize'
    ):
        """
        Creates a dataset of processed data chunks.

        Args:
            ds (xr.Dataset): The input xarray dataset.
            chunk_indices (List[int], optional): List of indices specifying which chunks to process. Defaults to None, indicating that chunks are chosen randomly.
            rand_chunk (bool, optional): If True, chunks are selected randomly when chunk_indices is None. Defaults to True.
            drop_nan (str, optional): Specifies how to handle NaN values in the data. Defaults to 'auto'.
                If 'auto', drop the entire sample if any NaN values are present.
                If 'if_all_nan', drop the sample if all values are NaN.
                If 'masked', drop the entire subarray if valid values according to the mask are NaN.
            apply_mask (bool): If True, apply a mask based on the filter_var to filter out invalid data. Defaults to True.
            drop_sample (bool): If true, NaN values are dropped during filter application.
            fill_method (str): The method to fill masked data. Defaults to None.
                If 'sample_mean', fill NaNs with the sample mean value.
                If 'mean', fill NaNs with the mean value of the non-NaN values.
                If 'noise', fill NaNs with random noise within the range of the non-NaN values.
                If 'constant', fill NaNs with the specified constant value.
            const (float): Constant value to fill masked data if fill_method is not specified. Defaults to None.
            filter_var (str): The variable containing the mask used for filtering invalid data (e.g., 'land_mask'). Defaults to 'land_mask'.
            patience (int): The number of consecutive iterations without finding a valid chunk before stopping the chunk retrieval process. Defaults to 500.
            block_size (List[Tuple[str, int]]): The block sizes for splitting the dataset into chunks. Each tuple contains the dimension name and block size. Defaults to None.
            sample_size (List[Tuple[str, int]]): The size of samples extracted from chunks, specified as a list of tuples with dimension names and sample sizes. Defaults to None.
            overlap (List[Tuple[str, float]]): Overlap size for creating overlapping samples from chunks. Each tuple contains the dimension name and overlap size. Defaults to None.
            callback (Callable): An optional callback function to apply additional processing to each chunk. Defaults to None.
            num_chunks (int): The number of unique chunks to process. If None, all chunks in the dataset will be processed. Defaults to None.
            to_pred (Union[str, List[str]]): The target variable(s) to predict. Used to split the dataset into features and target variables. Defaults to None.
            scale_fn (str): The scaling function to apply to the dataset.
                If 'standardize', standardization of the data is conducted.
                If 'normalize', the data is normalized.

        Attributes:
            total_chunks (int): The total number of chunks in the dataset, calculated based on the block_size.
            chunks (List[Dict[str, np.ndarray]]): The list of processed data chunks.
            dataset (Dict[str, np.ndarray]): The final concatenated dataset after processing all chunks.
            scale_params (Dict[str, Any]): Parameters used for scaling the dataset, based on the chosen scale_fn.
        """
        self.ds = ds
        self.chunk_indices = chunk_indices
        self.rand_chunk = rand_chunk
        self.drop_nan = drop_nan
        self.apply_mask = apply_mask
        self.drop_sample = drop_sample
        self.fill_method = fill_method
        self.const = const
        self.filter_var = filter_var
        self.patience = patience
        self.block_size = block_size
        self.sample_size = sample_size
        self.overlap = overlap
        self.callback = callback
        self.total_chunks = calculate_total_chunks(self.ds, self.block_size)
        if not self.chunk_indices is None:
            self.num_chunks = len(self.chunk_indices)
        elif num_chunks is not None and self.total_chunks >= num_chunks:
            self.num_chunks = num_chunks
        else:
            self.num_chunks = self.total_chunks
        self.scale_params = None
        self.to_pred = to_pred
        self.scale_fn = scale_fn

        self.chunks = None
        self.chunks = self.get_chunks()
        self.dataset = self.concatenate_chunks()
        if self.callback is not None:
            self.dataset = self.callback(self.dataset)
        if self.scale_fn is not None:
            self.scale_dataset()
        if self.to_pred is not None:
            self.dataset = self.split_dataset()

    def get_datasets(self) -> Union[Dict[str, np.ndarray], Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
        """
        Returns the processed dataset.

        Returns:
        Union[Dict[str, np.ndarray], Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
            If `to_pred` is None: Returns a dictionary where keys are variable names and values are concatenated numpy arrays representing the dataset.
            If `to_pred` is provided: Returns a tuple containing:
                (X_train, y_train): Training features and targets.
                (X_test, y_test): Testing features and targets.
        """
        return self.dataset

    def split_dataset(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Split the dataset into training and testing sets.

        Returns:
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
                A tuple containing:
                - (X_train, y_train): Training features and targets.
                - (X_test, y_test): Testing features and targets.
        """
        stack_axis = -1
        if self.sample_size is not None: stack_axis = 1

        X_train, X_test, y_train, y_test = create_split(
            data       = self.dataset,
            to_pred    = self.to_pred,
            filter_var = self.filter_var,
            stack_axis = stack_axis
        )

        train_data, test_data = (X_train, y_train), (X_test, y_test)
        print('set train and test data')
        return train_data, test_data

    def scale_dataset(self) -> None:
        """
        Scale the dataset using the specified scaling function (standardize or normalize).
            If 'standardize': Standardizes the dataset using mean and standard deviation.
            If 'normalize': Normalizes the dataset to a specified range (usually 0-1).

        Returns:
            None
        """
        if self.scale_fn == 'standardize':
            self.scale_params = get_statistics(self.dataset)
            self.dataset = standardize(self.dataset, self.scale_params)
        elif self.scale_fn == 'normalize':
            self.scale_params = get_range(self.dataset)
            self.dataset = normalize(self.dataset, self.scale_params)

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

        return concatenated_chunks

    def preprocess_chunk(self, chunk: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], bool]:
        """
        Preprocess a single chunk of data.

        Args:
            chunk (Dict[str, np.ndarray]): A dictionary containing the data chunk to preprocess.

        Returns:
            Tuple[Dict[str, np.ndarray], bool]: A tuple containing the preprocessed chunk and a boolean indicating if the chunk is valid.
        """
        cft, valid_chunk = process_chunk(chunk, self.apply_mask, self.drop_sample, self.filter_var, self.sample_size,
                                         self.overlap, self.fill_method, self.const, self.drop_nan)
        return cft, valid_chunk

    def get_chunks(self) -> List[Dict[str, np.ndarray]]:
        """
        Retrieve specific chunks of data from a dataset.

        Returns:
            List[Dict[str, np.ndarray]]: A list of processed data chunks.
        """
        chunks_idx = list()
        not_valid_chunks = list()
        chunks_list = list()
        chunk_index = 0
        no_valid_chunk_count = 0
        iterations = 0

        # Process chunks until 'num_chunks' unique chunks have been processed
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
            if self.chunk_indices is None and chunk_index != self.total_chunks:
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
                if chunk_index not in not_valid_chunks: not_valid_chunks.append(chunk_index)

            if len(chunks_idx) + len(not_valid_chunks) == self.total_chunks: break

            chunk_index += 1

        return chunks_list




