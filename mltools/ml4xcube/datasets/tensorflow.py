import random
import numpy as np
import xarray as xr
import tensorflow as tf
from typing import Tuple, List, Dict
from ml4xcube.cube_utilities import split_chunk
from ml4xcube.cube_utilities import get_chunk_by_index, calculate_total_chunks
from ml4xcube.preprocessing import apply_filter, drop_nan_values, fill_masked_data


class LargeScaleXrDataset:
    def __init__(self, xr_dataset: xr.Dataset, rand_chunk: bool = True, drop_nan: bool = True,
                 chunk_indices: list = None, drop_nan_masked: bool = False, use_filter: bool = True,
                 drop_sample: bool = False, fill_method: str = None, const: float = None,
                 filter_var: str = 'land_mask', num_chunks: int = None, callback_fn = None,
                 block_size: List[Tuple[str, int]] = None,
                 sample_size: List[Tuple[str, int]] = None,
                 overlap: List[Tuple[str, int]] = None):
        """
        Initialize the dataset for TensorFlow, managing large datasets efficiently.

        Attributes:
            ds (xr.Dataset): The xarray dataset.
            rand_chunk (bool): Whether to select chunks randomly.
            drop_nan (bool): Whether to drop NaN values.
            drop_nan_masked (bool): If true, NaN values are dropped using the mask specified by filter_var.
            use_filter (bool): If true, apply the filter based on the specified filter_var.
            drop_sample (bool): If true, drop the entire subarray if any value in the subarray does not belong to the mask (False).
            fill_method (str): Method to fill masked data, if any.
            const (float): Constant value to use for filling masked data, if needed.
            filter_var (str): Filtering variable name.
            num_chunks (int): Number of chunks to process dynamically.
            callback_fn (Callable): Function to apply to each chunk after preprocessing.
            block_size (List[Tuple[str, int]]): Block sizes for considered blocks (of (sub-)chunks).
            sample_size (ist[Tuple[str, int]]): Sample size for chunk splitting.
            overlap (List[Tuple[str, int]]): Overlap for overlapping samples due to chunk splitting.
            total_chunks (int): Total number of chunks in the dataset.
            chunk_indices (List[int]): List of indices specifying which chunks to process.
        """
        self.ds = xr_dataset
        self.rand_chunk = rand_chunk
        self.drop_nan = drop_nan
        self.drop_nan_masked = drop_nan_masked
        self.use_filter = use_filter
        self.drop_sample = drop_sample
        self.fill_method = fill_method
        self.const = const
        self.filter_var = filter_var
        self.num_chunks = num_chunks
        self.callback_fn = callback_fn
        self.block_size = block_size
        self.sample_size = sample_size
        self.overlap = overlap
        self.total_chunks = calculate_total_chunks(xr_dataset)
        if not chunk_indices is None:
            self.chunk_indices = chunk_indices
        elif num_chunks is not None and self.total_chunks >= num_chunks:
            self.chunk_indices = random.sample(range(self.total_chunks), num_chunks)
        else:
            self.chunk_indices = list(range(self.total_chunks))

    def __len__(self) -> int:
        """
        Return the number of chunks.

        Returns:
            int: Number of chunks.
        """
        return len(self.chunk_indices)

    def generate(self) -> Dict[str, np.ndarray]:
        """
        Generator function to yield chunks.

        Yields:
            dict: Processed chunk as a dictionary of NumPy arrays.
        """
        for idx in self.chunk_indices:
            chunk = get_chunk_by_index(self.ds, idx)
            cf = split_chunk(chunk, self.sample_size, overlap=self.overlap)

            if self.use_filter:
                cft = apply_filter(cf, self.filter_var, self.drop_sample)
            else:
                cft = cf

            if self.drop_nan:
                vars = list(cft.keys())
                if self.drop_nan_masked:
                    cft = drop_nan_values(cft, vars, self.filter_var)
                else:
                    cft = drop_nan_values(cft, vars)

            valid_chunk = all(np.nan_to_num(cft[var]).sum() > 0 for var in cf)

            if valid_chunk:
                if self.fill_method is not None:
                    vars = [var for var in cft.keys() if var != 'split' and var != self.filter_var]
                    cft = fill_masked_data(cft, vars, self.fill_method, self.const)

            if self.callback_fn:
                cft = self.callback_fn(cft)

            yield cft

    def get_dataset(self) -> tf.data.Dataset:
        """
        Creates a TensorFlow dataset from the generator.

        Returns:
            tf.data.Dataset: The TensorFlow Dataset object.
        """
        example_chunk = next(self.generate())
        if self.callback_fn is None:
            output_signature = {name: tf.TensorSpec(shape=(None,), dtype=tf.float32)
                                for name in example_chunk.keys()}
        else:
            output_signature = (
                tf.TensorSpec(shape=(None,None), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None), dtype=tf.float32)
            )

        return tf.data.Dataset.from_generator(
            self.generate,
            output_signature=output_signature
        )


def prepare_dataset(dataset: tf.data.Dataset, batch_size: int, shuffle: bool = True, num_parallel_calls: int = None, distributed: bool = False) -> tf.data.Dataset:
    """
    Prepares a TensorFlow dataset for training or evaluation.

    Args:
        dataset (tf.data.Dataset): The TensorFlow dataset from which to load the data.
        batch_size (int): How many samples per batch to load.
        shuffle (bool): Whether to shuffle the dataset. Defaults to True.
        num_parallel_calls (int, optional): How many threads to use for parallel processing of data loading. Defaults to None.
        distributed (bool): Specifies if distributed training is performed. Defaults to False.

    Returns:
        tf.data.Dataset: A TensorFlow Dataset object ready for iteration.
    """
    if distributed:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)

    if shuffle:
        # Shuffle the data, ideally the buffer size should be the size of the dataset, but it can be reduced if memory is limited
        dataset = dataset.shuffle(buffer_size=10000)  # Adjust buffer_size according to your dataset size and memory availability

    # Batching the dataset
    dataset = dataset.batch(batch_size)

    # Using `num_parallel_calls` with `prefetch` to improve pipeline performance
    if num_parallel_calls is None:
        num_parallel_calls = tf.data.experimental.AUTOTUNE

    dataset = dataset.prefetch(buffer_size=num_parallel_calls)

    return dataset
