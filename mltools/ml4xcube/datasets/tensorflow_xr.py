import random
import xarray as xr
import tensorflow as tf

from ml4xcube.cube_utilities import get_chunk_by_index, calculate_total_chunks
from ml4xcube.preprocessing import apply_filter, drop_nan_values

class LargeScaleXrDataset:
    def __init__(self, xr_dataset: xr.Dataset, num_chunks: int, rand_chunk: bool = True,
                 drop_nan: bool = True, filter_var: str = 'land_mask'):
        """
        Initialize the dataset for TensorFlow, managing large datasets efficiently.

        Args:
            xr_dataset (xr.Dataset): The xarray dataset.
            num_chunks (int): Number of chunks to process dynamically.
            rand_chunk (bool): Whether to select chunks randomly.
            drop_nan (bool): Whether to drop NaN values.
            filter_var (str): Filtering variable name, default 'land_mask'.
        """
        self.ds = xr_dataset
        self.num_chunks = num_chunks
        self.rand_chunk = rand_chunk
        self.drop_nan = drop_nan
        self.filter_var = filter_var
        self.total_chunks = calculate_total_chunks(xr_dataset)
        if self.total_chunks >= num_chunks:
            self.chunk_indices = random.sample(range(self.total_chunks), num_chunks)
        else:
            self.chunk_indices = list(range(self.total_chunks))


    def generate(self):
        """
        Generator function to yield chunks.

        Yields:
            Processed chunk as a dictionary of NumPy arrays.
        """
        for idx in self.chunk_indices:
            chunk = get_chunk_by_index(self.ds, idx)

            # Process the chunk
            cf = {x: chunk[x].ravel() for x in chunk.keys()}

            cft = apply_filter(cf, self.filter_var)

            if self.drop_nan:
                cft = drop_nan_values(cft, list(cft.keys()))

            yield cft

    def get_dataset(self):
        """
        Creates a TensorFlow dataset from the generator.

        Returns:
            tf.data.Dataset: The TensorFlow Dataset object.
        """
        return tf.data.Dataset.from_generator(
            self.generate,
            output_signature={name: tf.TensorSpec(shape=(None,), dtype=tf.float32)
                              for name in next(self.generate()).keys()}  # Assumes all data are float32
        )


def prepare_dataset(dataset: tf.data.Dataset, batch_size: int, shuffle: bool = True, num_parallel_calls: int = None, distributed: bool = False) -> tf.data.Dataset:
    """
    Prepares a TensorFlow dataset for training or evaluation.

    Parameters:
    - dataset: The TensorFlow dataset from which to load the data.
    - batch_size: How many samples per batch to load.
    - shuffle: Whether to shuffle the dataset.
    - num_parallel_calls: How many threads to use for parallel processing of data loading.
    - distributed: Specifies if distributed training is performed.

    Returns:
    A TensorFlow Dataset object ready for iteration.
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
