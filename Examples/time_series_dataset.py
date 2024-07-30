import xarray as xr
from xcube.core.store import new_data_store
from ml4xcube.data_split import assign_block_split
from ml4xcube.preprocessing import get_statistics, standardize
from ml4xcube.datasets.multiproc_sampler import MultiProcSampler


"""Prepare data for time series analysis"""

stats_dict = None


def standardize_data(chunk: xr.Dataset) -> xr.Dataset:
    """
    Standardize xarray chunks.

    Args:
        chunk (xr.Dataset): The data chunk to be standardized.

    Returns:
        xr.Dataset: The standardized data chunk.
    """
    global stats_dict
    chunk = standardize(chunk, stats_dict)

    return chunk


def prepare_dataset_creation() -> xr.Dataset:
    """
    Prepare the dataset for creation by fetching the data, calculating statistics, and assigning a land mask.

    Returns:
        xr.Dataset: The prepared dataset.
    """
    data_store = new_data_store("s3", root="esdl-esdc-v2.1.1", storage_options=dict(anon=True))
    dataset    = data_store.open_data('esdc-8d-0.083deg-184x270x270-2.1.1.zarr')
    start_time = "2002-05-21"
    end_time   = "2004-08-01"
    ds = dataset[["air_temperature_2m"]].sel(
        time=slice(start_time, end_time),
        lon=slice(0, 190),
        lat=slice(90, 0)
    )
    print("Dataset Dimensions:", ds.dims)


    global stats_dict
    stats_dict = get_statistics(ds, 'land_mask')
    print('Compute parameters for standardization:')
    print(f'air_temperature_2m: {stats_dict["air_temperature_2m"]}')

    # block sampling
    xds = assign_block_split(
        ds=ds,
        block_size=[("time", 12), ("lat", 135), ("lon", 135)],
        split=0.7
    )
    return xds


def create_datasets(ds: xr.Dataset) -> None:
    """
    Create dataset for distributed training.

    Args:
        ds (xr.Dataset): The prepared dataset.
    """
    # Preprocess data and split into training and testing sets
    train_set, test_set = MultiProcSampler(
        ds          = ds,
        train_cube  = 'train_cube.zarr',
        test_cube   = 'test_cube.zarr',
        sample_size = [('time', 6), ('lat', 16), ('lon', 16)],
        nproc       = 5,
        chunk_batch = 10,
        chunk_size  = (32, 6, 16, 16),
        array_dims  = ('samples', 'time', 'lat', 'lon'),
        callback_fn = standardize_data
    ).get_datasets()


def main() -> None:
    """
    Main function to prepare the dataset and create training/testing datasets.
    """
    ds = prepare_dataset_creation()
    create_datasets(ds)


if __name__ == "__main__":
    main()
