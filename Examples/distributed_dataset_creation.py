import numpy as np
import xarray as xr
import dask.array as da
from global_land_mask import globe
from xcube.core.store import new_data_store
from ml4xcube.data_split import assign_block_split
from ml4xcube.datasets.multiproc_sampler import MultiProcSampler
from ml4xcube.preprocessing import get_statistics, standardize, assign_mask


"""Before performing distributed machine learning, run this script in order to create the training and the test set."""


# Initialize global variables to None
at_stat = None
lst_stat = None


def standardize_data(chunk: xr.Dataset) -> xr.Dataset:
    """
    Standardize xarray chunks.

    Args:
        chunk (xr.Dataset): The data chunk to be standardized.

    Returns:
        xr.Dataset: The standardized data chunk.
    """
    global at_stat, lst_stat
    print(at_stat)
    print(lst_stat)
    chunk['air_temperature_2m']       = standardize(chunk['air_temperature_2m'], *at_stat)
    chunk['land_surface_temperature'] = standardize(chunk['land_surface_temperature'], *lst_stat)

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
    end_time   = "2002-08-01"
    ds = dataset[["land_surface_temperature", "air_temperature_2m"]].sel(time=slice(start_time, end_time))

    global at_stat, lst_stat
    at_stat  = get_statistics(ds, 'air_temperature_2m')
    lst_stat = get_statistics(ds, 'land_surface_temperature')
    print('Compute parameters for standardization:')
    print(f'air_temperature_2m:       {at_stat}')
    print(f'land_surface_temperature: {lst_stat}')

    # Create a land mask and assign land mask
    lon_grid, lat_grid = np.meshgrid(ds.lon, ds.lat)
    land_mask          = da.from_array(globe.is_land(lat_grid, lon_grid))
    ds                 = assign_mask(ds, land_mask, 'land_mask', 'time')

    xds = assign_block_split(
        ds=ds,
        block_size=[("time", 12), ("lat", 135), ("lon", 135)],
        split=0.7
    )
    return ds


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
        nproc       = 5,
        chunk_batch = 10,
        data_fraq   = 0.01,
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
