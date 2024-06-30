import numpy as np
import xarray as xr
import dask.array as da
from global_land_mask import globe
from xcube.core.store import new_data_store
from ml4xcube.cube_utilities import get_chunk_sizes
from ml4xcube.preprocessing import get_statistics, standardize
from ml4xcube.datasets.multiproc_sampler import MultiProcSampler


at_stat, lst_stat = None, None


def standardize_data(chunk: xr.Dataset):
    """
    Standardize xarray chunks.
    """
    global at_stat, lst_stat

    chunk['air_temperature_2m'] = standardize(chunk['air_temperature_2m'], *at_stat)
    chunk['air_temperature_2m'] = standardize(chunk['land_surface_temperature'], *lst_stat)

    return chunk


def create_datasets():
    """
    Create dataset for distributed training
    """
    data_store = new_data_store("s3", root="esdl-esdc-v2.1.1", storage_options=dict(anon=True))
    dataset    = data_store.open_data('esdc-8d-0.083deg-184x270x270-2.1.1.zarr')
    start_time = "2002-05-21"
    end_time   = "2002-08-01"
    ds         = dataset[["land_surface_temperature", "air_temperature_2m"]].sel(time=slice(start_time, end_time))

    global at_stat, lst_stat
    at_stat  = get_statistics(ds, 'air_temperature_2m')
    lst_stat = get_statistics(ds, 'land_surface_temperature')

    # Create a land mask
    lon_grid, lat_grid = np.meshgrid(ds.lon, ds.lat)
    lm0                = da.from_array(globe.is_land(lat_grid, lon_grid))
    lm                 = da.stack([lm0 for _ in range(ds.dims['time'])], axis=0)

    # Assign land mask to the dataset
    ds = ds.assign(land_mask=(['time', 'lat', 'lon'], lm.rechunk(chunks=([v for _, v in get_chunk_sizes(ds)]))))

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


def main():
    create_datasets()


if __name__ == "__main__":
    main()
