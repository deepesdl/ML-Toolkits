import torch
import numpy as np
import xarray as xr
import dask.array as da
from global_land_mask import globe
from xcube.core.store import new_data_store
from ml4xcube.cube_utilities import get_chunk_sizes
from ml4xcube.preprocessing import get_statistics, standardize
from ml4xcube.datasets.multiproc_sampler import MultiProcSampler
from ml4xcube.datasets.pytorch import LargeScaleXrDataset, prepare_dataloader
from ml4xcube.training.pytorch_distributed import ddp_init, dist_train, Trainer


at_stat, lst_stat = None, None


def preprocess_data(chunk: xr.Dataset):
    """
    Standardize xarray chunks.
    """
    global at_stat, lst_stat

    chunk['air_temperature_2m'] = standardize(chunk['air_temperature_2m'], *at_stat)
    chunk['air_temperature_2m'] = standardize(chunk['land_surface_temperature'], *lst_stat)

    return chunk


def map_function(batch):
    """
    Convert list of samples (batch) into tensors X and y.
    X corresponds to 'air_temperature_2m' and y corresponds to 'land_surface_temperature'.
    """

    if len(batch) == 0:
        X = torch.empty((1, 0))
        y = torch.empty((1, 0))
    else:
        # Extract the arrays from the list of dictionaries
        air_temperature_2m_list = []
        land_surface_temperature_list = []

        for d in batch:
            if 'air_temperature_2m' in d and 'land_surface_temperature' in d:
                air_temperature_2m_list.append(d['air_temperature_2m'])
                land_surface_temperature_list.append(d['land_surface_temperature'])
            else:
                print("Error: Could not find both required arrays in the list of dictionaries.")
                return torch.empty((1, 0)), torch.empty((1, 0))

        # Stack the arrays along the first dimension (assuming they are 1D arrays)
        X = np.stack(air_temperature_2m_list, axis=0)
        y = np.stack(land_surface_temperature_list, axis=0)

        X = X.reshape(-1, 1)  # Making it [num_samples, 1]
        y = y.reshape(-1, 1)

    return torch.tensor(X), torch.tensor(y)


def load_train_objs():
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

    # Assign land mask to the dataset and split data into blocks
    ds = ds.assign(land_mask=(['time', 'lat', 'lon'], lm.rechunk(chunks=([v for _, v in get_chunk_sizes(ds)]))))
    #
    # Preprocess data and split into training and testing sets
    train_set, test_set = MultiProcSampler(ds, data_fraq=0.02).get_datasets()

    # Create PyTorch data sets
    train_ds = LargeScaleXrDataset(train_set)
    test_ds = LargeScaleXrDataset(test_set)

    # Initialize model and optimizer
    model     = torch.nn.Linear(in_features=1, out_features=1, bias=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    return train_ds, test_ds, model, optimizer

def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt", best_model_path: str = 'Best_Model.pt'):
    """Main function to run distributed training process."""

    # Initialize distributed data parallel training
    ddp_init()

    # Load training objects
    train_set, test_set, model, optimizer = load_train_objs()

    # Prepare data loaders
    train_loader = prepare_dataloader(train_set, batch_size, num_workers=5, parallel=True)
    test_loader  = prepare_dataloader(test_set, batch_size, num_workers=5, parallel=True)

    # Initialize the trainer and start training
    trainer = Trainer(model, train_loader, test_loader, optimizer, save_every, best_model_path, snapshot_path)
    dist_train(trainer, total_epochs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=10, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=7, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    main(args.save_every, args.total_epochs, args.batch_size)
