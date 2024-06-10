import torch
import numpy as np
import xarray as xr
import dask.array as da
from torch import nn
from global_land_mask import globe
from torch.utils.data import TensorDataset
from xcube.core.store import new_data_store
from ml4xcube.datasets.xr_dataset import XrDataset
from ml4xcube.cube_utilities import get_chunk_sizes
from ml4xcube.data_assignment import assign_block_split
from ml4xcube.preprocessing import get_statistics, standardize
from ml4xcube.datasets.pytorch_xr import prepare_dataloader
from ml4xcube.training.pytorch_distributed import ddp_init, dist_train, Trainer


# To utilize ml4xcube for distributed training, use the following command with torchrun to initiate the process:
#
# torchrun --standalone --nproc_per_node=<number_of_processes> distributed_training.py <epochs>
#
# Replace `<number_of_processes>` with the number of processes you wish to run per node,
# and `<epochs>` with the total number of training epochs.


def preprocess_data(ds: xr.Dataset):
    ds = XrDataset(ds, 3).get_dataset()

    at_stat = get_statistics(ds, 'air_temperature_2m')
    lst_stat = get_statistics(ds, 'land_surface_temperature')

    X = standardize(ds['air_temperature_2m'], *at_stat)
    y = standardize(ds['land_surface_temperature'], *lst_stat)

    # Split the data based on the 'split' attribute
    X_train, X_test = X[ds['split'] == True], X[ds['split'] == False]
    y_train, y_test = y[ds['split'] == True], y[ds['split'] == False]

    X_train = X_train.reshape(-1, 1)  # Making it [num_samples, 1]
    y_train = y_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return X_train, X_test, y_train, y_test


def load_train_objs():
    """Load and preprocess dataset, returning prepared training and testing sets, model, and optimizer."""

    # Load dataset from storage
    data_store = new_data_store("s3", root="esdl-esdc-v2.1.1", storage_options=dict(anon=True))
    dataset = data_store.open_data('esdc-8d-0.083deg-184x270x270-2.1.1.zarr')
    ds = dataset[['land_surface_temperature', 'air_temperature_2m']]

    # Create a land mask
    lon_grid, lat_grid = np.meshgrid(ds.lon, ds.lat)
    lm0 = da.from_array(globe.is_land(lat_grid, lon_grid))
    lm = da.stack([lm0 for _ in range(ds.dims['time'])], axis=0)

    # Assign land mask to the dataset and split data into blocks
    ds = ds.assign(land_mask=(['time', 'lat', 'lon'], lm.rechunk(chunks=([v for _, v in get_chunk_sizes(ds)]))))
    ds = assign_block_split(ds, block_size=[("time", 10), ("lat", 100), ("lon", 100)], split=0.8)

    # Preprocess data and split into training and testing sets
    X_train, X_test, y_train, y_test = preprocess_data(ds)

    # Create TensorDataset objects for both training and testing sets
    train_set = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_set = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    # Initialize model and optimizer
    model = nn.Linear(in_features=1, out_features=1, bias=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    return train_set, test_set, model, optimizer


def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt", best_model_path: str = 'Best_Model.pt'):
    """Main function to run distributed training process."""

    # Initialize distributed data parallel training
    ddp_init()

    # Load training objects and prepare data loaders
    train_set, test_set, model, optimizer = load_train_objs()
    train_loader = prepare_dataloader(train_set, batch_size, num_workers=5, parallel=True)
    test_loader = prepare_dataloader(test_set, batch_size, num_workers=5, parallel=True)

    # Initialize the trainer and start training
    trainer = Trainer(model, train_loader, test_loader, optimizer, save_every, best_model_path, snapshot_path, task_type='supervised')
    dist_train(trainer, total_epochs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=10, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=7, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    main(args.save_every, args.total_epochs, args.batch_size)