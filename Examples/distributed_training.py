import zarr
import torch
import argparse
import numpy as np
import xarray as xr
import dask.array as da
from ml4xcube.cube_utilities import assign_dims
from ml4xcube.datasets.pytorch import LargeScaleXrDataset, prepare_dataloader
from ml4xcube.training.pytorch_distributed import ddp_init, dist_train, Trainer


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
        air_temperature_2m_list       = []
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
    train_store = zarr.open('train_cube.zarr')
    test_store = zarr.open('test_cube.zarr')

    # Convert Zarr stores to Dask arrays and then to xarray Datasets
    train_data = {var: da.from_zarr(train_store[var]) for var in train_store.array_keys()}
    test_data = {var: da.from_zarr(test_store[var]) for var in test_store.array_keys()}

    # Assign dimensions using the assign_dims function
    train_data = assign_dims(train_data, ('samples', ))
    test_data = assign_dims(test_data, ('samples', ))

    train_set = xr.Dataset(train_data)
    test_set = xr.Dataset(test_data)

    # Create PyTorch data sets
    train_ds = LargeScaleXrDataset(train_set)
    test_ds  = LargeScaleXrDataset(test_set)

    # Initialize model and optimizer
    model     = torch.nn.Linear(in_features=1, out_features=1, bias=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss      = torch.nn.MSELoss(reduction='mean')

    return train_ds, test_ds, model, optimizer, loss

def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt", best_model_path: str = 'Best_Model.pt'):
    """Main function to run distributed training process."""

    # Initialize distributed data parallel training
    ddp_init()

    # Load training objects
    train_set, test_set, model, optimizer, loss = load_train_objs()

    # Prepare data loaders
    train_loader = prepare_dataloader(train_set, batch_size, num_workers=5, parallel=True, callback_fn=map_function)
    test_loader  = prepare_dataloader(test_set, batch_size, num_workers=5, parallel=True, callback_fn=map_function)

    # Initialize the trainer and start training
    trainer = Trainer(
        model                = model,
        train_data           = train_loader,
        test_data            = test_loader,
        optimizer            = optimizer,
        save_every           = save_every,
        best_model_path      = best_model_path,
        early_stopping       = True,
        snapshot_path        = snapshot_path,
        patience             = 3,
        loss                 = loss,
        validate_parallelism = True
    )

    dist_train(trainer, total_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=10, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=7, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    main(args.save_every, args.total_epochs, args.batch_size)
