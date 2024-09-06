import torch
import numpy as np
import xarray as xr
from ml4xcube.datasets.pytorch import PTXrDataset, prep_dataloader
from ml4xcube.training.pytorch_distributed import ddp_init, Trainer

"""
Before performing distributed machine learning, run the distributed_datasets_creation.py in order to create the training and the test set.

To utilize ml4xcube for distributed training, use the following command with torchrun to initiate the process:

torchrun --standalone --nproc_per_node=<number_of_processes> distributed_training.py

Replace <number_of_processes> with the number of processes you wish to run
"""


def map_function(batch):
    """
    Convert list of samples (batch) into tensors X and y.
    X corresponds to 'air_temperature_2m' and y corresponds to 'land_surface_temperature'.
    """

    #X = np.stack([d['air_temperature_2m'] for d in batch], axis=0).reshape(-1, 1)
    #y = np.stack([d['land_surface_temperature'] for d in batch], axis=0).reshape(-1, 1)

    # Extract the arrays from the list of dictionaries
    air_temperature_2m_list       = []
    land_surface_temperature_list = []

    for d in batch:
        air_temperature_2m_list.append(d['air_temperature_2m'])
        land_surface_temperature_list.append(d['land_surface_temperature'])

    # Stack the arrays along the first dimension (assuming they are 1D arrays)
    X = np.stack(air_temperature_2m_list, axis=0).reshape(-1, 1)
    y = np.stack(land_surface_temperature_list, axis=0).reshape(-1, 1)

    return torch.tensor(X), torch.tensor(y)


def load_train_objs():
    train_set = xr.open_zarr('train_cube.zarr')
    test_set = xr.open_zarr('test_cube.zarr')

    # Create PyTorch data sets
    train_ds = PTXrDataset(train_set)
    test_ds  = PTXrDataset(test_set)

    # Initialize model and optimizer
    model     = torch.nn.Linear(in_features=1, out_features=1, bias=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss      = torch.nn.MSELoss(reduction='mean')

    return train_ds, test_ds, model, optimizer, loss


def main():
    """Main function to run distributed training process."""

    # Initialize distributed data parallel training
    ddp_init()

    # Load training objects
    train_set, test_set, model, optimizer, loss = load_train_objs()

    # Prepare data loaders
    train_loader, test_loader = prep_dataloader(train_set, test_set, batch_size=32, num_workers=5, parallel=True, callback=map_function)

    # Initialize the trainer and start training
    trainer = Trainer(
        model                = model,
        train_data           = train_loader,
        test_data            = test_loader,
        optimizer            = optimizer,
        model_path           = 'Best_Model.pt',
        epochs               = 1,
        early_stopping       = True,
        patience             = 3,
        loss                 = loss,
        validate_parallelism = True
    )

    trainer.train()


if __name__ == "__main__":

    main()
