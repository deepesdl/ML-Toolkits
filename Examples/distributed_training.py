from xcube.core.store import new_data_store
from global_land_mask import globe
import dask.array as da
import xarray as xr
import random
from torch import nn
import torch
import numpy as np
from torch.utils.data import TensorDataset

# add path, if mltools not installed
import sys
sys.path.append('../mltools')

from mltools import get_chunk_by_index, get_chunk_sizes, calculate_total_chunks
from mltools import ddp_init, prepare_dataloader, dist_train, Trainer
from mltools import assign_block_split
from mltools import standardize


def preprocess_data(ds: xr.Dataset):
    """Preprocess and split dataset into training and testing datasets.

    Args:
    ds (xr.Dataset): The input dataset to be processed.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Returns preprocessed and split training and testing data.
    """

    # Calculate the total number of data chunks in the dataset
    total_chunks = calculate_total_chunks(ds)

    # Initialize lists to hold the data from selected chunks
    X_train_all, X_test_all, y_train_all, y_test_all = [], [], [], []

    # Keep track of processed chunks to avoid repetition
    processed_chunks = list()

    # Process chunks until 3 unique chunks have been processed
    while len(processed_chunks) < 3:
        chunk_index = random.randint(0, total_chunks - 1)  # Select a random chunk index
        if chunk_index in processed_chunks:
            continue  # Skip if this chunk has already been processed

        # Retrieve the chunk by its index
        chunk = get_chunk_by_index(ds, chunk_index)

        # Flatten the data and select only land values, then drop NaN values
        cf = {x: chunk[x].ravel() for x in chunk.keys()}
        lm = cf['land_mask']
        cft = {x: cf[x][lm == True] for x in cf.keys()}
        lst = cft['land_surface_temperature']
        cfn = {x: cft[x][~np.isnan(lst)] for x in cf.keys()}

        # Proceed only if there are valid land surface temperature data points
        if len(cfn['land_surface_temperature']) > 0:
            processed_chunks.append(chunk_index)
            X = cfn['air_temperature_2m']
            y = cfn['land_surface_temperature']

            # Split the data based on the 'split' attribute
            X_train, X_test = X[cfn['split'] == True], X[cfn['split'] == False]
            y_train, y_test = y[cfn['split'] == True], y[cfn['split'] == False]

            # Append processed data to their respective lists
            X_train_all.append(X_train)
            X_test_all.append(X_test)
            y_train_all.append(y_train)
            y_test_all.append(y_test)

    # Concatenate the data from all processed chunks
    X_train = np.concatenate(X_train_all)
    X_test = np.concatenate(X_test_all)
    y_train = np.concatenate(y_train_all)
    y_test = np.concatenate(y_test_all)

    # Standardize the data
    X_mean, X_std = np.mean(X_train), np.std(X_train)
    y_mean, y_std = np.mean(y_train), np.std(y_train)
    X_train = standardize(X_train, X_mean, X_std)
    X_test = standardize(X_test, X_mean, X_std)
    y_train = standardize(y_train, y_mean, y_std)
    y_test = standardize(y_test, y_mean, y_std)

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
    train_loader = prepare_dataloader(train_set, batch_size, num_workers=5)
    test_loader = prepare_dataloader(test_set, batch_size, num_workers=5)

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
