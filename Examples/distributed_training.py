from typing import Optional, Tuple
from xcube.core.store import new_data_store
from global_land_mask import globe
from torch.utils.data import random_split
import dask.array as da
from torch import nn
from mltools.distributed_training import ddp_init, prepare_dataloader, dist_train, Trainer
import torch
from torch.utils.data import Dataset
import numpy as np
from mltools.cube_utilities import iter_data_var_blocks, get_chunk_by_index, get_chunk_sizes
from mltools.sampling import assign_block_split
from mltools.data_processing import standardize, get_statistics


class ChunkDataset(Dataset):
    """
        A dataset class that handles data chunks for distributed training.
        It takes a multi-dimensional dataset, processes it in chunks, and filters
        out relevant features for training.
        """
    def __init__(self, xds, block_sizes=None, at_stat=None, lst_stat=None):
        # Initialize dataset with given parameters
        self.xds = xds  # The multi-dimensional input dataset
        self.block_sizes = block_sizes if block_sizes is not None else get_chunk_sizes(xds)
        self.at_stat = at_stat  # Statistics for air temperature normalization
        self.lst_stat = lst_stat  # Statistics for land surface temperature normalization

        # Calculate total chunks without pre-loading them
        self.total_chunks = np.prod([len(range(0, xds.dims[dim_name], size)) for dim_name, size in self.block_sizes])
        # self.total_chunks = len(list(iter_data_var_blocks(self.xds, self.block_sizes)))

    def _preprocess_chunk(self, chunk: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # Preprocess each chunk of data: filter land areas and remove NaNs
        # Returns standardized feature and target arrays for training
        cf = {x: chunk[x].ravel() for x in chunk.keys()}
        lm = cf['land_mask']
        cft = {x: cf[x][lm == True] for x in cf.keys()}
        lst = cft['land_surface_temperature']
        cfn = {x: cft[x][~np.isnan(lst)] for x in cf.keys()}

        if len(cfn['land_surface_temperature']) > 0:
            X = standardize(cfn['air_temperature_2m'], *self.at_stat)
            y = standardize(cfn['land_surface_temperature'], *self.lst_stat)

            return X, y
        else:
            return None, None

    def __len__(self):
        # Returns the total number of chunks in the dataset
        return self.total_chunks

    def __getitem__(self, idx):
        # Retrieves a specific chunk by index, preprocesses it, and returns it as input and target tensors
        chunk = get_chunk_by_index(self.xds, idx, self.block_sizes)
        X, y = self._preprocess_chunk(chunk)
        if X is not None and len(X) > 0:
            ds = torch.utils.data.TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())
            return ds
        else:
            # None
            return torch.utils.data.TensorDataset(torch.tensor([]), torch.tensor([]))


def flatten_batch(batch):
    # A helper function to flatten nested batches of data for training
    # Converts a list of samples into a tensor with appropriate dimensions
    # Assuming each sample in the batch is a tensor of shape (mini_batch, channels, height, width)
    batch_tensor = torch.stack(batch, dim=0)

    # Flatten the batch and mini_batch dimensions
    batch_size, mini_batch, channels, height, width = batch_tensor.size()
    flattened_batch = batch_tensor.view(batch_size * mini_batch, channels, height, width)

    return flattened_batch


def load_train_objs():
    # Loads and preprocesses the training objects: the dataset, model, and optimizer
    # This includes loading the dataset, calculating statistics, creating a mask for land areas,
    # and splitting the dataset into training and test sets.
    data_store = new_data_store("s3", root="esdl-esdc-v2.1.1", storage_options=dict(anon=True))
    dataset = data_store.open_data('esdc-8d-0.083deg-184x270x270-2.1.1.zarr')
    ds = dataset[['land_surface_temperature', 'air_temperature_2m']]

    lon_grid, lat_grid = np.meshgrid(ds.lon,ds.lat)
    lm0 = da.from_array(globe.is_land(lat_grid, lon_grid))

    lm = da.stack([lm0 for i in range(ds.dims['time'])], axis = 0)
    at_stat = get_statistics(ds, 'air_temperature_2m')
    lst_stat = get_statistics(ds, 'land_surface_temperature')


    xdsm = ds.assign(land_mask= (['time','lat','lon'],lm.rechunk(chunks=([v for k,v in get_chunk_sizes(ds)]))))

    # block sampling
    xds = assign_block_split(xdsm, block_size=[("time", 10), ("lat", 100), ("lon", 100)], split=0.8)
    full_dataset = ChunkDataset(xds, at_stat=at_stat, lst_stat=lst_stat)
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # Split the dataset into training and test sets
    train_set, test_set = random_split(full_dataset, [train_size, test_size])

    model = nn.Linear(in_features=1, out_features=1, bias=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return train_set, test_set, model, optimizer


def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
    """
    The main function to run the distributed training process.
    It initializes distributed training, prepares data loaders, and starts the training loop.
    """
    ddp_init()
    train_set, test_set, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(train_set, batch_size, callback_fn=flatten_batch, num_workers=5)
    test_data = prepare_dataloader(test_set, batch_size, callback_fn=flatten_batch, num_workers=5)
    trainer = Trainer(model, train_data, test_data, optimizer, save_every, snapshot_path, task_type='supervised')
    dist_train(trainer, total_epochs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=7, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    main(args.save_every, args.total_epochs, args.batch_size)