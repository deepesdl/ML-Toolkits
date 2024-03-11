import torch
from torch.utils.data import Dataset
import numpy as np
from mltools.cube_utilities import iter_data_var_blocks, get_chunk_by_index, get_chunk_sizes
from mltools.data_processing import standardize


class ChunkDataset(Dataset):

    def __init__(self, xds, block_sizes=None, at_stat=None, lst_stat=None):
        self.xds = xds
        self.block_sizes = block_sizes if block_sizes is not None else get_chunk_sizes(xds)
        self.at_stat = at_stat
        self.lst_stat = lst_stat

        # Calculate total chunks without pre-loading them
        self.total_chunks = np.prod([len(range(0, xds.dims[dim_name], size)) for dim_name, size in self.block_sizes])
        # self.total_chunks = len(list(iter_data_var_blocks(self.xds, self.block_sizes)))

    def _preprocess_chunk(self, chunk):
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
        return self.total_chunks

    def __getitem__(self, idx):
        chunk = get_chunk_by_index(self.xds, idx, self.block_sizes)
        X, y = self._preprocess_chunk(chunk)
        if X is not None and len(X) > 0:
            ds = torch.utils.data.TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())
            return ds
        else:
            # None
            return torch.utils.data.TensorDataset(torch.tensor([]), torch.tensor([]))
