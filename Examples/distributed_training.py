import math
import numpy as np
import xarray as xr
from xcube.core.store import new_data_store
from global_land_mask import globe
from mltools.cube_utilities import get_chunk_sizes
from mltools.data_assignment import assign_split

import pandas as pd
import dask.array as da
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch.nn.functional import normalize

import nbimporter

data_store = new_data_store("s3", root="esdl-esdc-v2.1.1", storage_options=dict(anon=True))
dataset = data_store.open_data('esdc-8d-0.083deg-184x270x270-2.1.1.zarr')
ds = dataset[['land_surface_temperature', 'air_temperature_2m']]
print(ds)

lon_grid, lat_grid = np.meshgrid(ds.lon,ds.lat)
lm0 = da.from_array(globe.is_land(lat_grid, lon_grid))

lm = da.stack([lm0 for i in range(ds.dims['time'])], axis = 0)



xdsm = ds.assign(land_mask= (['time','lat','lon'],lm.rechunk(chunks=([v for k,v in get_chunk_sizes(ds)]))))

df = ds.sel({'time' : '2002-05-21'}).to_dataframe()
print(df)

# block sampling
xds = assign_split(xdsm, block_size=[("time", 10), ("lat", 100), ("lon", 100)], split=0.8)



lr = 0.1
epochs = 3

reg_model = nn.Linear(in_features=1, out_features=1, bias=True)
mse_loss = nn.MSELoss()
optimizer = torch.optim.SGD(reg_model.parameters(), lr=lr)
