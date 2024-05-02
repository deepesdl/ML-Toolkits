import numpy as np


def apply_filter(ds, filter_var):
    if filter_var and filter_var in ds:
        filter_mask = ds[filter_var]
        filtered_ds = {x: ds[x][filter_mask == True] for x in ds.keys()}
    else:
        filtered_ds = ds
    return filtered_ds


def drop_nan_values(ds, vars):
    for var in vars:
        ds = {x: ds[x][~np.isnan(ds[var])] for x in vars}
    return ds
