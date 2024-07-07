import sys
import numpy as np
import xarray as xr
from tqdm import tqdm
from ml4xcube.cube_utilities import get_dim_range

"""
Function get_insights(cube):
    Extracts and prints various characteristics of the data cube.
    Computes and prints dimension range (e.g. time, lat, lon), total size, layer size, gap size, and value range.

Function get_gap_heat_map(cube):
    Generates a heat map of value counts (non-NaN values) for dimension 1 and 2 (e.g. latitude/longitude pixel).

Example:
    Replace 'ds' with your actual dataset variable when calling the functions.
    
    import datetime
    from xcube.core.store import new_data_store
    import os
    
    variable = 'land_surface_temperature'
    data_store = new_data_store("s3", root="esdl-esdc-v2.1.1", storage_options=dict(anon=True))
    dataset = data_store.open_data('esdc-8d-0.083deg-184x270x270-2.1.1.zarr')
    ds = dataset[variable]
    ds = ds.sel(time=slice(datetime.date(2008, 1, 1), datetime.date(2008, 12, 31)))
    path = os.path.dirname(os.getcwd()) + "cube.zarr"
    
    ds.to_zarr(path)
    ds = xr.open_zarr(path)[variable]
    
    get_insights(ds)
    get_gap_heat_map(ds)
"""


# Function to get insights from the data cube
def get_insights(cube: xr.DataArray, variable: str, layer_dim: str = None) -> None:
    """
    Extracts and prints various characteristics of the data cube.

    Args:
        cube (xr.DataArray): The input data cube.
        variable (str): The variable name to extract from the data cube.
        layer_dim (str, optional): The dimension along which to iterate (e.g., 'time'). Defaults to None.

    Returns:
        None
    """
    # Extracting the dimension names
    dims = list(cube.dims)
    dim1, dim2, dim3 = dims[0], dims[1], dims[2]
    if layer_dim is None: layer_dim = dim1

    layer_coords = [s for s in dims if s != layer_dim]
    layer_coord1, layer_coord2 = layer_coords

    #Extracting the shape
    sizes = cube.sizes
    size_values = list(sizes.values())

    # Extracting the dimension long names
    dim1_long_name = cube[dim1].long_name
    dim2_long_name = cube[dim2].long_name
    dim3_long_name = cube[dim3].long_name


    dim1_min, dim1_max = get_dim_range(cube, dim1)
    dim2_min, dim2_max = get_dim_range(cube, dim2)
    dim3_min, dim3_max = get_dim_range(cube, dim3)


    # Calculating the size of the data cube and each layer
    cube_size = size_values[0]*size_values[1]*size_values[2]
    layer_size = sizes[layer_coord1] * sizes[layer_coord2]

    cube = cube[variable]

    # Calculating the number and percentage of NaN values (gaps) in the entire cube
    gap_values = np.sum(np.isnan(cube)).values.item()
    gap_values_percentage = round(gap_values / cube_size, 2)

    # Calculating the minimum and maximum values in the data cube, ignoring NaNs
    min_value = np.nanmin(cube)
    max_value = np.nanmax(cube)

    # Initializing dictionaries to track the maximum and minimum gap sizes and their dates
    max_gap = {"gap_size": 0, "dim_val": None}
    min_gap = {"gap_size": 1, "dim_val": None}

    # Looping through each (time) layer in the cube
    with tqdm(total=len(cube), file=sys.stdout, colour='GREEN', bar_format='{l_bar}{bar:20}{r_bar}') as pbar:
        for i in range(cube.sizes[layer_dim]):
            c = cube.isel({layer_dim: i})
            # Calculating the absolute and relative gap size for the current layer
            c_gap_absolute = np.sum(np.isnan(c)).values.item()
            c_gap_size = round(c_gap_absolute / layer_size, 2)

            # Updating max_gap and min_gap if the current layer's gap size is greater or smaller respectively
            if c_gap_size > max_gap["gap_size"]:
                try:
                    max_gap = {"gap_size": c_gap_size, "dim_val": str(np.datetime_as_string(c.time.values, unit='D'))}
                except:
                    max_gap = {"gap_size": c_gap_size, "dim_val": str(c.time.values)}
            if c_gap_size < min_gap["gap_size"]:
                try:
                    min_gap = {"gap_size": c_gap_size, "dim_val": str(np.datetime_as_string(c.time.values, unit='D'))}
                except:
                    min_gap = {"gap_size": c_gap_size, "dim_val": str(c.time.values)}

            # Update the progress bar
            pbar.update(1)

    # Printing the insights
    print("The data cube has the following characteristics:")
    print(" ")
    print(f"{'Variable:':<25} {variable}")
    print(f"{'Shape:':<25} ({dim1}: {size_values[0]}, {dim2}: {size_values[1]}, {dim3}: {size_values[2]})")
    print(f"{dim1_long_name + ' range:':<25} {round(dim1_min, 3)} - {round(dim1_max, 3)}")
    print(f"{dim2_long_name + ' range:':<25} {round(dim2_min, 3)} - {round(dim2_max, 3)}")
    print(f"{dim3_long_name + ' range:':<25} {round(dim3_min, 3)} - {round(dim3_max, 3)}")
    print(f"{'Total size:':<25} {cube_size:,}")
    print(f"{'Size of each layer:':<25} {layer_size:,}")
    print(f"{'Total gap size:':<25} {gap_values} -> {int(gap_values_percentage * 100)} %")
    print(f"{'Maximum gap size:':<25} {int(max_gap['gap_size'] * 100)} % on {max_gap['dim_val']}")
    print(f"{'Minimum gap size:':<25} {int(min_gap['gap_size'] * 100)} % on {min_gap['dim_val']}")
    print(f"{'Value range:':<25} {min_value:.2f} - {max_value:.2f}")


# Function to generate a heat map of gap (NaN) counts for each lat/lon pixel (or other dimensions)
def get_gap_heat_map(cube: xr.DataArray, count_dim: str) -> xr.DataArray:
    """
    Generates a heat map of value counts (non-NaN values) for each latitude/longitude pixel.

    Args:
        cube (xr.DataArray): The input data cube.
        count_dim (str): The name of the dimension along which to count NaN values.

    Returns:
        xr.DataArray: Heat map of non-NaN value counts for each lat/lon pixel.
    """
    # Count the number of non-NaN values for each lat/lon pixel
    nan_counts = cube.shape[0] - np.isnan(cube).sum(dim=count_dim)
    return nan_counts
