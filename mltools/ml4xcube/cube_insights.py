import sys
import numpy as np
import xarray as xr
from tqdm import tqdm

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
def get_insights(cube: xr.DataArray) -> None:
    """
    Extracts and prints various characteristics of the data cube.

    Args:
        cube (xr.DataArray): The input data cube.

    Returns:
        None
    """
    # Extracting the variable's long name
    variable = cube.long_name

    # Extracting the dimension names
    dim1 = cube.dims[0]
    dim2 = cube.dims[1]
    dim3 = cube.dims[2]

    # Extracting the dimension long names
    dim1_long_name = cube[dim1].long_name
    dim2_long_name = cube[dim2].long_name
    dim3_long_name = cube[dim3].long_name

    # Extracting the range of dim1 (e.g. time), dim2 and dim3 and rounding them to 3 decimal places (e.g. lat and lon)
    try:
        dim1_min = str(np.datetime_as_string(cube[dim1][0], unit='D'))
        dim1_max = str(np.datetime_as_string(cube[dim1][-1], unit='D'))
    except:
        dim1_min = str(cube[dim1][0])
        dim1_max = str(cube[dim1][-1])
    try:
        dim2_min = round(cube[dim2].values.min(), 3)
        dim2_max = round(cube[dim2].values.max(), 3)
    except:
        dim2_min = str(cube[dim2][0])
        dim2_max = str(cube[dim2][-1])
    try:
        dim3_min = round(cube[dim3].values.min(), 3)
        dim3_max = round(cube[dim3].values.max(), 3)
    except:
        dim3_min = str(cube[dim3][0])
        dim3_max = str(cube[dim3][-1])


    # Calculating the size of the data cube and each layer
    cube_size = cube.size
    layer_size = cube[0].size

    # Calculating the number and percentage of NaN values (gaps) in the entire cube
    gap_values = np.sum(np.isnan(cube)).values.item()
    gap_values_percentage = round(gap_values / cube_size, 2)

    # Calculating the minimum and maximum values in the data cube, ignoring NaNs
    min_value = np.nanmin(cube)
    max_value = np.nanmax(cube)

    # Initializing dictionaries to track the maximum and minimum gap sizes and their dates
    max_gap = {"gap_size": 0, "date": None}
    min_gap = {"gap_size": 1, "date": None}

    # Looping through each (time) layer in the cube
    with tqdm(total=len(cube), file=sys.stdout, colour='GREEN', bar_format='{l_bar}{bar:20}{r_bar}') as pbar:
        for c in cube:
            # Calculating the absolute and relative gap size for the current layer
            c_gap_absolute = np.sum(np.isnan(c)).values.item()
            c_gap_size = round(c_gap_absolute / layer_size, 2)

            # Updating max_gap and min_gap if the current layer's gap size is greater or smaller respectively
            if c_gap_size > max_gap["gap_size"]:
                try:
                    max_gap = {"gap_size": c_gap_size, "date": str(np.datetime_as_string(c.time.values, unit='D'))}
                except:
                    max_gap = {"gap_size": c_gap_size, "date": str(c.time.values)}
            if c_gap_size < min_gap["gap_size"]:
                try:
                    min_gap = {"gap_size": c_gap_size, "date": str(np.datetime_as_string(c.time.values, unit='D'))}
                except:
                    min_gap = {"gap_size": c_gap_size, "date": str(c.time.values)}

            # Update the progress bar
            pbar.update(1)

    # Printing the insights
    print("The data cube has the following characteristics:")
    print(" ")
    print(f"{'Variable:':<25} {variable}")
    print(f"{'Shape:':<25} ({dim1}: {cube.shape[0]}, {dim2}: {cube.shape[1]}, {dim3}: {cube.shape[2]})")
    print(f"{dim1_long_name + ' range:':<25} {dim1_min} - {dim1_max}")
    print(f"{dim2_long_name + ' range:':<25} {dim2_min} - {dim2_max}")
    print(f"{dim3_long_name + ' range:':<25} {dim3_min} - {dim3_max}")
    print(f"{'Total size:':<25} {cube_size:,}")
    print(f"{'Size of each layer:':<25} {layer_size:,}")
    print(f"{'Total gap size:':<25} {gap_values} -> {int(gap_values_percentage * 100)} %")
    print(f"{'Maximum gap size:':<25} {int(max_gap['gap_size'] * 100)} % on {max_gap['date']}")
    print(f"{'Minimum gap size:':<25} {int(min_gap['gap_size'] * 100)} % on {min_gap['date']}")
    print(f"{'Value range:':<25} {min_value:.2f} - {max_value:.2f}")


# Function to generate a heat map of gap (NaN) counts for each lat/lon pixel (or other dimensions)
def get_gap_heat_map(cube: xr.DataArray) -> xr.DataArray:
    """
    Generates a heat map of value counts (non-NaN values) for each latitude/longitude pixel (or other dimensions).

    Args:
        cube (xr.DataArray): The input data cube.

    Returns:
        xr.DataArray: Heat map of non-NaN value counts for each lat/lon pixel (or other dimensions).
    """
    # Count the number of non-NaN values for each lat/lon pixel (or other dimensions)
    dim1 = cube.dims[0]
    nan_counts = cube.shape[0] - np.isnan(cube).sum(dim=dim1)
    return nan_counts
