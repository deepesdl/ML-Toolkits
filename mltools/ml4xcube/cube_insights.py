import numpy as np
from tqdm import tqdm
import sys

"""
Function get_insights(cube):
    Extracts and prints various characteristics of the data cube.
    Computes and prints time range, latitude/longitude range, total size, layer size, gap size, and value range.

Function get_gap_heat_map(cube):
    Generates a heat map of value counts (non-NaN values) for each latitude/longitude pixel.

Example usage:
    Replace 'ds' with your actual dataset variable when calling the functions.
    import datetime
    from xcube.core.store import new_data_store
    import xarray as xr
    
    variable = 'land_surface_temperature'
    data_store = new_data_store("s3", root="esdl-esdc-v2.1.1", storage_options=dict(anon=True))
    dataset = data_store.open_data('esdc-8d-0.083deg-184x270x270-2.1.1.zarr')
    ds = dataset[variable]
    ds = ds.sel(lat=slice(time=slice((datetime.date(2008, 1, 1), datetime.date(2008, 12, 31)))))
    path = os.path.dirname(os.getcwd()) + "cube.zarr"
    
    ds.to_zarr(path)
    ds = xr.open_zarr(path)[variable]
    
    get_insights(ds)
    get_gap_heat_map(ds)
"""

# Function to get insights from the data cube
def get_insights(cube):
    # Extracting the variable's long name
    variable = cube.long_name

    # Extracting the time range
    time_min = str(np.datetime_as_string(cube.time[0], unit='D'))
    time_max = str(np.datetime_as_string(cube.time[-1], unit='D'))

    # Extracting latitude and longitude ranges and rounding them to 3 decimal places
    lat_min = round(cube.lat.values.min(), 3)
    lat_max = round(cube.lat.values.max(), 3)
    lon_min = round(cube.lon.values.min(), 3)
    lon_max = round(cube.lon.values.max(), 3)

    # Calculating the size of the data cube and each layer
    cube_size = cube.size
    layer_size = cube[0].size

    # Calculating the number and percentage of NaN values (gaps) in the entire cube
    gap_values = np.sum(np.isnan(cube)).values.item()
    gap_values_percentage = round(gap_values / cube_size, 2)

    # Calculating the minimum and maximum values in the data cube, ignoring NaNs
    min_value = round(np.nanmin(cube), 2)
    max_value = round(np.nanmax(cube), 2)

    # Initializing dictionaries to track the maximum and minimum gap sizes and their dates
    max_gap = {"gap_size": 0, "date": None}
    min_gap = {"gap_size": 1, "date": None}

    # Looping through each time layer in the cube
    with tqdm(total=len(cube), file=sys.stdout, colour='GREEN', bar_format='{l_bar}{bar:20}{r_bar}') as pbar:
        for c in cube:
            # Calculating the absolute and relative gap size for the current layer
            c_gap_absolute = np.sum(np.isnan(c)).values.item()
            c_gap_size = round(c_gap_absolute / layer_size, 2)

            # Updating max_gap and min_gap if the current layer's gap size is greater or smaller respectively
            if c_gap_size > max_gap["gap_size"]:
                max_gap = {"gap_size": c_gap_size, "date": str(np.datetime_as_string(c.time.values, unit='D'))}
            if c_gap_size < min_gap["gap_size"]:
                min_gap = {"gap_size": c_gap_size, "date": str(np.datetime_as_string(c.time.values, unit='D'))}

            # Update the progress bar
            pbar.update(1)

    # Printing the insights
    print("The data cube has the following characteristics:")
    print(" ")
    print(f"Variable:             {variable}")
    print(f"Shape:                (time: {cube.shape[0]}, lat: {cube.shape[1]}, lon: {cube.shape[2]})")
    print(f"Time range:           {time_min} - {time_max}")
    print(f"Latitude range:       {lat_min}° - {lat_max}°")
    print(f"Longitude range:      {lon_min}° - {lon_max}°")
    print(f"Total size:           {cube_size}")
    print(f"Size of each layer:   {layer_size}")
    print(f"Total gap size:       {gap_values} -> {int(gap_values_percentage * 100)} %")
    print(f"Maximum gap size:     {int(max_gap['gap_size'] * 100)} % on {max_gap['date']}")
    print(f"Minimum gap size:     {int(min_gap['gap_size'] * 100)} % on {min_gap['date']}")
    print(f"Value range:         ", min_value, "-", max_value)


# Function to generate a heat map of gap (NaN) counts for each lat/lon pixel
def get_gap_heat_map(cube):
    # Count the number of non-NaN values for each lat/lon pixel
    nan_counts = cube.shape[0] - np.isnan(cube).sum(dim='time')
    return nan_counts
