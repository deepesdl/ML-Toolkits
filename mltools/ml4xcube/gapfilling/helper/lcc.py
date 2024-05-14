from xcube.core.store import new_data_store
import numpy as np

"""
The python file gets global land cover classification data for the corresponding latitude and longitude 
coordinates globally.
The results will be stored in the 'global_lcc.nc' file.
During the area slicing process the nc-file is the source for the LCC data in order to be used as predictors.
Once the global nc-file is executed, the nc-file can be used for all gapfilling example applications.
This file can be helpful to extract other variables as predictors and match the coordinates.
"""


def get_global_coord():
    """
    Get latitude and longitude coordinates from land surface temperature dataset.

    This function initializes the xcube datastore for s3 object storage, opens the dataset,
    and extracts latitude and longitude coordinates for the land surface temperature data.

    Returns:
    np.ndarray, np.ndarray: Arrays containing latitude and longitude coordinates.
    """
    # Initializing the xcube datastore for s3 object storage and open the dataset
    data_store = new_data_store("s3", root="esdl-esdc-v2.1.1", storage_options=dict(anon=True))
    dataset = data_store.open_data('esdc-8d-0.083deg-184x270x270-2.1.1.zarr')
    # Extract latitude and longitude coordinates for the land surface temperature data
    ds = dataset['land_surface_temperature'][0]
    lat_coord = ds.lat.values
    lon_coord = ds.lon.values
    return lat_coord, lon_coord


def get_lccs(lat_coord_temp, lon_coord_temp):
    """
    Get Land Cover Classification System (LCCS) data for specified coordinates.

    This function initializes the xcube datastore for s3 object storage, opens the dataset,
    and extracts LCCS data for the specified latitude and longitude coordinates.

    Parameters:
    - lat_coord_temp (np.ndarray): Latitude coordinates for temperature data.
    - lon_coord_temp (np.ndarray): Longitude coordinates for temperature data.

    Returns:
    xr.DataArray: LCCS data for the specified coordinates.
    """
    # Initializing the xcube datastore for s3 object storage and open the dataset
    store = new_data_store("s3", root="deep-esdl-public", storage_options=dict(anon=True))
    dataset = store.open_data('LC-1x2160x2160-1.0.0.levels')
    ds = dataset.get_dataset(0)
    lcc_ds = ds['lccs_class'][0]

    # Extract latitude and longitude coordinates for the LCCS data
    lat_coord_lcc = lcc_ds.lat.values
    lon_coord_lcc = lcc_ds.lon.values

    # Find indices for mapping coordinates
    lon_indices = np.argmax(lon_coord_lcc[:, None] >= lon_coord_temp, axis=0) - 1
    lat_indices = np.argmax(lat_coord_lcc[:, None] <= lat_coord_temp, axis=0) - 1

    lon_indices = np.clip(lon_indices, 0, len(lon_coord_lcc) - 1)
    lat_indices = np.clip(lat_indices, 0, len(lat_coord_lcc) - 1)

    # Extract LCCS values based on indices
    lcc_array = lcc_ds[lat_indices, lon_indices]
    return lcc_array


def process_lccs(lcc_array):
    """
    Process and remap LCCS data and save it to a NetCDF file.

    This function remaps LCCS values based on a mapping dictionary, and then saves the processed
    LCCS data to a NetCDF file named 'global_lcc.nc'.

    Parameters:
    - lcc_array (xr.DataArray): LCCS data to be processed.

    Returns:
    None
    """
    # The granularity of the Land Cover Classes from the Earth System Data Cube is larger than necessary, e.g. different
    # types of mixed forests. Therefore, multiple types of a main land cover class are grouped together as one.
    value_mapping = {
        11: 10, 12: 10, 61: 60, 62: 60, 71: 70, 72: 70, 81: 80, 82: 80, 121: 120, 122: 120,
        151: 150, 152: 150, 153: 150, 201: 200, 202: 200
    }

    # Remap LCCS values based on the mapping dictionary
    for old_value, new_value in value_mapping.items():
        lcc_array = lcc_array.where(lcc_array != old_value, new_value)

    # Save the processed LCCS data to a NetCDF file
    lcc_array.to_netcdf('helper/global_lcc.nc')


def main():
    # Function to get latitude and longitude coordinates from the land surface temperature variable
    lat_coord, lon_coord = get_global_coord()
    # Function to get Land Cover Classification System (LCCS) data for the corresponding lat and long coordinates
    lcc_array = get_lccs(lat_coord, lon_coord)
    # Function to process LCCS data and remap land cover class to lower the granularity
    process_lccs(lcc_array)


if __name__ == "__main__":
    main()
