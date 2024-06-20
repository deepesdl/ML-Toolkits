from xcube.core.store import new_data_store
import numpy as np

"""
The python file gets global predictor data e.g. land cover classification data for the corresponding 
latitude and longitude coordinates globally.
The results will be stored in a '.zarr' dataset.
During the area slicing process the zarr dataset is the source for the data in order to be used as predictors.
Once the global zarr dataset is executed, the dataset can be used for all gapfilling example applications.
This file can be helpful to extract other variables as predictors and match the coordinates.
"""


class HelpingPredictor:
    """
    Get the predictor data for a specific variable.

    Attributes:
        ds_variable (xarray.DataArray): The dataset with the target variable you want to estimate
        ds_variable (xarray.DataArray): The dataset with the predictor variable that will help the estimation
        predictor (str): The name of the predictor
    """
    def __init__(self, ds_variable, ds_predictor, predictor='lccs_class'):
        self.ds_variable = ds_variable[0]
        self.ds_predictor = ds_predictor[0]
        self.predictor = predictor


    def get_predictor_data(self):
        # Get latitude and longitude coordinates from the dataset with variable that you will estimate.
        lat_coord_variable = self.ds_variable.lat.values
        lon_coord_variable = self.ds_variable.lon.values

        # Get predictor data for the corresponding lat and long coordinates
        predictor_array = self.extract_data(lat_coord_variable, lon_coord_variable)

        if self.predictor == 'lccs_class':
            # Function to process LCCS data and remap land cover class to lower the granularity
            predictor_array = self.process_lccs(predictor_array)

        # Save the processed predictor data to a zarr dataset
        try:
            if self.predictor == 'lccs_class':
                predictor_array.to_zarr('global_lcc.zarr')
            else:
                predictor_array.to_zarr('global_predictor.zarr')
        except:
            print("There is already a .zarr-file with this name. Please delete it if you want to create a new one.")

    def extract_data(self, lat_coord_variable, lon_coord_variable):
        """
        Get predictor data for specified coordinates.

        This function extracts the predictor data for the specified latitude and longitude coordinates.
        """
        # Extract latitude and longitude coordinates for the predictor data
        lat_coord_lcc = self.ds_predictor.lat.values
        lon_coord_lcc = self.ds_predictor.lon.values

        # Find indices for mapping coordinates
        lon_indices = np.argmax(lon_coord_lcc[:, None] >= lon_coord_variable, axis=0) - 1
        lat_indices = np.argmax(lat_coord_lcc[:, None] <= lat_coord_variable, axis=0) - 1

        lon_indices = np.clip(lon_indices, 0, len(lon_coord_lcc) - 1)
        lat_indices = np.clip(lat_indices, 0, len(lat_coord_lcc) - 1)

        # Extract predictor values based on indices
        predictor_array = self.ds_predictor[lat_indices, lon_indices]
        return predictor_array

    def process_lccs(self, lcc_array):
        """
        Process and remap LCCS data.

        This function remaps LCCS values based on a mapping dictionary and returns the processed data
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

        return lcc_array


def main():
    # Initializing the xcube datastore for s3 object storage and open the dataset of the variable you want to estimate
    data_store = new_data_store("s3", root="esdl-esdc-v2.1.1", storage_options=dict(anon=True))
    dataset = data_store.open_data('esdc-8d-0.083deg-184x270x270-2.1.1.zarr')
    ds_variable = dataset['land_surface_temperature']

    # Initializing the xcube datastore for s3 object storage and open the dataset of the predictor variable
    predictor = 'lccs_class'
    data_store = new_data_store("s3", root="deep-esdl-public", storage_options=dict(anon=True))
    dataset = data_store.open_data('LC-1x2160x2160-1.0.0.levels')
    ds_predictor = dataset.get_dataset(0)[predictor]

    HelpingPredictor(ds_variable, ds_predictor, predictor).get_predictor_data()


if __name__ == "__main__":
    main()
