import os
import numpy as np
import xarray as xr
from typing import List

"""
The python file gets global predictor data e.g. land cover classification data for the corresponding dimensions as the 
examined data cube (e.g. latitude and longitude coordinates). 
The results will be stored in a '.zarr' dataset.
During the area slicing process the zarr dataset is the source for the data in order to be used as predictors.
Once the global zarr dataset is executed, the dataset can be used for all gapfilling example applications.
This file can be helpful to extract other variables as predictors and match the coordinates.

"""


class HelpingPredictor:
    def __init__(self, ds: xr.Dataset, variable: str, ds_predictor: xr.DataArray, predictor_path: str,
                 predictor: str = 'lccs_class', layer_dim: str = None):
        """
        Class to get the predictor data for a specific variable.

        Attributes:
            ds_variable (xarray.DataArray): The dataset with the target variable you want to estimate.
            ds_predictor (xarray.DataArray): The dataset with the predictor variable that will help the estimation.
            predictor (str): The name of the predictor variable.
            predictor_path (str): The path to save the processed predictor data.
            layer_dim (str): The dimension along which to iterate (e.g., 'time').
        """
        self.layer_dim = layer_dim
        self.ds_variable = ds[variable]
        self.initialize_dimensions(list(self.ds_variable.dims))
        self.ds_variable = self.ds_variable.isel({self.layer_dim: 0})
        self.ds_predictor = ds_predictor.isel({self.layer_dim: 0})
        self.dim1 = self.ds_variable.dims[0]
        self.dim2 = self.ds_variable.dims[1]
        self.predictor = predictor
        self.predictor_path = predictor_path

    def initialize_dimensions(self, dims: List[str]) -> None:
        """
        Initializes the dimensions for the predictor and target variables.

        Args:
            dims (List[str]): List of dimensions in the dataset.

        Returns:
            None
        """
        dim1, dim2, dim3 = dims[0], dims[1], dims[2]
        if self.layer_dim is None:
            self.layer_dim = dim1
        layer_coords = [s for s in dims if s != self.layer_dim]
        self.dim1 = layer_coords[0]
        self.dim2 = layer_coords[1]


    def get_predictor_data(self) -> str:
        """
        Gets the predictor data for the specified variable and saves it to a zarr dataset.

        Returns:
            str: The file path of the saved zarr dataset.
        """
        # Get the coordinates from both dimensions (e.g. lat, lon) from the dataset with variable that you will estimate.
        dim1_coord_variable = self.ds_variable[self.dim1].values
        dim2_coord_variable = self.ds_variable[self.dim2].values

        # Get predictor data for the corresponding coordinates
        predictor_array = self.extract_data(dim1_coord_variable, dim2_coord_variable)

        if self.predictor == 'lccs_class':
            # Function to process LCCS data and remap land cover class to lower the granularity
            predictor_array = self.process_lccs(predictor_array)

        # Save the processed predictor data to a zarr dataset
        filepath = os.path.join(self.predictor_path, 'global_' + self.predictor + '.zarr')
        predictor_array.to_zarr(filepath, mode="w")
        return filepath

    def extract_data(self, dim1_coord_variable: np.ndarray, dim2_coord_variable: np.ndarray) -> xr.DataArray:
        """
        Extracts predictor data for specified coordinates.

        Args:
            dim1_coord_variable (np.ndarray): Coordinates of the first dimension (e.g., latitude).
            dim2_coord_variable (np.ndarray): Coordinates of the second dimension (e.g., longitude).

        Returns:
            xr.DataArray: Extracted predictor data.
        """
        # Extract the coordinates for the predictor data
        dim1_coord_predictor = self.ds_predictor[self.dim1].values
        dim2_coord_predictor = self.ds_predictor[self.dim2].values

        # Find indices for mapping coordinates
        dim1_indices = np.argmax(dim1_coord_predictor[:, None] <= dim1_coord_variable, axis=0) - 1
        dim2_indices = np.argmax(dim2_coord_predictor[:, None] >= dim2_coord_variable, axis=0) - 1

        dim1_indices = np.clip(dim1_indices, 0, len(dim1_coord_predictor) - 1)
        dim2_indices = np.clip(dim2_indices, 0, len(dim2_coord_predictor) - 1)

        # Extract predictor values based on indices
        predictor_array = self.ds_predictor[dim1_indices, dim2_indices]
        return predictor_array

    def process_lccs(self, lcc_array: xr.DataArray) -> xr.DataArray:
        """
        Processes and remaps LCCS data.

        Args:
            lcc_array (xr.DataArray): The LCCS data array to be processed.

        Returns:
            xr.DataArray: Processed LCCS data with remapped values.
        """
        # The granularity of the Land Cover Classes from the Earth System Data Cube is larger than necessary, e.g.
        # different types of mixed forests. Therefore, multiple types of a main LCC are grouped together as one.
        value_mapping = {
            11: 10, 12: 10, 61: 60, 62: 60, 71: 70, 72: 70, 81: 80, 82: 80, 121: 120, 122: 120,
            151: 150, 152: 150, 153: 150, 201: 200, 202: 200
        }

        # Remap LCCS values based on the mapping dictionary
        for old_value, new_value in value_mapping.items():
            lcc_array = lcc_array.where(lcc_array != old_value, new_value)

        return lcc_array
