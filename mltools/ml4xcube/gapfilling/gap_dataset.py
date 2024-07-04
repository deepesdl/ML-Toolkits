import os
import sys
import time
import shutil
import datetime
import numpy as np
import xarray as xr
from typing import Dict, List, Union, Optional

"""
In this file we prepare the data for the gapfilling algorithm. 
We slice a specific dimensions (e.g. area and time range) from the data cube with the values for the specified variable.
If requested artificial gaps can be inserted in the array. 

Example
```
    from xcube.core.store import new_data_store
    
    # Directory name
    ds_name = 'GermanyNB123'
    variable = 'land_surface_temperature'
    dimensions = {
        'lat': (54, 48),
        'lon': (6, 15),
        'time': (datetime.date(2008, 1, 1), datetime.date(2008, 12, 31))
    }
    artificial_gaps = None
    actual_matrix = datetime.date(2008, 9, 1)
    data_store = new_data_store("s3", root="esdl-esdc-v2.1.1", storage_options=dict(anon=True))
    dataset = data_store.open_data('esdc-8d-0.083deg-184x270x270-2.1.1.zarr')
    ds = dataset[variable]
    
    path = os.path.dirname(os.getcwd()) + "GermanyNB123.zarr"
    
    ds.to_zarr(path, mode="w")
    ds = xr.open_zarr(path)[variable]
    
    GapDataset(ds, ds_name, dimensions, artificial_gaps, actual_matrix).get_data()
```
"""


class GapDataset:
    def __init__(self, ds: xr.DataArray, ds_name: str = 'Test123',
                 dimensions: Optional[Dict[str, tuple]] = None,
                 artificial_gaps: Optional[List[float]] = None,
                 actual_matrix: Union[str, datetime.date] = 'Random'):
        """
        Represents a dataset for handling data gaps.

        Attributes:
            ds (xr.DataArray): The original dataset that might be sliced later on.
            ds_name (str): The name of the dataset.
            dimensions (Optional[Dict[str, tuple]]): Dict containing dimension ranges (e.g. lat, lon, times); no slicing if no dim specified.
            artificial_gaps (Optional[List[float]]): List of artificial gap sizes; None if no artificial gaps should be created.
            actual_matrix (Union[str, datetime.date]): Specifies the actual data matrix or 'Random' for random selection.
            directory (str): The directory where data will be stored.
            extra_data (xr.DataArray): Additional data used as predictors (e.g. Land Cover Classes).
            sliced_ds (xr.DataArray): The sliced dataset.
        """
        self.ds = ds
        self.ds_name = ds_name
        self.dimensions = dimensions
        self.artificial_gaps = artificial_gaps
        self.actual_matrix = actual_matrix

        self.directory = os.path.dirname(os.getcwd()) + '/application_results/' + ds_name + "/" if \
            os.getcwd().split('/')[-1] != 'gapfilling' else 'application_results/' + ds_name + "/"
        self.extra_data = None
        self.sliced_ds = None

    def get_data(self) -> None:
        """
        Retrieve and process (area-)specific data.

        This method performs the following tasks:
        - Creates a directory or cleans it if it already exists.
        - Slices the dimensions from a global dataset.
        - Retrieves additional data (e.g., land cover classes) for use as predictors.
        - Processes the data and optionally creates artificial data gaps for gap filling.

        Returns:
            None
        """
        start_time = time.time()

        # Create a directory or clean it if it already exists
        shutil.rmtree(self.directory, ignore_errors=True)
        os.makedirs(self.directory, exist_ok=True)
        # Slice the dimensions from a global dataset
        self.slice_dataset()
        # Retrieve other extra matrix to use them as predictors (e.g. land cover data)
        self.get_extra_matrix()
        # If requested create artificial data gaps which values will be estimated later on
        self.process_actual_matrix()

        print("Runtime:", round(time.time() - start_time, 2))

    def slice_dataset(self) -> None:
        """
        Slice the dataset to extract the specific dimensions (e.g.latitude, longitude, and time range)

        This method slices the dataset based on the specified dimensions (e.g. lat, lon, and time) and extracts them.

        Returns:
            None
        """
        dim1 = self.ds.dims[0]
        dim2 = self.ds.dims[1]
        dim3 = self.ds.dims[2]

        # Slice the dataset to extract the specific dimensions (e.g. latitude, longitude and time range)
        try:
            self.ds = self.ds.sel(
                **{dim1: slice(self.dimensions[dim1][0], self.dimensions[dim1][1]),
                   dim2: slice(self.dimensions[dim2][0], self.dimensions[dim2][1]),
                   dim3: slice(self.dimensions[dim3][0], self.dimensions[dim3][1])}
            )
        except:
            print("Please provide correct dimension names!")
            print(f"The cube contains the following dimension: {dim1}, {dim2}, {dim3}")
            print("In 'dimensions' you provided the following dimensions:", ", ".join(self.dimensions))
            sys.exit()

        print(self.ds_name, self.ds.sizes.mapping)

        # To avoid exceptions due to type errors, save the dataset and load it again
        self.ds.to_zarr(self.directory + "cube.zarr", mode="w")
        self.ds = xr.open_zarr(self.directory + "cube.zarr")[self.ds.name]

    def get_extra_matrix(self) -> None:
        """
        Retrieve an extra variable matrix to use it as predictor (e.g. Land Cover Classes).

        This method opens a zarr dataset containing global predictor data, selects and slices the predictor data
        based on the specified dimension range (e.g. lat and lon) and saves it as an extra data matrix
        for use in gap filling.

        Returns:
            None
        """
        # Open the predictor datasets from zarr with the global predictor data
        # Via Looping through all files in the target directory that start with "global"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(current_dir, 'helper')

        for filename in os.listdir(target_dir):
            if filename.startswith('global') and filename.endswith('.zarr'):
                try:
                    data_file = os.path.join(target_dir, filename)

                    # Open the predictor dataset
                    predictor_dataset = xr.open_zarr(data_file)
                    predictor_variable = str(list(predictor_dataset.data_vars)[0])
                    predictor_dataset = predictor_dataset[predictor_variable]

                    # Extracting the dimension names
                    dim1 = predictor_dataset.dims[0]
                    dim2 = predictor_dataset.dims[1]
                    # Select and slice the predictor data based on the specified dimension ranges (e.g. lat and lon)
                    self.extra_data = predictor_dataset.sel(**{dim1: slice(self.dimensions[dim1][0], self.dimensions[dim1][1]),
                                                               dim2: slice(self.dimensions[dim2][0], self.dimensions[dim2][1])})

                    self.extra_data.to_zarr(self.directory + "extra_matrix_" + predictor_variable + ".zarr", mode="w")
                    print(f"Extra predictor data matrix for has {predictor_variable} been created.")
                except:
                    print(f"No extra predictor data matrix for {predictor_variable} has been created. ")

    def process_actual_matrix(self) -> None:
        """
        Process the actual data matrix.

        This method processes the actual data matrix, including selecting a random or specified layer (e.g. date),
        calculating the real gap size percentage, and creating artificial data gaps if requested.

        Returns:
            None
        """
        # Select the relevant/random layer (e.g. date) with the gaps to be filled and slice the corresponding array
        dim1 = self.ds.dims[0]
        dim1_values = self.ds.coords[dim1].values
        try:
            actual_value = np.random.choice(dim1_values) if self.actual_matrix == 'Random' else np.datetime64(
                self.actual_matrix)
        except:
            actual_value = np.random.choice(dim1_values) if self.actual_matrix == 'Random' else self.actual_matrix

        actual_matrix = self.ds.sel(**{dim1: actual_value})

        # Calculate the real gap size percentage and print it with the relevant value (e.g. date) to give insights
        real_gap_size = round(np.sum(np.isnan(actual_matrix)).values.item() / actual_matrix.size * 100)
        print("Actual matrix:", actual_value)
        print("Real gap size: ", real_gap_size, "%")

        # Save the original array
        actual_matrix.to_zarr(self.directory + "actual.zarr", mode="w")

        # If requested create artificial data gaps which values will be estimated later on
        if self.artificial_gaps:
            self.create_gaps(actual_matrix, str(actual_value))

    def create_gaps(self, actual_matrix: xr.DataArray, actual_value: str) -> None:
        """
        Create artificial data gaps.

        This method creates artificial data gaps in the desired array based on the specified gap sizes.

        Args:
            actual_matrix (xr.DataArray): Original array in which the gaps are created.
            actual_value (str): The relevant layer (e.g. date) of the array.

        Returns:
            None
        """
        # Find the indices of non-NaN values in the original data
        non_nan_indices = np.argwhere(~np.isnan(actual_matrix.values))
        # Define a new directory for saving data with artificial gaps
        new_directory = self.directory + "GapImitation/"
        shutil.rmtree(new_directory, ignore_errors=True)
        os.makedirs(new_directory, exist_ok=True)

        # Iterate through different gap sizes to create them in the original array
        gap_creation_count = 0
        for gap_size in self.artificial_gaps:
            # Calculate the absolute gap size based on the original data
            gap_size_absolute = round(gap_size * actual_matrix.size)
            # Check if there are enough non-NaN values to create the desired gap size and skip them if not
            if len(non_nan_indices) < gap_size_absolute:
                continue

            # Create a copy of the original data to insert the artificial gaps
            array_with_gaps = actual_matrix.copy()
            # Randomly select indices for creating artificial gaps
            selected_indices = np.random.choice(non_nan_indices.shape[0], gap_size_absolute, replace=False)
            selected_indices = non_nan_indices[selected_indices]
            # Loop through each of these selected indices and insert -100 as a value
            # avoid recursion error by making a copy of the xarray values
            array_with_gaps_values = array_with_gaps.values
            for index in selected_indices:
                array_with_gaps_values[index[0], index[1]] = -100
            array_with_gaps.values = array_with_gaps_values

            # Save the data with artificial gaps in the GapImitation directory
            array_with_gaps.to_zarr(new_directory + actual_value + "_" + str(gap_size) + ".zarr", mode="w")
            gap_creation_count += 1

        # Format the different gap sized in order to print them to give insights
        formatted_gaps = [f"{float(element * 100)}%" for element in self.artificial_gaps]
        print(gap_creation_count, "array(s) with the following gaps were created:", ', '.join(formatted_gaps[:gap_creation_count]))
        print(f"These arrays are saved in /{self.directory}")
        if gap_creation_count < len(formatted_gaps):
            print("However, the original array doesn't contain enough non-Nan values to imitate the following gap sizes:", ', '.join(formatted_gaps[gap_creation_count:]))
