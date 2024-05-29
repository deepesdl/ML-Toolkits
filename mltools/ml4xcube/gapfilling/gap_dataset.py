import os
import time
import shutil
import datetime
import numpy as np
import xarray as xr

"""
In this file we prepare the data for the gapfilling algorithm. 
We slice a specific dimensions (e.g. area and time range) from the data cube with the values for the specified variable.
If requested artificial gaps can be inserted in the array. 
"""


class GapDataset:
    """
    Represents a dataset for handling data gaps.

    Attributes:
        ds (xarray.DataArray): The original unsliced dataset.
        ds_name (str): The name of the dataset.
        dimensions (dict): Dict containing dimension ranges (e.g. lat, lon, times); sample values if no dim specified
        artificial_gaps (list): List of artificial gap sizes; None if no artificial gaps should be created
        actual_matrix (str or datetime.date): Specifies the actual data matrix or 'Random' for random selection.
        directory (str): The directory where data will be stored.
        extra_data: Additional data used as predictors (e.g. Land Cover Classes).
        sliced_ds: The sliced dataset.

    """
    def __init__(self, ds, ds_name='Test123',
                 dimensions=None,
                 artificial_gaps=None, actual_matrix='Random'):
        self.ds = ds
        self.ds_name = ds_name
        self.dimensions = dimensions
        self.artificial_gaps = artificial_gaps
        self.actual_matrix = actual_matrix

        self.directory = os.path.dirname(os.getcwd()) + '/application_results/' + ds_name + "/" if \
            os.getcwd().split('/')[-1] != 'gapfilling' else 'application_results/' + ds_name + "/"
        self.extra_data = None
        self.sliced_ds = None

    def get_data(self):
        """
         Retrieve and process (area-)specific data.

         This method performs the following tasks:
         - Creates a directory or cleans it if it already exists.
         - Slices the dimensions from a global dataset.
         - Retrieves additional data (e.g., land cover classes) for use as predictors.
         - Process the data and optionally creates artificial data gaps for gap filling.
         """
        start_time = time.time()

        # Create a directory or clean it if it already exists
        shutil.rmtree(self.directory, ignore_errors=True)
        os.makedirs(self.directory, exist_ok=True)
        # Slice the dimensions from a global dataset
        self.slice_dataset()
        # Retrieve land cover data or other extra matrix to use them as predictors
        self.get_extra_matrix()
        # If requested create artificial data gaps which values will be estimated later on
        self.process_actual_matrix()

        print("runtime:", round(time.time() - start_time, 2))

    def slice_dataset(self):
        """
        Slice the dataset to extract the specific area, latitude, longitude, and time range.

        This method slices the dataset based on the specified dimensions (lat, lon, and times) and if no dimensions
        are specified, sample values will be used.

        Returns:
            None
        """
        # Create a data store for accessing global data and open the dataset
        # Slice the dataset to extract the specific area, latitude, longitude, and time range
        if self.dimensions is None:
            self.dimensions = {'lat': (54, 48),
                               'lon': (6, 15),
                               'times': (datetime.date(2008, 11, 1), datetime.date(2008, 12, 31))}

        sliced_ds = self.ds.sel(lat=slice(self.dimensions['lat'][0], self.dimensions['lat'][1]),
                                lon=slice(self.dimensions['lon'][0], self.dimensions['lon'][1]),
                                time=slice(self.dimensions['times'][0], self.dimensions['times'][1]))
        print(self.ds_name, sliced_ds.sizes.mapping)

        # To avoid exceptions due to type errors, save the dataset and load it again
        sliced_ds.to_netcdf(self.directory + "cube.nc")
        self.sliced_ds = xr.open_dataset(self.directory + "cube.nc")[sliced_ds.name]

    def get_extra_matrix(self):
        """
        Retrieve Land Cover Classes (LCC) for use as predictors.

        This method opens a NetCDF file containing global LCC data, selects and slices the LCC data
        based on the specified latitude and longitude range, and saves it as an extra data matrix
        for use in gap filling.

        Returns:
            None
        """
        # Open the LCCS dataset from a NetCDF file with the global LCC data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(current_dir, 'helper', 'global_lcc.nc')

        # Check if the file exists at the specified path
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"File not found: {data_file}")

        # Open the LCCS dataset from the NetCDF file
        lcc_dataset = xr.open_dataset(data_file)['lccs_class']
        # Select and slice the LCCS data based on the specified latitude and longitude range
        self.extra_data = lcc_dataset.sel(lat=slice(self.dimensions['lat'][0], self.dimensions['lat'][1]),
                                          lon=slice(self.dimensions['lon'][0], self.dimensions['lon'][1]))
        self.extra_data.to_netcdf(self.directory + "extra_matrix_lcc.nc")

    def process_actual_matrix(self):
        """
        Process the actual data matrix.

        This method processes the actual data matrix, including selecting a random or specified date,
        calculating the real gap size percentage, and creating artificial data gaps if requested.

        Returns:
            None
        """
        # Select the relevant/random date with the gaps to be filled and slice the corresponding array
        dates = self.sliced_ds.coords["time"].values
        actual_date = np.random.choice(dates) if self.actual_matrix == 'Random' else np.datetime64(self.actual_matrix)
        actual_matrix = self.sliced_ds.sel(time=actual_date)

        # Calculate the real gap size percentage and print it with the relevant date to give insights
        real_gap_size = round(np.isnan(actual_matrix).sum().item() / actual_matrix.size * 100)
        actual_date = np.datetime_as_string(actual_date, unit='D')
        print("date:", actual_date)
        print("real gap size: ", real_gap_size, "%")

        # Save the original array
        actual_matrix.to_netcdf(self.directory + "actual.nc")

        # If requested create artificial data gaps which values will be estimated later on
        if self.artificial_gaps:
            self.create_gaps(actual_matrix, actual_date)

    def create_gaps(self, actual_matrix, actual_date):
        """
        Create artificial data gaps.

        This method creates artificial data gaps in the desired array based on the specified gap sizes.

        Args:
            actual_matrix (xArray): Original array in which the gaps are created.
            actual_date (str): The date of the array.

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
            for index in selected_indices:
                array_with_gaps[index[0], index[1]] = -100

            # Save the data with artificial gaps in the GapImitation directory
            array_with_gaps.to_netcdf(new_directory + actual_date + "_" + str(gap_size) + ".nc")
            gap_creation_count += 1

        # Format the different gap sized in order to print them to give insights
        formatted_gaps = [f"{float(element * 100)}%" for element in self.artificial_gaps]
        print(gap_creation_count, "arrays with the following gaps were created:", ', '.join(formatted_gaps[:gap_creation_count]))
        print(f"These arrays are saved in /{self.directory}")
        if gap_creation_count < len(formatted_gaps):
            print("However, the original array doesn't contain enough non-Nan values to imitate the following gap sizes:", ', '.join(formatted_gaps[gap_creation_count:]))