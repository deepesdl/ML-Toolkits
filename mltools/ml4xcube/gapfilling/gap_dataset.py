import os
import sys
import time
import shutil
import datetime
import numpy as np
import xarray as xr
from tqdm import tqdm

"""
In this file we slice a specific dimensions (e.g. area and time range) from the data cube with the values for the specified variable.
For every time step the corresponding array will be stored as historical training data.
One file will be randomly chosen or specified as actual matrix and artificial gaps can be added.
This file step will be stored separately so the results of the gapfilling process (and if requested, 
of the estimated values of the artificial gaps) can be compared later on.
"""


class GapDataset:
    """
    Represents a dataset for handling data gaps.

    Attributes:
        ds_name (str): The name of the dataset.
        variable (str): The variable of interest.
        dimensions (dict): Dictionary containing dimension ranges (e.g. lat, lon, times).
        artificial_gaps (list): List of artificial gap sizes; None if no artificial gaps should be created
        actual_matrix (str or datetime.date): Specifies the actual data matrix or 'Random' for random selection.
        directory (str): The directory where data will be stored.
        extra_data: Additional data used as predictors (e.g. Land Cover Classes).
        sliced_ds: The sliced dataset.
    """
    def __init__(self, ds, ds_name='Test123',
                 dimensions=None,
                 artificial_gaps=None, actual_matrix='Random'):
        if dimensions is None:
            dimensions = {'lat': (54, 48),
                          'lon': (6, 15),
                          'times': (datetime.date(2008, 11, 1), datetime.date(2008, 12, 31))}
        self.artificial_gaps = artificial_gaps
        self.actual_matrix = actual_matrix
        self.ds_name = ds_name
        self.artificial_gaps = artificial_gaps
        self.dimensions = dimensions
        self.directory = os.path.dirname(os.getcwd()) + '/application_results/' + ds_name + "/" if \
            os.getcwd().split('/')[-1] != 'gapfilling' else 'application_results/' + ds_name + "/"
        self.extra_data = None
        self.sliced_ds = None
        self.ds = ds

    def get_data(self):
        """
         Retrieve and process (area-)specific data.

         This method performs the following tasks:
         - Creates a directory or cleans it if it already exists.
         - Slices the dimensions from a global dataset.
         - Retrieves historical data files for use as training data.
         - Retrieves additional data (e.g., land cover classes) for use as predictors.
         - Optionally creates artificial data gaps for gap filling.
         """
        start_time = time.time()
        # Create a directory or clean it if it already exists
        self.make_directory(self.directory)
        # Slice the dimensions from a global dataset
        self.slice_dataset()
        # Retrieve historical data files to use them as trainings data later on
        self.get_history_files()
        # Retrieve land cover data or other extra matrix to use them as predictors
        self.get_extra_matrix()
        # If requested create artificial data gaps which values will be estimated later on
        self.process_actual_matrix()

        print("runtime:", round(time.time() - start_time, 2))

    def make_directory(self, directory):
        """
        Create a directory or remove it if it already exists.

        Args:
            directory (str): The directory path to be created or removed.

        Returns:
            None
        """
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

    def slice_dataset(self):
        """
        Abstract method to slice the dataset.

        This method should be overridden by subclasses to implement specific slicing logic.

        Raises:
            Exception: This method should be implemented in inherited classes.
        """
        raise Exception("You need to initialize an inherited class!")

    def get_history_files(self):
        """
        Retrieve historical data files for training and quality tracking.

        This method retrieves historical data files for the specified dimensions (e.g. area and time range),
        calculates data quality metrics, and saves the data as NumPy files in the History directory.

        Returns:
            None
        """
        # Define the directory where historical data will be stored and create it if it does not exist
        directory = self.directory + "History/"
        self.make_directory(directory)

        # Initialize a dictionary to track data quality
        quality = {'complete': 0, 'empty': 0, 'gaps': 0}
        print("No of files:", len(self.sliced_ds.time))

        # Iterate through each time step in the sliced area and track process with a status bar
        for t in tqdm(self.sliced_ds.time, file=sys.stdout, colour='GREEN',
                      bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):

            # Extract data for the current time step and convert it to a NumPy array
            new_ds = self.sliced_ds.sel(time=t.data).to_numpy()
            # Count the number of NaN values in the data
            nan_values = np.isnan(new_ds).sum()
            # Calculate the total number of values in the data
            total_values = new_ds.size
            # Calculate the number of known (non-NaN) values
            known_values = total_values - nan_values
            # Calculate the percentage of known values in the data
            percentage = int(known_values / total_values * 100)
            # Determine the status of the data based on completeness
            status = 'complete' if known_values == total_values else 'empty' if known_values == 0 else 'gaps'
            # Generate a filename based on the timestamp and data completeness percentage
            filename = np.datetime_as_string(t.values, unit='s').partition('T')[0] + "_" + str(percentage)
            # Save the data as a NumPy file in the History directory
            np.save(os.path.join(directory, filename), new_ds)
            # Update the data quality dictionary based on data status
            quality[status] += 1

        # Print the overall data quality structure
        print('Structure', quality)

    def get_extra_matrix(self):
        """
        Retrieve additional data matrix.

        This method should be overridden by subclasses to retrieve additional data, such as land cover data,
        to use as predictors for gap filling.

        Returns:
            None
        """
        pass

    def process_actual_matrix(self):
        """
        Process the actual data matrix.

        This method processes the actual data matrix, including selecting a random historical data file
        or a specified date, calculating the real gap size percentage, and creating artificial data gaps
        if requested.

        Returns:
            None
        """
        directory = self.directory + "History/"
        files = [file for file in os.listdir(directory)]
        # Select a random historical data file
        if self.actual_matrix == 'Random':
            actual_file = np.random.choice(files)
        else:
            file_prefix = self.actual_matrix.strftime("%Y-%m-%d")
            actual_file = [file for file in files if file.startswith(file_prefix)][0]
        # Calculate the real gap size percentage
        real_gap_size = abs(100 - int(actual_file.split("_")[1].split(".")[0]))
        actual_date = actual_file.split(".")[0].split("_")[0]
        print("date:", actual_date)
        print("real gap size: ", real_gap_size, "%")

        # If requested create artificial data gaps which values will be estimated later on
        if self.artificial_gaps:
            self.create_gaps(actual_file)

        # Move the original historical data file so it will not be used as training data and only for result comparison
        file_path = directory + actual_file
        destination_directory = self.directory
        new_file_name = actual_date + '_actual_matrix.npy'
        destination_path = destination_directory + new_file_name
        shutil.move(file_path, destination_path)

    def create_gaps(self, actual_file):
        """
        Create artificial data gaps.

        This method creates artificial data gaps in the historical data based on the specified gap sizes.

        Args:
            actual_file (str): The path to the historical data file to use as a template for gap creation.

        Returns:
            None
        """
        directory = self.directory + "History/"
        # Load the original historical data from the selected file
        array_og = np.load(directory + actual_file)
        # Find the indices of non-NaN values in the original data
        non_nan_indices = np.argwhere(~np.isnan(array_og))
        # Define a new directory for saving data with artificial gaps
        new_directory = self.directory + "GapImitation/"
        self.make_directory(new_directory)

        gap_creation_count = 0
        # Iterate through different gap sizes and to create them
        for gap_size in self.artificial_gaps:
            # Calculate the absolute gap size based on the original data size and gap size percentage
            gap_size_absolute = round(gap_size * array_og.size)

            # Check if there are enough non-NaN values to create the desired gap size
            if len(non_nan_indices) < gap_size_absolute:
                print("Exception: gap size", gap_size * 100, "% -> contains not enough non-NaN values. "
                                                             "No array with imitated gaps was created.")
                continue

            # Randomly select indices for creating artificial gaps
            selected_indices = np.random.choice(non_nan_indices.shape[0], gap_size_absolute, replace=False)
            selected_indices = non_nan_indices[selected_indices]
            # Create a copy of the original data with artificial gaps
            array_with_gaps = array_og.copy()
            for index in selected_indices:
                array_with_gaps[index[0], index[1]] = -100
            # Save the data with artificial gaps in the GapImitation directory
            np.save(new_directory + actual_file[:-4] + "_" + str(gap_size), array_with_gaps)
            gap_creation_count += 1

        print(gap_creation_count, "arrays with gaps were created!")
        print(f"These arrays are saved in /{self.directory}")


class EarthSystemDataCubeS3(GapDataset):
    """
    Represents a class for accessing Earth System Data Cube (ESDC) data stored on Amazon S3
    with the ability to slice and process specific data areas.

    This class is a subclass of the GapDataset class and inherits its attributes and methods.

    Methods:
    - slice_dataset(): Slice the dataset to extract the specific area, latitude, longitude, and time range.

    - get_extra_matrix(): Retrieve additional data matrix, such as Land Cover Classes (LCC), for use as predictors.

    Attributes (inherited from GapDataset):
    - ds_name (str): The name of the dataset.
    - variable (str): The variable of interest.
    - dimensions (dict): Dictionary containing dimension ranges (e.g., lat, lon, times).
    - artificial_gaps (list): List of artificial gap sizes; None if no artificial gaps should be created.
    - actual_matrix (str or datetime.date): Specifies the actual data matrix or 'Random' for random selection.
    - directory (str): The directory where data will be stored.
    - extra_data: Additional data used as predictors (e.g., Land Cover Classes).
    - sliced_ds: The sliced dataset.

    Example:
    ```
    EarthSystemDataCubeS3(ds_name='Germany', variable='land_surface_temperature',
                          dimensions={'lat': (54, 48), 'lon': (6, 15),
                                      'times': (datetime.date(2008, 1, 1), datetime.date(2008, 12, 31))},
                          artificial_gaps=[0.001, 0.005],
                          actual_matrix='Random').get_data()
    ```
    """

    def slice_dataset(self):
        """
         Slice the dataset to extract the specific area, latitude, longitude, and time range.

         This method initializes an xcube data store for accessing global data on Amazon S3,
         opens the dataset, selects the variable of interest, and slices the dataset based on
         the specified dimensions (lat, lon, and times).

         Returns:
         - None
         """
        # Create a data store for accessing global data and open the dataset
        # Slice the dataset to extract the specific area, latitude, longitude, and time range
        self.sliced_ds = self.ds.sel(lat=slice(self.dimensions['lat'][0], self.dimensions['lat'][1]),
                                lon=slice(self.dimensions['lon'][0], self.dimensions['lon'][1]),
                                time=slice(self.dimensions['times'][0], self.dimensions['times'][1]))
        print(self.ds_name, self.sliced_ds.sizes.mapping)


    def get_extra_matrix(self):
        """
        Retrieve Land Cover Classes (LCC) for use as predictors.

        This method opens a NetCDF file containing global LCC data, selects and slices the LCC data
        based on the specified latitude and longitude range, and saves it as an extra data matrix
        for use in gap filling.

        Returns:
        - None
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
        np.save(self.directory + 'extra_matrix_lcc.npy', self.extra_data)