# Changelog

## [1.0.1] - 2024-07-07

### Removed
- Files that were unintentionally included in version 1.0.0 during the build have been removed to streamline the package and correct the release content.

## [1.0.0] - 2024-07-07

### Added
- `cube_insights` module introduced to provide an overview of the data cube state.
  - `get_insights` function for extracting and printing various characteristics of the data cube.
  - `get_gap_heat_map` for generating heat maps of value counts (non-NaN values) across dimensions.
- `training` module: 
  - `plot_loss` method, accessible if `create_loss_plot` is true in Trainer classes (`pytorch`, `pytorch_distributed`, `sklearn`, `tensorflow`).
  - If `mlflow_run` specified, models are saved under the name extracted from user defined model path instead of saving them under 'model' on `MlFlow` server.
  - Option to define validation metrics dictionary added to Trainer classes added.
- `datasets` module: 
  - Introduced `MultiProcSampler` for efficient creation of train/test sets as zarr data cubes.
- `cube_utilities` module:
  - Added `split_chunk` to divide a chunk of the data cube into machine learning samples.
  - Added `get_dim_range` for retrieving minimum and maximum values of specific cube dimensions.
  - `assign_dims` function added to map user-defined dimension names for later transformation to xarray.
- preprocessing module:
  - Added `fill_masked_data` to fill NaNs using different methods (mean, noise, constant).

### Changed:
- `xr_plots` module:
  - Renamed from `geo_plots` and transformed the `plot_geo_data` to `plot_slice`, utilizing NumPy instead of GeoPandas for increased performance and broader usability.
- `gapfilling` module:
  - Generalized to accommodate any naming convention of cube dimensions.
  - Enabled user-defined directory storage for additional predictors extracted by `HelpingPredictor`.
- `datasets` module: 
  - Updated `LargeScaleXrDataset` for both, `PyTorch` and `TensorFlow`, and the `XrDataset` class to handle multidimensional data samples.
- preprocessing module:
  - `drop_nan_values` and `apply_filter` methods updated to handle multi-dimensional data.
  - Introduced `drop_sample` option in `apply_filter` to decide on dropping samples or setting unmatched values to NaN.
  - Updated `drop_nan_values` to utilize a mask for sample validity, dropping invalid samples or those with NaN values in valid data.

### Updated Use Cases:
- Added and updated examples in the `Examples` directory:
  - `distributed_dataset_creation.py` demonstrating the use of `MultiProcSampler`.
  - Updated `distributed_training.py` to utilize the new train and test sets.
  - New masked learning examples on ESDCs (`use_case_lst_pytorch_masked.ipynb` and `use_case_lst_tensorflow_masked.ipynb`) for coastal region predictions, excluding water regions.

## [0.0.*] - 2024-05-14

### Added
- `training` module:
  - Trainer classes for `sklearn`, `PyTorch`, and `TensorFlow` to improve usability.
  - `pytorch_distributed.py` for distributed machine learning (originally `distributed_training.py`).
- `datasets` module:
  - `XrDataset` for smaller `xarray` data manageable in memory.
  - `LargeScaleXrDataset` for `PyTorch` and `TensorFlow` to handle large datasets by iterating over (batches of) chunks.
  - Enabled batch training (partial fit) for the `training.sklearn` trainer class using both `LargeScaleXrDataset` (from `datasets.pytorch_xr`) or `XrDataset`, integrated with a `PyTorch` data loader.
  - Configuration options  for `XrDataset` and both `LargeScaleXrDataset` to select training data
  - `prepare_dataloader` method for `PyTorch` and `prepare_dataset` for `TensorFlow` to configure data processing during training (e.g., batch size, distributed training).
- `preprocessing.py` module with methods:
  - `apply_filter` for filtering training data based on a filter variable contained in the dataset.
  - `drop_nan_values` to remove data points containing any NaN values.
- Published the `ml4xcube` package (v0.0.6) on PyPI and Conda-forge.

### Changed
- Renamed `mltools` to `ml4xcube` due to name conflict.
- Updated `gapfilling` module:
  - Added a progress bar to visualize progress during gap filling.
  - Updated final print statement to show the location of gap-filled data.
- Renamed `rand` method to `assign_rand_split` in `data_assignment` module and unified its usage with `assign_block_split` to improve usability.
- Updated the use cases withing the `Examples` directory to demonstrate new / edited functionalities.
- Renamed module `distributed_training`  to `pytorch_distributed`

### Removed
- Removed `torch_training` module including the containing methods `train_one_epoch` and `test`.

## [Unreleased] - 2024-04-11

### Added
- New functionality for gap filling in datasets, implemented in the `gap_dataset.py` and `gap_filling.py` scripts within the `mltools` package. This includes:
    - The `GapDataset` class for handling datasets with data gaps, enabling the slicing of specific dimensions and the addition of artificial gaps for testing gap-filling algorithms.
    - The `EarthSystemDataCubeS3` subclass to facilitate operations on ESDC.
    - The `Gapfiller` class to fill gaps using machine learning models, with a focus on Support Vector Regression (SVR). This class supports various hyperparameter search methods and predictor strategies for gap filling.
    - Methods in `gap_dataset.py` for retrieving additional data matrices as predictors and processing the actual data matrix for gap analysis.
- Example script (`gapfilling_process.py`) and detailed usage instructions for the new gap-filling functionality, demonstrating how to apply these methods to real-world datasets.
- New `geo_plots.py` script within the `mltools` package, which includes:
    - `plot_geo_data` function for creating geographical plots of data from a DataFrame, complete with customizable visual features and the ability to save figures.

### Changed
- The `data_processing` module of the `mltools` package has been renamed to `statistics` to more accurately reflect its purpose and functionality.
- Updated use cases in the `Examples` directory now utilize the `plot_geo_data` function for enhanced visualization of geographic data.
- Revision of Examples: Updated Python examples for `sklearn`, `PyTorch`, and `TensorFlow` to be compatible with the current versions of these packages.

### Improved
- Streamlined the Conda (and pip) package creation process to facilitate faster and more efficient package builds with Miniconda.

##  [Unreleased] - 2024-03-13

### Added
- Transformed `mltools.py` into a Python package with modules: `cube_utilities.py`, `data_processing.py`, `torch_training.py`, `sampling.py`.
- New functions integrated into the corresponding modules based on functionality.
- Two new methods in `cube_utilities.py`:
  - `get_chunk_by_index` for retrieving data cube chunks by index.
  - `rechunk_cube` for re-chunking the data cube.
- `distributed_training.py` added to the `mltools` package for PyTorch distributed training.
- Corresponding `distributed_training.py` example added to the `Examples` directory for practical use.
- Pip packaging support with the `pip_env` directory, including `setup.py` and `requirements.txt`.
- Conda packaging option with the `conda_recipe` directory containing `meta.yaml`.

### Changed
- Organize the initial `mltools.py` functions by functionality and expand the collection with new functions.
- Improved `get_statistics` (`data_processing.py`) method performance by utilizing Dask's parallel computation capabilities.

### Fixed
- Removed semicolons in use cases in the `Examples` directory to maintain Python syntax.
