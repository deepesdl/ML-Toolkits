# Changelog

## 0.2.0 - [2024-04-11]
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

## 0.1.0 - [2024-03-13]

### Added
- Transformed `mltools.py` into a Python package with modules: `cube_utilities.py`, `data_processing.py`, `torch_training.py`, `sampling.py`.
- New functions integrated into the corresponding modules based on functionality.
- Two new methods in `cube_utilities.py`:
  - `get_chunk_by_index` for retrieving data cube chunks by index.
  - `rechunk_cube` for rechunking the data cube.
- `distributed_training.py` added to the `mltools` package for PyTorch distributed training.
- Corresponding `distributed_training.py` example added to the `Examples` directory for practical use.
- Pip packaging support with the `pip_env` directory, including `setup.py` and `requirements.txt`.
- Conda packaging option with the `conda_recipe` directory containing `meta.yaml`.

### Changed
- Organize the initial `mltools.py` functions by functionality and expand the collection with new functions.
- Improved `get_statistics` (`data_processing.py`) method performance by utilizing Dask's parallel computation capabilities.

### Fixed
- Removed semicolons in use cases in the `Examples` directory to maintain Python syntax.
