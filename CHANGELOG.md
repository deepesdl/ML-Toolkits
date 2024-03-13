
# Changelog

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
- Improved `get_statistics` (`Examples/data_processing.py`) method performance by utilizing Dask's parallel computation capabilities.

### Fixed
- Removed semicolons in use cases in the `Examples` directory to maintain Python syntax.
