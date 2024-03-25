# ML-Toolkits

The ML Toolkits provide a set of best practice Python-based Jupyter Notebooks that showcase the implementation of the three start-of-the-art Machine Learning libraries (1) scikit-learn, (2) PyTorch and (3) TensorFlow based on the Earth System Data Cube.

## Installation

Create a conda package by executing: 
```bash
conda build .
```

Determine your Anaconda path with:
```bash
which conda
```

You can install mltools using the resulting *.tar.bz2 file:
```bash
conda install /path/to/anaconda/conda-bld/noarch/mltools-0.1.0-py_0.tar.bz2
```

Make sure you have Python version 3.8 or higher.

If you're planning to use `mltools` with TensorFlow or PyTorch, set up these frameworks properly in your Conda environment. 

## Features

- Data preprocessing and normalization functions
- Distributed training framework compatible with PyTorch
- Utilities and sampling techniques for working with data cubes

## Usage

To use mltools in your project, simply import the necessary module:

```python
from src.mltools.data_processing import normalize, standardize
from src.mltools.torch_training import train_one_epoch
# Other imports...
```

You can then call the functions directly:

```python
# Normalizing data
normalized_data = normalize(your_data, data_min, data_max)

# Training a model for one epoch
model, train_pred, last_loss = train_one_epoch(epoch_index, training_loader, model, loss_fn, optimizer, device)
```

Checkout the `Examples` directory to discover additional use cases.

## Distributed Training with `mltools`

To utilize `mltools` for distributed training, navigate to the `Examples` directory. Use the following command with `torchrun` to initiate the process:

```bash
torchrun --standalone --nproc_per_node=<number_of_processes> distributed_training.py <epochs>
```

Replace `<number_of_processes>` with the number of processes you wish to run per node, and `<epochs>` with the total number of training epochs.

## Changes

For a complete list of changes, see the [CHANGELOG](https://github.com/deepesdl/ML-Toolkits/blob/develop/CHANGELOG.md).

## License

MLTools is released under the MIT License. See the [LICENSE](https://github.com/deepesdl/ML-Toolkits/blob/develop/LICENSE) file for more details.
