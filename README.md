# ML-Toolkits

The ML Toolkits provide a set of best practice Python-based Jupyter Notebooks that showcase the implementation of the three start-of-the-art Machine Learning libraries (1) scikit-learn, (2) PyTorch and (3) TensorFlow based on the Earth System Data Cube.

## Installation

Creaqte a Python package by executing: 
```bash
python setup.py sdist bdist_wheel
```

You can install mltools using the resulting *.whl file:
```bash
pip install ./dist/mltools-0.1-py3-none-any.whl
```

Make sure you have Python version 3.8 or higher.

## Features

- Data preprocessing and normalization functions
- Distributed training framework compatible with PyTorch
- Utilities for working with ML data structures, such as datasets and data loaders

## Usage

To use mltools in your project, simply import the necessary module:

```python
from mltools.data_processing import normalize, standardize
from mltools.torch_training import train_one_epoch
# Other imports...
```

You can then call the functions directly:

```python
# Normalizing data
normalized_data = normalize(your_data, data_min, data_max)

# Training a model for one epoch
model, train_pred, last_loss = train_one_epoch(epoch_index, training_loader, model, loss_fn, optimizer, device)
```

Checkout the Examples directory to discover additional usage samples.
## License

MLTools is released under the MIT License. See the [LICENSE](https://github.com/deepesdl/ML-Toolkits/blob/develop/LICENSE) file for more details.
