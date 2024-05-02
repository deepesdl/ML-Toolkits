# ML-Toolkits

The ML Toolkits provide a set of best practice Python-based Jupyter Notebooks that showcase the implementation of the three start-of-the-art Machine Learning libraries (1) scikit-learn, (2) PyTorch and (3) TensorFlow based on the Earth System Data Cube.

## Installation

You can install `ml4xcube` directly via pip:
```bash
pip install ml4xcube
```

Make sure you have Python version 3.8 or higher.

If you're planning to use `ml4xcube` with TensorFlow or PyTorch, set up these frameworks properly in your Conda environment. 

## Features

- Data preprocessing and normalization/standardization functions
- Gap filling features
- Dataset creation and train-/ test split sampling techniques
- Trainer classes for `sklearn`, `TensorFlow` and `PyTorch`
- Distributed training framework compatible with `PyTorch`
- chunk utilities for working with data cubes

## Usage

To use ml4xcube in your project, simply import the necessary module:

```python
from ml4xcube.statistics import normalize, standardize
from ml4xcube.training.pytorch import Trainer
# Other imports...
```

You can then call the functions directly:

```python
# Normalizing data
normalized_data = normalize(your_data, data_min, data_max)

# Trainer instance
trainer = Trainer(
    model           = reg_model,
    train_data      = train_loader,
    test_data       = test_loader,
    optimizer       = optimizer,
    best_model_path = best_model_path,
    early_stopping  = True,
    patience        = 3,
    epochs          = epochs
)

# Start training
reg_model = trainer.train()
```

## License

ml4xcube is released under the MIT License. See the [LICENSE](https://github.com/deepesdl/ML-Toolkits/blob/master/LICENSE) file for more details.
