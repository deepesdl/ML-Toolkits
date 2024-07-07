import os
import joblib
import numpy as np
from sklearn.base import BaseEstimator
from typing import Union, Tuple, Optional, Callable, List, Any


class Trainer:
    """
    A trainer class for training scikit-learn models using either a PyTorch DataLoader or numpy arrays.
    This allows for flexible data input for large datasets or in-memory data.
    """

    def __init__(
            self,
            model: BaseEstimator,
            train_data: Union[Any, Tuple[np.ndarray, np.ndarray]],
            test_data: Union[Any, Tuple[np.ndarray, np.ndarray]] = None,
            metrics: List[Callable] = None,
            model_path: str = None,
            batch_training: bool = False,
            mlflow_run=None,
            task_type: str = 'supervised'
        ):
        """
        Initialize a Trainer object.

        Attributes:
            model (BaseEstimator): A scikit-learn estimator that supports partial_fit.
            train_data (Union[DataLoader, Tuple[np.ndarray, np.ndarray]]): PyTorch DataLoader for batch training or a tuple of numpy arrays (X_train, y_train).
            test_data (Union[DataLoader, Tuple[np.ndarray, np.ndarray]]): PyTorch DataLoader for batch validation/testing or a tuple of numpy arrays (X_test, y_test).
            metrics (List[Callable]): A list of functions that compute a metric between predictions and true values.
            model_path (str): Path to save the best model.
            batch_training (bool): Whether to use batch training; if False, the model will be trained on complete data at once.
            mlflow_run: An MLflow run instance to log training and validation metrics.
            task_type (str): The type of task, either 'supervised' or 'unsupervised'.
        """
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.metrics = metrics
        self.model_path = model_path
        self.batch_training = batch_training
        self.mlflow_run = mlflow_run
        self.task_type = task_type
        self.model_name = os.path.basename(os.path.normpath(self.model_path))

    def run_batch_training(self, mode: str = 'Training') -> Optional[float]:
        """
        Perform batch training or validation.

        Args:
            mode (str): Mode of operation, either 'Training' or 'Validation'.

        Returns:
            Optional[float]: The average score computed using the specified metrics.
        """
        if self.test_data and self.metrics:
            metric_sums = {name: 0.0 for name in self.metrics.keys()}
            count = 0
            if mode == 'Training':
                for batch in self.train_data:
                    if self.task_type == 'supervised':
                        inputs, targets = batch
                        inputs = inputs.numpy()
                        targets = targets.numpy()
                        self.model.partial_fit(inputs, targets)
                    else:
                        batch = batch.numpy()
                        self.model.partial_fit(batch)

                data_loader = self.train_data
            else:
                data_loader = self.test_data

            for batch in data_loader:
                if self.task_type == 'supervised':
                    inputs, targets = batch
                    inputs = inputs.numpy()
                    targets = targets.numpy()
                    for name, metric in self.metrics.items():
                        metric_value = metric(targets, self.model.predict(inputs))
                        metric_sums[name] += metric_value * len(inputs)

                else:
                    inputs = batch.numpy()
                    for name, metric in self.metrics.items():
                        metric_value = metric(inputs, self.model.predict(inputs))
                        metric_sums[name] += metric_value * len(inputs)
                count += len(inputs)
            avg_metrics = {name: total / count for name, total in metric_sums.items()}
            for name, value in avg_metrics.items():
                print(f"{mode} {name}: {value:.4f}")
            return avg_metrics
        return None

    def train(self) -> BaseEstimator:
        """
        Train the model using the specified data.

        Returns:
            BaseEstimator: The trained scikit-learn model.
        """
        if not self.batch_training:
            if self.task_type == 'supervised':
                X_train, y_train = self.train_data
                X_test, y_test = self.test_data
                self.model.fit(X_train, y_train)

                train_scores = {name: metric(y_train, self.model.predict(X_train)) for name, metric in
                                self.metrics.items()}
                for name, score in train_scores.items():
                    print(f"Training | {name}: {score:.4f}")

                val_scores = {name: metric(y_test, self.model.predict(X_test)) for name, metric in self.metrics.items()}
                for name, score in val_scores.items():
                    print(f"Validation | {name}: {score:.4f}")
            else:
                X_train = self.train_data
                self.model.fit(X_train)

                # For unsupervised tasks, we evaluate using the training data itself
                train_pred = self.model.predict(X_train)

                # Calculating the scores based on available metrics
                train_scores = {name: metric(X_train, train_pred) for name, metric in self.metrics.items()}
                for name, score in train_scores.items():
                    print(f"{name}: {score:.4f}")

        else:
            train_scores = self.run_batch_training(mode='Training')
            if self.task_type == 'supervised':
                val_scores = self.run_batch_training(mode='Validation')

        if self.mlflow_run:
            # Logging metrics
            for name, score in train_scores.items():
                self.mlflow_run.log_metric(f"training_{name}", score)
            if self.task_type == 'supervised':
                for name, score in val_scores.items():
                    self.mlflow_run.log_metric(f"validation_{name}", score)

        if self.model_path:
            # Pickling the model to save
            joblib.dump(self.model, self.model_path)
            print(f"Model parameters saved.")

        if self.mlflow_run:
            self.mlflow_run.log_artifact(self.model_path, self.model_name)

        return self.model

