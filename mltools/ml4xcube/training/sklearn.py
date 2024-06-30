import os
import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
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
            test_data: Optional[Union[Any, Tuple[np.ndarray, np.ndarray]]] = None,
            metrics: List[Callable] = [mean_squared_error],
            model_path: Optional[str] = None,
            batch_training: bool = False,
            mlflow_run=None,
            task_type: str = 'supervised'
        ):
        """
        Initialize a Trainer object.

        Attributes:
            model (BaseEstimator): A scikit-learn estimator that supports partial_fit.
            train_data (Union[DataLoader, Tuple[np.ndarray, np.ndarray]]): PyTorch DataLoader for batch training or a tuple of numpy arrays (X_train, y_train).
            test_data (Optional[Union[DataLoader, Tuple[np.ndarray, np.ndarray]]]): PyTorch DataLoader for batch validation/testing or a tuple of numpy arrays (X_test, y_test).
            metrics (List[Callable]): A list of functions that compute a metric between predictions and true values.
            model_path (Optional[str]): Path to save the best model.
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
            total_score = 0
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
                    score = self.metrics[0](targets, self.model.predict(inputs))

                else:
                    inputs = batch.numpy()
                    score = self.metrics[0](inputs, self.model.predict(inputs))
                total_score += score * len(inputs)
                count += len(inputs)
            avg_score = total_score / count
            print(f"{mode} Score: {avg_score:.4f}")
            return avg_score
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
                train_score = self.metrics[0](y_train, self.model.predict(X_train))
                print(f"Training Score: {train_score:.4f}")
                val_score = self.metrics[0](y_test, self.model.predict(X_test))
                print(f"Validation Score: {val_score:.4f}")
            else:
                X_train = self.train_data
                self.model.fit(X_train)

                # For unsupervised tasks, we evaluate using the training data itself
                train_pred = self.model.predict(X_train)

                # Calculating the scores based on available metrics
                train_score = self.metrics[0](X_train, train_pred)
                print(f"Score: {train_score:.4f}")

        else:
            train_score = self.run_batch_training(mode='Training')
            if self.task_type == 'supervised':
                val_score = self.run_batch_training(mode='Validation')

        if self.mlflow_run:
            # Logging metrics
            self.mlflow_run.log_metric("training_score", train_score)
            if self.task_type == 'supervised':
                self.mlflow_run.log_metric("validation_score", val_score)

        if self.model_path:
            # Pickling the model to save
            joblib.dump(self.model, self.model_path)
            print(f"Model parameters saved.")

        if self.mlflow_run:
            self.mlflow_run.log_artifact(self.model_path, self.model_name)

        return self.model

