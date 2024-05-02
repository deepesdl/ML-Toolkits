import joblib
import numpy as np
from typing import Union, Tuple
from sklearn.metrics import mean_squared_error


class Trainer:
    """
    A trainer class for training scikit-learn models using either a PyTorch DataLoader or numpy arrays.
    This allows for flexible data input for large datasets or in-memory data.

    Attributes:
        model: A scikit-learn estimator that supports partial_fit.
        train_data (Union[DataLoader, Tuple[np.ndarray, np.ndarray]]): PyTorch DataLoader for batch training or a tuple of numpy arrays (X_train, y_train).
        test_data (Union[DataLoader, Tuple[np.ndarray, np.ndarray]]): PyTorch DataLoader for batch validation/testing or a tuple of numpy arrays (X_test, y_test).
        metrics (list, optional): A list of functions that compute a metric between predictions and true values.
        model_path (str, optional): Path to save the best model.
        batch_training (bool): Whether to use batch training; if False, the model will be trained on complete data at once.
        mlflow_run: An MLflow run instance to log training and validation metrics.
    """

    def __init__(
            self,
            model,
            train_data,
            test_data,
            metrics: list = [mean_squared_error],
            model_path: str = None,
            batch_training: bool = True,
            mlflow_run=None
    ) -> None:
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.metrics = metrics
        self.model_path = model_path
        self.batch_training = batch_training
        self.mlflow_run = mlflow_run

    def run_batch_training(self, mode='Training'):
        """Validates the model on the test dataset if provided."""
        if self.test_data and self.metrics:
            total_score = 0
            count = 0
            if mode == 'Training':
                for inputs, targets in self.train_data:
                    inputs = inputs.numpy()
                    targets = targets.numpy()
                    self.model.partial_fit(inputs, targets)
                data_loader = self.train_data
            else:
                data_loader = self.test_data

            for inputs, targets in data_loader:
                inputs = inputs.numpy()
                targets = targets.numpy()
                score = self.metrics[0](targets, self.model.predict(inputs))
                total_score += score * len(inputs)
                count += len(inputs)
            avg_score = total_score / count
            print(f"{mode} Score: {avg_score:.4f}")
            return avg_score
        return None

    def train(self):
        if not self.batch_training:
            X_train, y_train = self.train_data
            X_test, y_test = self.train_data
            self.model.fit(X_train, y_train)
            avg_train_score = self.metrics[0](y_train, self.model.predict(X_train))
            print(f"Training Score: {avg_train_score:.4f}")
            avg_val_score = self.metrics[0](y_test, self.model.predict(X_test))
            print(f"Validation Score: {avg_val_score:.4f}")
        else:
            avg_train_score = self.run_batch_training(mode='Training')
            avg_val_score = self.run_batch_training(mode='Validation')

        if self.mlflow_run:
            # Logging metrics
            self.mlflow_run.log_metric("training_score", avg_train_score)
            self.mlflow_run.log_metric("validation_score", avg_val_score)

        if self.model_path:
            # Pickling the model to save
            joblib.dump(self.model, self.model_path)
            print(f"Model parameters saved.")

        if self.mlflow_run:
            self.mlflow_run.log_artifact(self.model_path, "model")

        return self.model

