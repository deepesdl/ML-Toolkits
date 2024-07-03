import os
import torch
from typing import Callable, Dict
from torch.utils.data import DataLoader
from ml4xcube.training.train_plots import plot_loss


class Trainer:
    """
    A trainer class for training PyTorch models on single or no GPU systems.
    """
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            test_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            best_model_path: str,
            early_stopping: bool = True,
            patience: int = 10,
            loss: Callable = None,
            metrics: Dict[str, Callable] = None,
            epochs: int = 10,
            mlflow_run=None,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            create_loss_plot: bool = False,
    ):
        """
        Initialize the Trainer.

        Attributes:
            model (torch.nn.Module): The PyTorch model to train.
            train_data (DataLoader): DataLoader for the training data.
            test_data (DataLoader): DataLoader for the validation/test data.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            best_model_path (str): Path to save the best model.
            early_stopping (bool): Enable or disable early stopping. Defaults to True.
            patience (int): Number of epochs to wait for improvement before stopping early.
            loss (torch.nn.Module): Loss function.
            metrics (Dict[str, Callable]): Dictionary of metrics to evaluate, with metric names as keys and metric functions as values.
            epochs (int): Number of training epochs.
            mlflow_run: An MLflow run instance to log training and validation metrics.
            device (torch.device): Device to run the training on.
            create_loss_plot (bool): Whether to create a plot of training and validation loss.
        """
        self.model = model.to(device)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.best_model_path = best_model_path
        self.early_stopping = early_stopping
        self.patience = patience
        self.best_val_loss = float('inf')
        self.strikes = 0
        self.loss = loss
        self.metrics = metrics
        self.max_epochs = epochs
        self.device = device
        self.mlflow_run = mlflow_run
        self.model_name = os.path.basename(os.path.normpath(self.best_model_path))
        self.val_list = list()
        self.train_list = list()
        self.create_loss_plot = create_loss_plot

    def _run_batch(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Runs a single batch of training data through the model.

        Args:
            inputs (torch.Tensor): Input data for the model.
            targets (torch.Tensor): Target data for the training.

        Returns:
            float: The loss value for the batch.
        """
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        if loss.requires_grad:  # Check if loss requires gradients
            loss.backward()
        return loss.item()

    def _run_epoch(self, epoch: int) -> None:
        """
        Runs a single epoch of training.

        Args:
            epoch (int): The current epoch number.
        """
        self.model.train()
        total_loss = 0.0
        total_count = 0
        for batch in self.train_data:
            inputs, targets = batch
            with torch.set_grad_enabled(True):
                if inputs.numel() == 0: continue
                loss = self._run_batch(inputs, targets)
            total_loss += loss * len(inputs)
            total_count += len(inputs)

        train_avg_loss = total_loss / total_count
        self.train_list.append(train_avg_loss)

        print(f"Epoch {epoch + 1}: Average Loss: {train_avg_loss:.4e}")
        if self.mlflow_run:  # Check if an MLflow run instance is available
            self.mlflow_run.log_metric("training_loss", (total_loss / total_count), step=epoch)

    def _validate(self, epoch: int) -> float:
        """
        Validates the model on the test dataset.

        Args:
            epoch (int): The current epoch number.

        Returns:
            float: The average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        total_count = 0
        metric_sums = {}
        if self.metrics is not None:
            metric_sums = {name: 0.0 for name in self.metrics.keys()}
        for batch in self.test_data:
            inputs, targets = batch
            if inputs.numel() == 0: continue
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)
                total_loss += loss.item() * len(inputs)
                total_count += len(inputs)

                if self.metrics is not None:
                    for name, metric in self.metrics.items():
                        metric_value = metric(outputs, targets).item()
                        metric_sums[name] += metric_value * len(inputs)

        avg_val_loss = total_loss / total_count
        print(f"Epoch {epoch + 1}: Validation Loss: {avg_val_loss:.4e}")

        if self.metrics is not None:
            avg_metrics = {name: total / total_count for name, total in metric_sums.items()}
            for name, value in avg_metrics.items():
                print(f"Epoch {epoch + 1}: {name}: {value:.4e}")

        if self.mlflow_run:
            self.mlflow_run.log_metric("validation_loss", avg_val_loss, step=epoch)

        self.val_list.append(avg_val_loss)
        return avg_val_loss

    def train(self) -> torch.nn.Module:
        """
        The main training loop.

        Returns:
            torch.nn.Module: The trained model.
        """
        for epoch in range(self.max_epochs):
            self._run_epoch(epoch)
            val_loss = self._validate(epoch)

            if val_loss < self.best_val_loss:
                self.strikes = 0
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved with validation loss: {val_loss}")
            else:
                self.strikes += 1

            if self.early_stopping and self.strikes >= self.patience:
                print('Stopping early due to no improvement.')
                break

            # Load the best model weights at the end of training
        self.model.load_state_dict(torch.load(self.best_model_path))
        print("Loaded best model weights.")
        if self.mlflow_run:
            self.mlflow_run.pytorch.log_model(self.model, "model")
            print("Log best model weights.")

        if self.create_loss_plot:
            plot_loss(self.train_list, self.val_list)

        return self.model
