import torch
from torch.utils.data import DataLoader


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
            metrics: list = None,
            epochs: int = 10,
            task_type: str = "supervised",
            mlflow_run=None,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> None:
        self.model = model.to(device)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.best_model_path = best_model_path
        self.early_stopping = early_stopping
        self.patience = patience
        self.best_val_loss = float('inf')
        self.strikes = 0
        self.metrics = metrics if metrics is not None else [torch.nn.MSELoss(reduction='mean')]
        self.max_epochs = epochs
        self.device = device
        self.task_type = task_type
        self.mlflow_run = mlflow_run

    def _run_batch(self, inputs, targets):
        """
        Runs a single batch of training data through the model.
        """
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.metrics[0](outputs, targets)
        if loss.requires_grad:  # Check if loss requires gradients
            loss.backward()
        return loss.item()

    def _run_epoch(self, epoch: int):
        """
        Runs a single epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        total_count = 0
        for batch in self.train_data:
            if self.task_type == 'supervised':
                inputs, targets = batch
            else:
                inputs = batch
                targets = inputs
            with torch.set_grad_enabled(True):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                loss = self._run_batch(inputs, targets)
            total_loss += loss * len(inputs)
            total_count += len(inputs)

        print(f"Epoch {epoch}: Average Loss: {(total_loss / total_count):.4e}")
        if self.mlflow_run:  # Check if an MLflow run instance is available
            self.mlflow_run.log_metric("training_loss", (total_loss / total_count), step=epoch)

    def _validate(self, epoch) -> float:
        """
        Validates the model on the test dataset.
        """
        self.model.eval()
        total_loss = 0.0
        total_count = 0
        for batch in self.test_data:
            if self.task_type == 'supervised':
                inputs, targets = batch
            else:
                inputs = batch
                targets = inputs
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.metrics[0](outputs, targets)
                total_loss += loss.item() * len(inputs)
                total_count += len(inputs)

        avg_val_loss = total_loss / total_count
        print(f"Epoch {epoch}: Validation Loss: {avg_val_loss:.4e}")
        if self.mlflow_run:
            self.mlflow_run.log_metric("validation_loss", avg_val_loss, step=epoch)
        return avg_val_loss

    def train(self):
        """
        The main training loop.
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

            if self.early_stopping and self.strikes > self.patience:
                print('Stopping early due to no improvement.')
                break

            # Load the best model weights at the end of training
        self.model.load_state_dict(torch.load(self.best_model_path))
        print("Loaded best model weights.")
        if self.mlflow_run:
            self.mlflow_run.pytorch.log_model(self.model, "model")
            print("Log best model weights.")

        return self.model
