import os
import torch
import torch.distributed as dist
from time import time
from torch.utils.data import DataLoader
from ml4xcube.training.train_plots import plot_loss
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


def ddp_init() -> None:
    """
    Initializes the distributed process group.
    Uses NCCL (NVIDIA Collective Communications Library) as the backend for GPU-based distributed training.
    Sets the current device based on the worker's local rank environment variable.
    """
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class Trainer:
    """
    A trainer class for distributed training of PyTorch models.

    Supports supervised and unsupervised (reconstruction) tasks, early stopping,
    and periodic snapshot saving.
    """
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            test_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            save_every: int,
            best_model_path: str,
            snapshot_path: str = None,
            early_stopping: bool = True,
            patience: int = 10,
            loss = None,
            metrics: list = None,
            print_loss_per_gpu: bool = False,  # New parameter to control loss printing
            create_loss_plot: bool = False
    ):
        """
        Initialize the Trainer for distributed training.

        Attributes:
            model (torch.nn.Module): The PyTorch model to train.
            train_data (DataLoader): DataLoader for the training data.
            test_data (DataLoader): DataLoader for the validation/test data.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            save_every (int): Frequency of saving training snapshots (in epochs).
            best_model_path (str): Path to save the best model.
            snapshot_path (Optional[str]): Path to save training snapshots. Defaults to None.
            early_stopping (bool): Enable or disable early stopping. Defaults to True.
            patience (int): Number of epochs to wait for improvement before stopping early. Defaults to 10.
            loss (Optional[Callable]): Loss function. Defaults to None.
            metrics (Optional[List[Callable]]): List of metrics to evaluate. Defaults to None.
            print_loss_per_gpu (bool): Whether to print loss for each GPU. Defaults to False.
            create_loss_plot (bool): Whether to create a plot of training and validation loss. Defaults to False.
        """
        self.gpu_id = int(os.environ["LOCAL_RANK"])  # GPU ID for the current process
        self.model = model.to(self.gpu_id)  # Moves model to the correct device
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.save_every = save_every  # Frequency of saving training snapshots
        self.epochs_run = 0  # Tracks the number of epochs run
        self.best_model_path = best_model_path # Path to best model computed in current training
        self.snapshot_path = snapshot_path  # Path to save snapshots
        self.early_stopping = early_stopping  # Enables/disables early stopping
        self.patience = patience  # Number of epochs to wait before early stop if no progress on the validation set
        self.best_val_loss = float('inf')  # Best validation loss for early stopping
        self.strikes = 0  # Counter for epochs without improvement
        self.loss = loss # Loss function
        self.metrics = metrics # List of metrics to compute for validation purposes
        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)  # Wraps the model for DDP
        self.print_loss_per_gpu = print_loss_per_gpu
        self.create_loss_plot = create_loss_plot
        self.train_list = list()
        self.val_list = list()

    def _load_snapshot(self, snapshot_path: str) -> None:
        """
        Loads a training snapshot to resume training.

        Args:
            snapshot_path (str): The path to the training snapshot file.
        """
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Runs a single batch of training data through the model.

        Args:
            inputs (torch.Tensor): Input data for the model.
            targets (torch.Tensor): Target data for the training.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        start_time = time()
        inputs, targets = inputs.to(self.gpu_id), targets.to(self.gpu_id)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        if loss.requires_grad:  # Check if loss requires gradients
            loss.backward()

        end_time = time()  # End timing the batch processing
        batch_processing_time = end_time - start_time  # Calculate processing time

        if self.print_loss_per_gpu:  # Check if loss printing is enabled
            # Print loss for the current GPU along with the processing time of the batch
            print(
                f"GPU {self.gpu_id} | Batch Loss: {loss.item():.4f} | Processing Time: {batch_processing_time:.2f} seconds")
        return loss

    def _run_epoch(self, epoch: int) -> None:
        """
        Runs a single epoch of training.

        Args:
            epoch (int): The current epoch number.
        """
        running_loss = 0.0
        running_size = 0
        self.train_data.sampler.set_epoch(epoch)
        for inputs, targets in self.train_data:
            with torch.set_grad_enabled(True):
                if inputs.numel() == 0: continue
                loss = self._run_batch(inputs, targets)
            running_loss += loss.item() * len(inputs)
            running_size += len(inputs)

        avg_epoch_loss = running_loss / running_size
        self.train_list.append(avg_epoch_loss)
        if self.gpu_id == 0:
            print(f"Epoch {epoch} | Average Loss: {avg_epoch_loss:.4f}")

    def _validate(self) -> float:
        """
        Validates the model on the test dataset.

        Returns:
            float: The average validation loss across all test data.
        """
        self.model.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        running_size = 0

        for inputs, targets in self.test_data:
            with torch.no_grad():  # No need to track gradients during validation
                if inputs.numel() == 0: continue
                loss = self._run_batch(inputs, targets)
            running_loss += loss.item() * len(inputs)
            running_size += len(inputs)

        # Convert running loss and size to tensors for all_reduce operation
        running_loss_tensor = torch.tensor([running_loss], device=self.gpu_id)
        running_size_tensor = torch.tensor([running_size], device=self.gpu_id)

        dist.barrier()

        # Use dist.all_reduce to sum the losses and sizes from all GPUs
        dist.all_reduce(running_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_size_tensor, op=dist.ReduceOp.SUM)

        # Compute the average loss across all GPUs and samples
        avg_val_loss = running_loss_tensor.item() / running_size_tensor.item()

        self.val_list.append(avg_val_loss)

        if self.gpu_id == 0:
            print(f"Validation Loss: {avg_val_loss:.4e}")

        return avg_val_loss

    def _save_snapshot(self, epoch: int) -> None:
        """
        Saves a training snapshot.

        Args:
            epoch (int): The current epoch number, for tracking in the snapshot.
        """
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int) -> None:
        """
        The main training loop.

        Args:
            max_epochs (int): The maximum number of epochs to train for.
        """
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0 and self.snapshot_path is not None:
                self._save_snapshot(epoch)
            dist.barrier()

            epoch_val_loss = self._validate()

            # Early Stopping Logic
            if epoch_val_loss < self.best_val_loss:
                self.strikes = 0
                self.best_val_loss = epoch_val_loss
                # Saving the best model
                torch.save(self.model.module.state_dict(), self.best_model_path)
                if self.gpu_id == 0:
                    print(f"New best model saved with validation loss: {epoch_val_loss}")
            else:
                self.strikes += 1

            if self.early_stopping:
                if self.strikes > self.patience:
                    if self.gpu_id == 0:
                        print('Stopping early due to no improvement.')
                    break

        if self.create_loss_plot:
            plot_loss(self.train_list, self.val_list)


@record
def dist_train(trainer: Trainer, total_epochs: int) -> None:
    """
    A utility function to manage the distributed training process.

    Args:
        trainer (Trainer): The Trainer instance to conduct the training.
        total_epochs (int): The total number of epochs to train the model.
    """
    trainer.train(total_epochs)
    destroy_process_group()


