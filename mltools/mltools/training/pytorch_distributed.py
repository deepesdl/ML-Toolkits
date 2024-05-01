from torch.utils.data import DataLoader
from time import time
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import torch
import os


def ddp_init():
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
            metrics: list = None,
            task_type: str = "supervised",
            print_loss_per_gpu: bool = False  # New parameter to control loss printing
    ) -> None:
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
        self.metrics = metrics if metrics is not None else [torch.nn.MSELoss(reduction='mean')]
        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)  # Wraps the model for DDP
        self.task_type = task_type  # The type of training task (supervised or unsupervised)
        self.print_loss_per_gpu = print_loss_per_gpu

    def _load_snapshot(self, snapshot_path: str):
        """
        Loads a training snapshot to resume training.

        Parameters:
        - snapshot_path: The path to the training snapshot file.
        """
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, inputs, targets):
        """
        Runs a single batch of training data through the model.

        Parameters:
        - inputs: Input data for the model.
        - targets: Target data for the training.

        Returns:
        The loss value for the batch.
        """
        start_time = time()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.metrics[0](outputs, targets)
        if loss.requires_grad:  # Check if loss requires gradients
            loss.backward()

        end_time = time()  # End timing the batch processing
        batch_processing_time = end_time - start_time  # Calculate processing time

        if self.print_loss_per_gpu:  # Check if loss printing is enabled
            # Print loss for the current GPU along with the processing time of the batch
            print(
                f"GPU {self.gpu_id} | Batch Loss: {loss.item():.4f} | Processing Time: {batch_processing_time:.2f} seconds")
        return loss

    def _run_epoch(self, epoch: int):
        """
        Runs a single epoch of training.

        Parameters:
        - epoch: The current epoch number.
        """
        running_loss = 0.0
        running_size = 0
        self.train_data.sampler.set_epoch(epoch)
        for batch in self.train_data:
            if self.task_type == 'supervised':
                inputs, targets = batch
            elif self.task_type == 'reconstruction':
                inputs = batch
                targets = inputs  # For reconstruction tasks, inputs are used as targets
            start_time = time()
            with torch.set_grad_enabled(True):
                inputs, targets = inputs.to(self.gpu_id), targets.to(self.gpu_id)
                loss = self._run_batch(inputs, targets)
            running_loss += loss.item() * batch.size(0)
            running_size += batch.size(0)

        avg_epoch_loss = running_loss / running_size
        if self.gpu_id == 0:
            print(f"Epoch {epoch} | Average Loss: {avg_epoch_loss:.4f}")

    def _validate(self) -> float:
        """
        Validates the model on the test dataset.

        Returns:
        The average validation loss across all test data.
        """
        self.model.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        running_size = 0

        for batch in self.test_data:
            if self.task_type == 'supervised':
                inputs, targets = batch
            elif self.task_type == 'reconstruction':
                inputs = batch
                targets = inputs
            with torch.no_grad():  # No need to track gradients during validation
                inputs, targets = inputs.to(self.gpu_id), targets.to(self.gpu_id)
                loss = self._run_batch(inputs, targets)
            running_loss += loss.item() * batch.size(0)
            running_size += batch.size(0)

        # Convert running loss and size to tensors for all_reduce operation
        running_loss_tensor = torch.tensor([running_loss], device=self.gpu_id)
        running_size_tensor = torch.tensor([running_size], device=self.gpu_id)

        # Use dist.all_reduce to sum the losses and sizes from all GPUs
        dist.all_reduce(running_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_size_tensor, op=dist.ReduceOp.SUM)

        # Compute the average loss across all GPUs and samples
        avg_val_loss = running_loss_tensor.item() / running_size_tensor.item()

        if self.gpu_id == 0:
            print(f"Validation Loss: {avg_val_loss:.4e}")

        return avg_val_loss

    def _save_snapshot(self, epoch: int):
        """
        Saves a training snapshot.

        Parameters:
        - epoch: The current epoch number, for tracking in the snapshot.
        """
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        """
        The main training loop.

        Parameters:
        - max_epochs: The maximum number of epochs to train for.
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


@record
def dist_train(trainer: Trainer, total_epochs: int):
    """
    A utility function to manage the distributed training process.

    Parameters:
    - trainer: The Trainer instance to conduct the training.
    - total_epochs: The total number of epochs to train the model.
    """
    trainer.train(total_epochs)
    destroy_process_group()


