from typing import Callable
from torch.utils.data import Dataset, DataLoader
from time import time
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import torch
import os


def ddp_init():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def prepare_dataloader(dataset: Dataset, batch_size: int, callback_fn: Callable, num_workers: int = 0):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=callback_fn,
        sampler=DistributedSampler(dataset)
    )


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            test_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            save_every: int,
            snapshot_path: str,
            early_stopping: bool = True,
            patience: int = 10,
            metrics: list = None,
            task_type: str = "supervised"
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.early_stopping = early_stopping
        self.patience = patience
        self.best_val_loss = float('inf')
        self.strikes = 0
        self.metrics = metrics if metrics is not None else [torch.nn.MSELoss(reduction='mean')]
        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)
        self.task_type = task_type

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, inputs, targets):
        start_batch = time()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.metrics[0](outputs, targets)
        if loss.requires_grad:  # Check if loss requires gradients
            loss.backward()
        return loss

    def _run_epoch(self, epoch, ):
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
            running_loss += loss.item() * batch.size(0) * batch.size(1)
            running_size += batch.size(0) * batch.size(1)

        avg_epoch_loss = running_loss / running_size
        if self.gpu_id == 0:
            print(f"Epoch {epoch} | Average Loss: {avg_epoch_loss:.4f}")

    def _validate(self):
        self.model.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        running_size = 0

        for batch in self.test_data:
            with torch.no_grad():  # No need to track gradients during validation
                loss = self._run_batch(batch)
            running_loss += loss.item() * batch.size(0) * batch.size(1)
            running_size += batch.size(0) * batch.size(1)

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


    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            dist.barrier()

            epoch_val_loss = self._validate()

            # Early Stopping Logic
            if epoch_val_loss < self.best_val_loss:
                self.strikes = 0
                self.best_val_loss = epoch_val_loss
                # Saving the best model
                best_model_path = 'Best_Model.pt'
                torch.save(self.model.module.state_dict(), best_model_path)
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
def dist_train(trainer, total_epochs):
    trainer.train(total_epochs)
    destroy_process_group()


