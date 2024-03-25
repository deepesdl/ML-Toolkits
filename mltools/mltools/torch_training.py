import numpy as np
import torch


def train_one_epoch(epoch_index: int, training_loader: torch.utils.data.DataLoader, model: torch.nn.Module, loss_fn: torch.nn.modules.loss, optimizer: torch.optim, device:str):
    """pytorch model training, training of one epoch"""
    running_loss = 0.
    last_loss = 0.
    train_pred = np.empty(0)
    for i, data in enumerate(training_loader):
        inputs, labels = data
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        X = inputs.unsqueeze(1)
        X = X.to(device)
        model = model.to(device)
        outputs = model(X)
        train_pred = np.append(train_pred, outputs.cpu().detach().numpy().ravel())

        l = labels.unsqueeze(1)
        l = l.to(device)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, l)

        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            running_loss = 0.

    return model, train_pred, last_loss


def test(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, loss_fn: torch.nn.modules.loss, device: str):
    """pytorch model testing"""
    test_pred = np.empty(0)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.unsqueeze(1)
            y = y.unsqueeze(1)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_pred= np.append(test_pred, pred.cpu().detach().numpy().ravel())
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    return test_pred, test_loss
