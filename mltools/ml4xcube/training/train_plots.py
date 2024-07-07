import matplotlib.pyplot as plt
from typing import List, Tuple


def plot_loss(train_loss: List[float], val_loss: List[float], figsize: Tuple[int, int] = (4, 2)) -> None:
    """
    Plot the training and validation loss over epochs.

    Args:
        train_loss (List[float]): List of training loss values for each epoch.
        val_loss (List[float]): List of validation loss values for each epoch.
        figsize (Tuple[int, int], optional): Size of the figure. Defaults to (4, 2).

    Returns:
        None
    """
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=figsize)  # Reduced image size
    plt.plot(epochs, train_loss, 'r', label='Training loss')  # Changed color to red
    plt.plot(epochs, val_loss, 'y', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    # Ensure that x-axis ticks are only integers
    plt.xticks(ticks=range(1, len(train_loss) + 1, 2))

    plt.show()