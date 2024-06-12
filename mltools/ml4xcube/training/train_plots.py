import matplotlib.pyplot as plt


def plot_loss(train_loss, val_loss, figsize=(4, 2)):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=figsize)  # Reduced image size
    plt.plot(epochs, train_loss, 'r', label='Training loss')  # Changed color to red
    plt.plot(epochs, val_loss, 'y', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    # Ensure that x-axis ticks are only integers
    plt.xticks(ticks=range(1, len(train_loss) + 1, 2))

    plt.show()