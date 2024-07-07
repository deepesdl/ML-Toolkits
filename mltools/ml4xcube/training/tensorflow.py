import os
import tensorflow as tf
from ml4xcube.training.train_plots import plot_loss

class Trainer:

    """
    A trainer class for training TensorFlow models on single or no GPU systems.
    """
    def __init__(
            self,
            model: tf.keras.Model,
            train_data: tf.data.Dataset,
            test_data: tf.data.Dataset,
            best_model_path: str,
            early_stopping: bool = True,
            patience: int = 10,
            tf_log_dir: str = './logs',
            mlflow_run = None,
            epochs: int = 100,
            train_epoch_steps: int = None,
            val_epoch_steps: int = None,
            create_loss_plot: bool = False,
    ):
        """
        Initialize a Trainer object.

        Attributes:
            model (tf.keras.Model): The TensorFlow model to be trained.
            train_data (tf.data.Dataset): The dataset for training.
            test_data (tf.data.Dataset): The dataset for validation.
            best_model_path (str): Path to save the best model.
            early_stopping (bool): Whether to use early stopping. Defaults to True.
            patience (int): Number of epochs with no improvement after which training will be stopped. Defaults to 10.
            tf_log_dir (str): Directory to save TensorBoard logs. Defaults to './logs'.
            mlflow_run: MLflow run object for logging artifacts. Defaults to None.
            epochs (int): Number of epochs to train the model. Defaults to 100.
            train_epoch_steps (int): Number of steps per training epoch. Defaults to None.
            val_epoch_steps (int): Number of steps per validation epoch. Defaults to None.
            create_loss_plot (bool): Whether to create a plot of training and validation loss. Defaults to False.
        """
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.best_model_path = best_model_path
        self.max_epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.tf_log_dir = tf_log_dir
        self.mlflow_run = mlflow_run
        self.steps_per_train_epoch = train_epoch_steps
        self.steps_per_validation_epoch = val_epoch_steps
        self.model_name = os.path.basename(os.path.normpath(self.best_model_path))
        self.create_loss_plot = create_loss_plot

    def train(self) -> tf.keras.Model:
        """
        Train the TensorFlow model.

        The main training loop with specific steps per epoch for training and validation.

        Returns:
            tf.keras.Model: The trained TensorFlow model.
        """
        if self.steps_per_train_epoch is None:
            for batch in self.train_data.take(1):
                batch_size = batch[0].shape[0]
            self.steps_per_train_epoch = len(self.train_data) // batch_size

        if self.steps_per_validation_epoch is None:
            for batch in self.test_data.take(1):
                batch_size = batch[0].shape[0]
            self.steps_per_validation_epoch = len(self.test_data) // batch_size

        callbacks = [tf.keras.callbacks.TensorBoard(log_dir=self.tf_log_dir, histogram_freq=1)]

        if self.early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                verbose=1,
                mode='min',
                restore_best_weights=True
            ))

        if self.best_model_path:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                self.best_model_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            ))

        history = self.model.fit(
            self.train_data,
            epochs=self.max_epochs,
            steps_per_epoch=self.steps_per_train_epoch,
            validation_data=self.test_data,
            validation_steps=self.steps_per_validation_epoch,
            callbacks=callbacks
        )

        if self.mlflow_run:
            self.mlflow_run.log_artifact(self.best_model_path, self.model_name)

        if self.create_loss_plot:
            train_loss = history.history['loss']
            val_loss = history.history['val_loss']
            plot_loss(train_loss, val_loss)

        return self.model

