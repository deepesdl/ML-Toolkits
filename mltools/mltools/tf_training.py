import tensorflow as tf
from typing import Union

class Trainer:

    """
    A trainer class for training TensorFlow models on single or no GPU systems.
    """
    def __init__(
            self,
            model: tf.keras.Model,
            train_data: tf.data.Dataset,
            test_data: tf.data.Dataset,
            optimizer: Union[tf.keras.optimizers.Optimizer, str],
            best_model_path: str,
            learning_rate: float = 0.001,
            loss: Union[tf.keras.losses.Loss, str] = "mean_squared_error",
            early_stopping: bool = True,
            patience: int = 10,
            metrics: list = ['mae'],
            tf_log_dir: str = './logs',
            mlflow_run=None,
            summery: bool = False,
            epochs: int = 100,
            batch_size: int = 32,
    ) -> None:
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.best_model_path = best_model_path
        self.loss = loss
        self.max_epochs = epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.metrics = metrics
        self.tf_log_dir = tf_log_dir
        self.mlflow_run = mlflow_run
        self.summery = summery

    def train(self):

        """
        The main training loop with specific steps per epoch for training and validation.
        """
        steps_per_train_epoch = len(self.train_data) // self.batch_size
        steps_per_validation_epoch = len(self.test_data) // self.batch_size

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

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        if self.summery: self.model.summary()

        self.model.optimizer.learning_rate.assign(self.learning_rate)

        self.model.fit(
            self.train_data.batch(self.batch_size),
            epochs=self.max_epochs,
            steps_per_epoch=steps_per_train_epoch,
            validation_data=self.test_data.batch(self.batch_size),
            validation_steps=steps_per_validation_epoch,
            callbacks=callbacks
        )

        if self.mlflow_run:
            self.mlflow_run.log_artifact(self.best_model_path, "model")

        return self.model


