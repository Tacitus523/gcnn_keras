import tensorflow as tf

class LearningRateLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        current_lr = optimizer.lr.numpy().item() # get the current learning rate, JSON serializable
        if logs is not None:
            logs["lr"] = current_lr