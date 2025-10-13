import tensorflow as tf
import time
from datetime import timedelta

class LearningRateLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        current_lr = optimizer.lr.numpy().item() # get the current learning rate, JSON serializable
        if logs is not None:
            logs["lr"] = current_lr

class TrainingTimeCallback(tf.keras.callbacks.Callback):
    """Custom callback to track training time and add it to history."""
    
    def __init__(self):
        super().__init__()
        self.start_time = None
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.start_process_time = time.process_time()
        
    def on_train_end(self, logs=None):
        if self.start_time is not None:
            training_time = timedelta(seconds=time.time() - self.start_time)
            training_process_time = timedelta(seconds=time.process_time() - self.start_process_time)
            # Add training time to history
            if hasattr(self.model, 'history'):
                if not hasattr(self.model.history, 'history'):
                    self.model.history.history = {}
                self.model.history.history['training_time'] = training_time