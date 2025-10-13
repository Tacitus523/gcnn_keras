import tensorflow as tf
import time

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

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = (time.time() - self.start_time) / 60  # in minutes
        elapsed_process_time = (time.process_time() - self.start_process_time) / 60  # in minutes
        if logs is not None:
            logs['elapsed_time'] = elapsed_time
            logs['elapsed_process_time'] = elapsed_process_time
        
    def on_train_end(self, logs=None):
        training_time = (time.time() - self.start_time) / 60  # in minutes
        training_process_time = (time.process_time() - self.start_process_time) / 60  # in minutes
        # Add training time to history
        if hasattr(self.model, 'history'):
            if not hasattr(self.model.history, 'history'):
                self.model.history.history = {}
            self.model.history.history['total_training_time'] = [training_time]*len(self.model.history.epoch)
            self.model.history.history['total_training_process_time'] = [training_process_time]*len(self.model.history.epoch)