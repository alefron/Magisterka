from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import Callback
import csv

class CustomCSVLogger(Callback):
    def __init__(self, filename, separator=',', append=False):
        super(CustomCSVLogger, self).__init__()
        self.filename = filename
        self.separator = separator
        self.append = append

    def on_train_begin(self, logs=None):
        if not self.append:
            with open(self.filename, 'w', newline='') as csvfile:
                fieldnames = ['epoch', 'accuracy', 'categorical_accuracy', 'top_k_categorical_accuracy', 'loss', 'precision', 'recall', 'val_categorical_accuracy', 'val_top_k_categorical_accuracy']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

    def on_epoch_end(self, epoch, logs=None):
        with open(self.filename, 'a', newline='') as csvfile:
            fieldnames = ['epoch', 'accuracy', 'categorical_accuracy', 'top_k_categorical_accuracy', 'loss', 'precision', 'recall', 'val_categorical_accuracy', 'val_top_k_categorical_accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'epoch': epoch,
                             'accuracy': logs.get('accuracy'),
                             'categorical_accuracy': logs.get('categorical_accuracy'),
                             'top_k_categorical_accuracy': logs.get('top_k_categorical_accuracy'),
                             'loss': logs.get('loss'),
                             'precision': logs.get('precision'),
                             'recall': logs.get('recall'),
                             'val_categorical_accuracy': logs.get('val_categorical_accuracy'),
                             'val_top_k_categorical_accuracy': logs.get('val_top_k_categorical_accuracy')})
