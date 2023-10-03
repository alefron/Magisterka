from datetime import datetime

import tensorflow.python.keras.metrics
from tensorflow.keras.models import load_model
from tensorflow.python.keras import metrics
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import os
import random
from keras.callbacks import ModelCheckpoint, LambdaCallback
from tensorflow.keras.callbacks import LearningRateScheduler
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
                fieldnames = ['epoch', 'accuracy', 'categorical_accuracy', 'top_k_categorical_accuracy', 'loss', 'precision', 'recall']  # Dodaj pozostałe metryki
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

    def on_epoch_end(self, epoch, logs=None):
        with open(self.filename, 'a', newline='') as csvfile:
            fieldnames = ['epoch', 'accuracy', 'categorical_accuracy', 'top_k_categorical_accuracy', 'loss', 'precision', 'recall', 'val_categorical_accuracy', 'val_top_k_categorical_accuracy']  # Dodaj pozostałe metryki
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

def schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

from tensorflow.python.framework.ops import EagerTensor

features_folder = '/Volumes/Czerwony/features/beat_features/'

test_feature = '/Volumes/Czerwony/features/2/116_CQT.json'

# Wczytanie modelu
model = load_model('model_base2.h5')

# Wczytanie pliku CSV
df = pd.read_csv('metadata_filtered_without_bad_labels.csv')
df = df.sample(frac=1).reset_index(drop=True)
#df = df[df['SOURCE'] == 'IA']

annotations_path = 'annotations/parsed_annotations/'

# slownik labeli
labels_dictionary = {
    "Bridge": "Bridge",
    "build": "Bridge",
    "Chorus": "Chorus",
    "Theme": "Chorus",
    "Pre-Chorus": "Chorus",
    "Main-Theme": "Chorus",
    "post-chorus": "Chorus",
    "Solo": "Instrumental",
    "Instrumental": "Instrumental",
    "Recap": "Instrumental",
    "third": "Instrumental",
    "guitar": "Instrumental",
    "Secondary-theme": "Instrumental",
    "variation-2": "Instrumental",
    "gypsy": "Instrumental",
    "variation": "Instrumental",
    "steel": "Instrumental",
    "organ": "Instrumental",
    "hammond": "Instrumental",
    "variation-1": "Instrumental",
    "Interlude": "Interlude",
    "Transition": "Interlude",
    "break": "Interlude",
    "Intro": "Intro",
    "Head": "Intro",
    "pick-up": "Intro",
    "vocals": "Intro",
    "backing": "Intro",
    "no-function": "No-function",
    "voice": "No-function",
    "spoken": "No-function",
    "banjo": "No-function",
    "applause": "No-function",
    "stage-sounds": "No-function",
    "spoken-voice": "No-function",
    "stage-speaking": "No-function",
    "crowd-sounds": "No-function",
    "count-in": "No-function",
    "&pause": "No-function",
    "Outro": "Outro",
    "Coda": "Outro",
    "Fade-out": "Outro",
    "out": "Outro",
    "outro": "Outro",
    "Silence": "Silence",
    "End": "Silence",
    "silence": "Silence",
    "Verse": "Verse",
    "Pre-Verse": "Verse"
}

labels_coding = {
 'Bridge': np.array([1., 0., 0., 0., 0., 0., 0., 0., 0.]),
 'Chorus': np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.]),
 'Instrumental': np.array([0., 0., 1., 0., 0., 0., 0., 0., 0.]),
 'Interlude': np.array([0., 0., 0., 1., 0., 0., 0., 0., 0.]),
 'Intro': np.array([0., 0., 0., 0., 1., 0., 0., 0., 0.]),
 'No-function': np.array([0., 0., 0., 0., 0., 1., 0., 0., 0.]),
 'Outro': np.array([0., 0., 0., 0., 0., 0., 1., 0., 0.]),
 'Silence': np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.]),
 'Verse': np.array([0., 0., 0., 0., 0., 0., 0., 0., 1.])
}

selected_ids = [song_id for song_id in df['SONG_ID'] if os.path.exists(features_folder + f'{song_id}_CQT.json') and os.path.exists(annotations_path + f'{song_id}.json')]
all_ids = pd.Series(selected_ids) if isinstance(df['SONG_ID'], pd.Series) else pd.DataFrame(selected_ids)

def devide_set(all, small_percent):
    all_ids_count = len(all)
    val_count = int(small_percent * all_ids_count)
    train_count = all_ids_count - val_count

    all_ids_copy = all.copy()

    validation_set = all_ids_copy.sample(n=val_count)

    for elem in validation_set:
        all_ids_copy = all_ids_copy[all_ids_copy != elem]

    train_set = all_ids_copy

    return validation_set, train_set


def generator(ids):
    df_filtered = df[df['SONG_ID'].isin(ids)]
    for index, row in df_filtered.iterrows():
        if os.path.exists(features_folder + f"{row['SONG_ID']}_CQT.json") and os.path.exists(features_folder + f"{row['SONG_ID']}_MFCC.json") and os.path.exists(features_folder + f"{row['SONG_ID']}_Tempogram.json"):
            with open(features_folder + f"{row['SONG_ID']}_CQT.json", 'r') as cqt_file:
                cqts = json.load(cqt_file)['Feature']['Vectors']
            with open(features_folder + f"{row['SONG_ID']}_MFCC.json", 'r') as mfcc_file:
                mfccs = json.load(mfcc_file)['Feature']['Vectors']
            with open(features_folder + f"{row['SONG_ID']}_Tempogram.json", 'r') as tempo_file:
                tempograms = json.load(tempo_file)['Feature']['Vectors']

            print("\n")
            print(f"SONG_ID: {row['SONG_ID']}")
            print(f"piosenka: {index + 1} / {len(df_filtered)}")

            beat_times = [np.float64(float(cqt[0])) for cqt in cqts]

            labels = calculate_labels(beat_times, row['SONG_ID'])

            cqts_values = [np.array([np.complex64(complex(float(value['real']), float(value['imag']))) for value in cqt[1]]) for cqt in cqts]
            mfccs_values = [np.array([np.float32(float(value)) for value in mfcc[1]]) for mfcc in mfccs]
            tempograms_values = [np.array([np.float64(float(value)) for value in tempo[1]]) for tempo in tempograms]

            indices = list(range(len(cqts_values)))
            random.shuffle(indices)

            for i in indices:
                if 2 <= i <= (len(indices) - 2):
                    if cqts_values[i].shape == (84,) and mfccs_values[i].shape == (14,) and tempograms_values[i].shape == (192,) and labels[i] is not None and labels[i].shape == (9,):
                        if cqts_values[i+1].shape == (84,) and mfccs_values[i+1].shape == (14,) and tempograms_values[i+1].shape == (192,):
                            if cqts_values[i-2].shape == (84,) and mfccs_values[i-2].shape == (14,) and tempograms_values[i-2].shape == (192,):
                                if cqts_values[i - 1].shape == (84,) and mfccs_values[i - 1].shape == (14,) and tempograms_values[i - 1].shape == (192,):
                                    cqt_result = np.vstack((cqts_values[i-2], cqts_values[i-1], cqts_values[i], cqts_values[i+1])).T
                                    mfcc_result = np.vstack((mfccs_values[i - 2], mfccs_values[i - 1], mfccs_values[i], mfccs_values[i + 1])).T
                                    tempogram_result = np.vstack((tempograms_values[i - 2], tempograms_values[i - 1], tempograms_values[i], tempograms_values[i + 1])).T

                                    cqt_expanded = np.array([[[b] for b in a] for a in cqt_result])
                                    mfcc_expanded = np.array([[[b] for b in a] for a in mfcc_result])
                                    tempogram_expanded = np.array([[[b] for b in a] for a in tempogram_result])

                                    yield { "cqt": cqt_expanded, "mfcc": mfcc_expanded, "tempogram": tempogram_expanded }, labels[i]

            # Po wykorzystaniu danych usuwamy z pamięci
            del cqt_file
            del mfcc_file
            del tempo_file
            del beat_times
            del cqts
            del mfccs
            del tempograms
            del labels
            del cqts_values
            del mfccs_values
            del tempograms_values
            del indices
            del cqt_result
            del mfcc_result
            del tempogram_result
            del cqt_expanded
            del mfcc_expanded
            del tempogram_expanded


def calculate_labels(beat_times, song_id):
    output = []
    for i in range(0, len(beat_times), 1):
        beat_duration = 0.1
        if (i + 1) < len(beat_times):
            beat_duration = beat_times[i+1] - beat_times[i]
        timestamp = beat_times[i] + beat_duration/2
        with open(annotations_path + f"{song_id}.json", 'r') as annotations_file:
            annotations = json.load(annotations_file)

        # Szukanie segmentu, który obejmuje podany moment
        segment_name = None
        for segment in annotations['data']:
            if segment['time'] <= timestamp <= segment['time'] + segment['duration']:
                segment_name = np.array(labels_coding[labels_dictionary[segment['value']]], dtype=float)
                break

        output.append(segment_name)
    return output

def generator_test():
    if os.path.exists(test_feature):
        with open(test_feature) as feature_file:
            feature = json.load(feature_file)
        feature_formatted = [np.array(group).T for group in feature['Feature']['Vectors']]
        feature_formatted2 = [np.array([np.array([np.array([c["real"], c["imag"]]) for c in b]) for b in a]) for a in feature_formatted]

        labels = calculate_labels(len(feature_formatted2), 116)

        indices = list(range(len(feature_formatted2)))
        random.shuffle(indices)

        # Mieszanie obu list w tym samym porządku
        shuffled_feature = [feature_formatted2[i] for i in indices]
        shuffled_labels = [labels[i] for i in indices]

        for i in range(len(shuffled_feature)):
            if shuffled_feature[i].shape == (5, 4, 2):
                return {"cqt": shuffled_feature[i]}, shuffled_labels[i]



val, train = devide_set(all_ids, 0.15)

#res = generator(train)

print(f"train size: {len(train)}")
print(f"val size: {len(val)}")

dataset_train = tf.data.Dataset.from_generator(lambda: generator(train), output_types=({ "cqt": np.ndarray(shape=(84, 4, 1), dtype=np.complex64), "mfcc": np.ndarray(shape=(14, 4, 1), dtype=np.float32), "tempogram": np.ndarray(shape=(192, 4, 1), dtype=np.float64) }, np.ndarray(shape=(1, 9))))
dataset_val = tf.data.Dataset.from_generator(lambda: generator(val), output_types=({ "cqt": np.ndarray(shape=(84, 4, 1), dtype=np.complex64), "mfcc": np.ndarray(shape=(14, 4, 1), dtype=np.float32), "tempogram": np.ndarray(shape=(192, 4, 1), dtype=np.float64) }, np.ndarray(shape=(1, 9))))


#dataset_train = tf.data.Dataset.from_generator(lambda: generator(train), output_types=({"cqt": np.ndarray, "mfcc": np.ndarray, "tempogram": np.ndarray}, tf.Tensor))
#dataset_test = tf.data.Dataset.from_generator(lambda: generator(test), output_types=({"cqt": tf.Tensor, "mfcc": tf.Tensor, "tempogram": tf.Tensor}, tf.Tensor))
#dataset_val = tf.data.Dataset.from_generator(lambda: generator(val), output_types=({"cqt": tf.Tensor, "mfcc": tf.Tensor, "tempogram": tf.Tensor}, tf.Tensor))

dataset_train = dataset_train.batch(3)
dataset_val = dataset_val.batch(3)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model_checkpoint = ModelCheckpoint(filepath=f'checkpoints/model_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}' + '_{epoch:02d}.h5', save_freq='epoch')
lr_schedule = LearningRateScheduler(schedule)
# Definiowanie ścieżki do zapisu pliku CSV
#csv_logger = CSVLogger('training.log', separator=',', append=False)
csv_logger = CustomCSVLogger('training.csv', separator=',', append=False)

#model = load_model('checkpoints/model_2023-09-29_19-17-51_17.h5')
#result = model.evaluate(dataset_test)

# Trenowanie modelu
model.fit(dataset_train, validation_data=dataset_val, epochs=100, callbacks=[early_stopping, model_checkpoint, csv_logger])

# Zapisuje zarówno architekturę, jak i wagi modelu
model.save(f'models/model_base_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.h5')




