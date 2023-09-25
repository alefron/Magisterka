from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import os
import random
from keras.callbacks import ModelCheckpoint, LambdaCallback

from tensorflow.python.framework.ops import EagerTensor

features_folder = 'features_sample/'

test_feature = 'test_feature.json'

# Wczytanie modelu
model = load_model('model_base.h5')

# Wczytanie pliku CSV
df = pd.read_csv('metadata_filtered_without_bad_labels.csv')

annotations_path = 'annotations/parsed_annotations/'

# ilośc milisekund na 1 wektor
vector_duration = 61.9179

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
    "applause": "Noises",
    "stage-sounds": "Noises",
    "spoken-voice": "Noises",
    "stage-speaking": "Noises",
    "crowd-sounds": "Noises",
    "count-in": "Noises",
    "&pause": "Noises",
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
 'Bridge': np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
 'Chorus': np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]),
 'Instrumental': np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]),
 'Interlude': np.array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]),
 'Intro': np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]),
 'No-function': np.array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
 'Noises': np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]),
 'Outro': np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]),
 'Silence': np.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]),
 'Verse': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
}


all_ids = df['SONG_ID']

def devide_set(all, small_percent):
    all_ids_count = len(all)
    val_count = int(small_percent * all_ids_count)
    train_count = all_ids_count - val_count

    all_ids_copy = all.copy()

    validation_set = all_ids_copy.sample(n=val_count, random_state=42)

    for elem in validation_set:
        all_ids_copy = all_ids_copy[all_ids_copy != elem]

    train_set = all_ids_copy

    return validation_set, train_set


def generator(ids):
    df_filtered = df[df['SONG_ID'].isin(ids)]
    for index, row in df_filtered.iterrows():
        if os.path.exists(features_folder + f"{row['SONG_ID']}_CQT.json") and os.path.exists(features_folder + f"{row['SONG_ID']}_MFCC.json") and os.path.exists(features_folder + f"{row['SONG_ID']}_Tempogram.json"):
            with open(features_folder + f"{row['SONG_ID']}_CQT.json", 'r') as cqt_file:
                cqt = json.load(cqt_file)
            with open(features_folder + f"{row['SONG_ID']}_MFCC.json", 'r') as mfcc_file:
                mfcc = json.load(mfcc_file)
            with open(features_folder + f"{row['SONG_ID']}_Tempogram.json", 'r') as tempo_file:
                tempogram = json.load(tempo_file)

            cqt_formatted = [np.array(group).T for group in cqt['Feature']['Vectors']]
            mfcc_formatted = [np.array(group).T for group in mfcc['Feature']['Vectors']]
            tempogram_formatted = [np.array(group).T for group in tempogram['Feature']['Vectors']]

            cqt_formatted2 = [np.array([np.array([np.array([c["real"], c["imag"]]) for c in b]) for b in a]) for a in cqt_formatted]
            mfcc_formatted2 = [np.array([np.array([np.array([c["real"], c["imag"]]) for c in b]) for b in a]) for a in mfcc_formatted]
            tempogram_formatted2 = [np.array([np.array([np.array([c["real"], c["imag"]]) for c in b]) for b in a]) for a in tempogram_formatted]

            labels = calculate_labels(len(cqt_formatted), row['SONG_ID'])

            indices = list(range(len(mfcc_formatted2)))
            random.shuffle(indices)

            # Mieszanie obu list w tym samym porządku
            shuffled_cqt = [cqt_formatted2[i] for i in indices]
            shuffled_mfcc = [mfcc_formatted2[i] for i in indices]
            shuffled_tempogram = [tempogram_formatted2[i] for i in indices]
            shuffled_labels = [labels[i] for i in indices]

            for i in range(len(shuffled_mfcc)):
                if shuffled_cqt[i].shape == (84, 4, 2) and shuffled_mfcc[i].shape == (14, 4, 2) and shuffled_tempogram[i].shape == (192, 4, 2):
                    yield {"cqt": shuffled_cqt[i], "mfcc": shuffled_mfcc[i], "tempogram": shuffled_tempogram[i]}, shuffled_labels[i]

            # Po wykorzystaniu danych usuwamy z pamięci
            del cqt
            del mfcc
            del tempogram
            del labels
            del cqt_formatted
            del cqt_formatted2
            del mfcc_formatted
            del mfcc_formatted2
            del tempogram_formatted
            del tempogram_formatted2
            del cqt_file
            del mfcc_file
            del tempo_file


def calculate_labels(output_vector_length, song_id):
    output = []
    for i in range(0, output_vector_length, 1):
        timestamp = i * vector_duration + 2 * vector_duration + 0.5 * vector_duration
        with open(annotations_path + f"{song_id}.json", 'r') as annotations_file:
            annotations = json.load(annotations_file)
        timestamp_in_sec = timestamp / 1000

        # Szukanie segmentu, który obejmuje podany moment
        segment_name = None

        for segment in annotations['data']:
            if segment['time'] <= timestamp_in_sec <= segment['time'] + segment['duration']:
                segment_name = np.array(labels_coding[labels_dictionary[segment['value']]], dtype=float)
                #tensor = tf.convert_to_tensor(segment_name, dtype=float)
                break

        output.append(segment_name)
    #output_array = np.array(output)
    return output

def generator_test():
    if os.path.exists(test_feature):
        with open(test_feature) as feature_file:
            feature = json.load(feature_file)
        feature_formatted = [np.array(group).T for group in feature['Feature']['Vectors']]
        feature_formatted2 = [np.array([np.array([np.array([c["real"], c["imag"]]) for c in b]) for b in a]) for a in feature_formatted]

        labels = calculate_labels(len(feature_formatted2), 10)

        indices = list(range(len(feature_formatted2)))
        random.shuffle(indices)

        # Mieszanie obu list w tym samym porządku
        shuffled_feature = [feature_formatted2[i] for i in indices]
        shuffled_labels = [labels[i] for i in indices]

        for i in range(len(shuffled_feature)):
            if shuffled_feature[i].shape == (5, 4, 2):
                return {"cqt": shuffled_feature[i]}, shuffled_labels[i]

#res = generator_test()

test_val, train = devide_set(all_ids, 0.3)
test, val = devide_set(test_val, 0.5)

dataset_train = tf.data.Dataset.from_generator(generator, args=train, output_types=({ "cqt": np.ndarray(shape=(84, 4, 2)), "mfcc": np.ndarray(shape=(14, 4, 2)), "tempogram": np.ndarray(shape=(192, 4, 2)) }, np.ndarray(shape=(1, 10))))
dataset_test = tf.data.Dataset.from_generator(generator, args=test, output_types=({ "cqt": np.ndarray(shape=(84, 4, 2)), "mfcc": np.ndarray(shape=(14, 4, 2)), "tempogram": np.ndarray(shape=(192, 4, 2)) }, np.ndarray(shape=(1, 10))))
dataset_val = tf.data.Dataset.from_generator(generator, args=val, output_types=({ "cqt": np.ndarray(shape=(84, 4, 2)), "mfcc": np.ndarray(shape=(14, 4, 2)), "tempogram": np.ndarray(shape=(192, 4, 2)) }, np.ndarray(shape=(1, 10))))

dataset_train = dataset_train.batch(3)
dataset_test = dataset_test.batch(3)
dataset_val = dataset_val.batch(3)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model_checkpoint = ModelCheckpoint(filepath='model_{epoch:02d}.h5', save_freq='epoch')
print_epochs = LambdaCallback(on_epoch_end=lambda epoch, logs: print(f'Epoch {epoch}/{logs["epochs"]}'))

# Trenowanie modelu
model.fit(dataset_train, validation_data=dataset_val, epochs=10, callbacks=[early_stopping, model_checkpoint, print_epochs])



