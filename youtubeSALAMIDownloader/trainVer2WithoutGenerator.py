from datetime import datetime
from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import os
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from customLogger import CustomCSVLogger
from analyzeDataset import analyzeDataset

features_train_folder = '/Volumes/Czerwony/features/beat_features_train/'
features_val_folder = '/Volumes/Czerwony/features/beat_features_val/'
features_test_folder = '/Volumes/Czerwony/features/beat_features_test/'

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

test_feature = '/Volumes/Czerwony/features/2/116_CQT.json'

# Wczytanie modelu
model = load_model('model_base3.h5')

annotations_path = 'annotations/parsed_annotations/'

# Wczytanie pliku CSV
df = pd.read_csv('metadata_filtered_without_bad_labels.csv')
#df = df[df['SOURCE'] == 'IA']
train_ids = [song_id for song_id in df['SONG_ID'] if os.path.exists(features_train_folder + f'{song_id}_CQT.json') and os.path.exists(annotations_path + f'{song_id}.json')]
df_train = df[df['SONG_ID'].isin(train_ids)]



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

def generateDataset(ids, features_folder):
    cqtData = []
    mfccData = []
    tempogramData = []
    labelsData = []
    df_filtered = df[df['SONG_ID'].isin(ids)]
    for index, row in df_filtered.iterrows():
        if os.path.exists(features_folder + f"{row['SONG_ID']}_CQT.json") and os.path.exists(features_folder + f"{row['SONG_ID']}_MFCC.json") and os.path.exists(features_folder + f"{row['SONG_ID']}_Tempogram.json"):
            with open(features_folder + f"{row['SONG_ID']}_CQT.json", 'r') as cqt_file:
                cqts = json.load(cqt_file)['Feature']['Vectors']
            with open(features_folder + f"{row['SONG_ID']}_MFCC.json", 'r') as mfcc_file:
                mfccs = json.load(mfcc_file)['Feature']['Vectors']
            with open(features_folder + f"{row['SONG_ID']}_Tempogram.json", 'r') as tempo_file:
                tempograms = json.load(tempo_file)['Feature']['Vectors']

            beat_times = [np.float64(float(cqt[0])) for cqt in cqts]

            labels = calculate_labels(beat_times, row['SONG_ID'])

            cqts_values = [np.array([np.complex64(complex(float(value['real']), float(value['imag']))) for value in cqt[1]]) for cqt in cqts]
            mfccs_values = [np.array([np.float32(float(value)) for value in mfcc[1]]) for mfcc in mfccs]
            tempograms_values = [np.array([np.float64(float(value)) for value in tempo[1]]) for tempo in tempograms]

            for i, value in enumerate(cqts_values):
                if 2 <= i <= (len(cqts_values) - 2):
                    if cqts_values[i].shape == (84,) and mfccs_values[i].shape == (14,) and tempograms_values[i].shape == (192,) and labels[i] is not None and labels[i].shape == (9,):
                        if cqts_values[i+1].shape == (84,) and mfccs_values[i+1].shape == (14,) and tempograms_values[i+1].shape == (192,):
                            if cqts_values[i-2].shape == (84,) and mfccs_values[i-2].shape == (14,) and tempograms_values[i-2].shape == (192,):
                                if cqts_values[i - 1].shape == (84,) and mfccs_values[i - 1].shape == (14,) and tempograms_values[i - 1].shape == (192,):
                                    cqt_result = np.vstack((cqts_values[i-2], cqts_values[i-1], cqts_values[i], cqts_values[i+1])).T
                                    mfcc_result = np.vstack((mfccs_values[i - 2], mfccs_values[i - 1], mfccs_values[i], mfccs_values[i + 1])).T
                                    tempogram_result = np.vstack((tempograms_values[i - 2], tempograms_values[i - 1], tempograms_values[i], tempograms_values[i + 1])).T

                                    cqt_no_complex = np.array([[[float(compl.real), float(compl.imag)] for compl in four] for four in cqt_result])
                                    cqt_expanded = np.array([[b for b in a] for a in cqt_no_complex])
                                    mfcc_expanded = np.array([[[b] for b in a] for a in mfcc_result])
                                    tempogram_expanded = np.array([[[b] for b in a] for a in tempogram_result])

                                    cqtData.append(cqt_expanded)
                                    mfccData.append(mfcc_expanded)
                                    tempogramData.append(tempogram_expanded)
                                    labelsData.append(labels[i])

    return np.array(cqtData), np.array(mfccData), np.array(tempogramData), np.array(labelsData)


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

#trenowanie po kawałku w pętli
step = 200
epochs = 100

print('Generating validation data...')

val_ids = [song_id for song_id in df['SONG_ID'] if os.path.exists(features_val_folder + f'{song_id}_CQT.json') and os.path.exists(annotations_path + f'{song_id}.json')]
val_ids_series = pd.Series(val_ids) if isinstance(df['SONG_ID'], pd.Series) else pd.DataFrame(val_ids)

cqt_val, mfcc_val, tempogram_val, labels_val = generateDataset(val_ids_series, features_val_folder)
val_set_stats = analyzeDataset(labels_val)
for word, stats in val_set_stats.items():
    print(f"Word: {word}, Count: {stats['count']}, Percentage: {stats['percentage']}%")

entire_train_stats = {
 'Bridge': 0,
 'Chorus': 0,
 'Instrumental': 0,
 'Interlude': 0,
 'Intro': 0,
 'No-function': 0,
 'Outro': 0,
 'Silence': 0,
 'Verse': 0
}

for epoch_number in range(1, epochs, 1):
    print(f'EPOCH: {epoch_number}:')
    historical_val_loss = []
    history = {}
    for i in range(0, df_train.shape[0], step):
        low_index = i
        high_index = i + step
        if high_index >= df_train.shape[0]:
            high_index = df_train.shape[0] - 1

        print(f'EPOCH: {epoch_number}, Songs range: {low_index}-{high_index}')

        df_part = df_train[low_index:high_index]
        part_ids = [song_id for song_id in df_part['SONG_ID'] if os.path.exists(features_train_folder + f'{song_id}_CQT.json') and os.path.exists(annotations_path + f'{song_id}.json')]
        ids_series = pd.Series(part_ids) if isinstance(df_part['SONG_ID'], pd.Series) else pd.DataFrame(part_ids)

        print('Generating training data...')
        cqt, mfcc, tempogram, labels_data = generateDataset(ids_series, features_train_folder)
        train_set_stats = analyzeDataset(labels_data)
        for word, stats in train_set_stats.items():
            print(f"Word: {word}, Count: {stats['count']}, Percentage: {stats['percentage']}%")
            if epoch_number == 1:
                entire_train_stats[word] = entire_train_stats[word] + stats['count']

        history = model.fit(
            [cqt, tempogram],
            labels_data,
            epochs=1,
            validation_data=([cqt_val, tempogram_val], labels_val),
            shuffle=True,
            batch_size=100
        )
        del cqt
        del mfcc
        del tempogram
        del df_part

    model.save(f'checkpoints/{now}_epoch_{epoch_number}.h5')

    last_epoch_metrics = {metric: values[-1] for metric, values in history.history.items()}
    df_metrics = pd.DataFrame(last_epoch_metrics, index=[0])

    if not os.path.exists(f'training_{now}.csv'):
        # Tworzymy nowy DataFrame z nagłówkami
        df_header = pd.DataFrame(columns=['loss', 'accuracy', 'categorical_accuracy',
                                          'top_k_categorical_accuracy', 'precision',
                                          'recall', 'val_loss', 'val_accuracy', 'val_categorical_accuracy',
                                          'val_top_k_categorical_accuracy', 'val_precision', 'val_recall'])
        # Zapisujemy do pliku
        df_header.to_csv(f'training_{now}.csv', index=False)

    df_metrics.to_csv(f'training_{now}.csv', mode='a', header=False, index=False)
    print(f'Epoch {epoch_number} saved to csv')

    if historical_val_loss.count == 3:
        if historical_val_loss[0] < historical_val_loss[1] < historical_val_loss[2]:
            print(f'early stopping after {epoch_number} epoch')
            #break
        historical_val_loss[0] = historical_val_loss[1]
        historical_val_loss[1] = historical_val_loss[2]
        historical_val_loss[2] = float(df_metrics['val_loss'])
    else:
        historical_val_loss.append(df_metrics['val_loss'])



