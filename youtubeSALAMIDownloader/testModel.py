from sklearn.metrics import classification_report
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import os
import csv
import json
from analyzeDataset import analyzeDataset

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

                                    cqt_expanded = np.array([[[b] for b in a] for a in cqt_result])
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



features_train_folder = '/Volumes/Czerwony/features/beat_features_train/'
features_val_folder = '/Volumes/Czerwony/features/beat_features_val/'
features_test_folder = '/Volumes/Czerwony/features/beat_features_test/'
annotations_path = 'annotations/parsed_annotations/'

df = pd.read_csv('metadata_filtered_without_bad_labels.csv')[0:230]
val_ids = [song_id for song_id in df['SONG_ID'] if os.path.exists(features_val_folder + f'{song_id}_CQT.json') and os.path.exists(annotations_path + f'{song_id}.json')]
df_val = df[df['SONG_ID'].isin(val_ids)]
val_ids_series = pd.Series(val_ids) if isinstance(df_val['SONG_ID'], pd.Series) else pd.DataFrame(val_ids)


train_ids = [song_id for song_id in df['SONG_ID'] if os.path.exists(features_train_folder + f'{song_id}_CQT.json') and os.path.exists(annotations_path + f'{song_id}.json')]
df_train = df[df['SONG_ID'].isin(train_ids)]
train_ids_series = pd.Series(train_ids) if isinstance(df_train['SONG_ID'], pd.Series) else pd.DataFrame(train_ids)

model = load_model('checkpoints/2023-10-03_23-00-51_epoch_8.h5')
cqt, mfcc, tempogram, labels = generateDataset(train_ids_series, features_train_folder)
stats_dataset = analyzeDataset(labels)
for word, stats in stats_dataset.items():
    print(f"Word: {word}, Count: {stats['count']}, Percentage: {stats['percentage']}%")

predictions = model.predict([cqt, mfcc, tempogram])
y_test = np.argmax(labels, axis=1)
y_pred = np.argmax(predictions, axis=1)
print(classification_report(y_test, y_pred))
print('DONE')

