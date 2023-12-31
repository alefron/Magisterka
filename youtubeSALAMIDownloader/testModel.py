from sklearn.metrics import classification_report
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import os
import csv
import json
from analyzeDataset import analyzeDataset
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# slownik labeli
labels_dictionary = {
    "Bridge": "Bridge",
    "build": "Bridge",
    "Chorus": "Chorus",
    "Theme": "Solo",
    "Pre-Chorus": "Chorus",
    "Main-Theme": "Solo",
    "post-chorus": "Chorus",
    "Solo": "Solo",
    "Instrumental": "Solo",
    "Recap": "Solo",
    "third": "Solo",
    "guitar": "Solo",
    "Secondary-theme": "Solo",
    "variation-2": "Solo",
    "gypsy": "Solo",
    "variation": "Solo",
    "steel": "Solo",
    "organ": "Solo",
    "hammond": "Solo",
    "variation-1": "Solo",
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
    "banjo": "Solo",
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
    "Silence": "No-function",
    "End": "No-function",
    "silence": "No-function",
    "Verse": "Verse",
    "Pre-Verse": "Verse"
}

labels_coding = {
 'Bridge': np.array([1., 0., 0., 0., 0., 0., 0., 0.]),
 'Chorus': np.array([0., 1., 0., 0., 0., 0., 0., 0.]),
 'Solo': np.array([0., 0., 1., 0., 0., 0., 0., 0.]),
 'Interlude': np.array([0., 0., 0., 1., 0., 0., 0., 0.]),
 'Intro': np.array([0., 0., 0., 0., 1., 0., 0., 0.]),
 'No-function': np.array([0., 0., 0., 0., 0., 1., 0., 0.]),
 'Outro': np.array([0., 0., 0., 0., 0., 0., 1., 0.]),
 'Verse': np.array([0., 0., 0., 0., 0., 0., 0., 1.])
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

            for i in range(0, len(cqts_values), 2):
                if 3 <= i <= (len(cqts_values) - 5):
                    if cqts_values[i].shape == (84,) and tempograms_values[i].shape == (192,) and labels[i] is not None and labels[i].shape == (8,):
                        if cqts_values[i + 1].shape == (84,) and tempograms_values[i + 1].shape == (192,):
                            if cqts_values[i + 2].shape == (84,) and tempograms_values[i + 2].shape == (192,):
                                if cqts_values[i + 3].shape == (84,) and tempograms_values[i + 3].shape == (192,):
                                    if cqts_values[i + 4].shape == (84,) and tempograms_values[i + 4].shape == (192,):
                                        if cqts_values[i - 2].shape == (84,) and tempograms_values[i - 2].shape == (192,):
                                            if cqts_values[i - 1].shape == (84,) and tempograms_values[i - 1].shape == (192,):
                                                if cqts_values[i - 3].shape == (84,) and tempograms_values[i - 3].shape == (192,):
                                                    cqt_result = np.vstack((cqts_values[i - 3], cqts_values[i - 2],
                                                                            cqts_values[i - 1], cqts_values[i],
                                                                            cqts_values[i + 1], cqts_values[i + 2],
                                                                            cqts_values[i + 3], cqts_values[i + 4])).T
                                                    # mfcc_result = np.vstack((mfccs_values[i - 2], mfccs_values[i - 1], mfccs_values[i], mfccs_values[i + 1])).T
                                                    tempogram_result = np.vstack((tempograms_values[i - 3],
                                                                                  tempograms_values[i - 2],
                                                                                  tempograms_values[i - 1],
                                                                                  tempograms_values[i],
                                                                                  tempograms_values[i + 1],
                                                                                  tempograms_values[i + 2],
                                                                                  tempograms_values[i + 3],
                                                                                  tempograms_values[i + 4])).T

                                                    cqt_no_complex = np.array([[[float(compl.real), float(compl.imag)] for compl in four] for four in cqt_result])
                                                    cqt_expanded = np.array([[b for b in a] for a in cqt_no_complex])
                                                    # mfcc_expanded = np.array([[[b] for b in a] for a in mfcc_result])
                                                    tempogram_expanded = np.array([[[b] for b in a] for a in tempogram_result])

                                                    cqtData.append([cqt_expanded, row['SONG_ID'], beat_times[i]])
                                                    # mfccData.append([mfcc_expanded, row['SONG_ID'], beat_times[i]])
                                                    tempogramData.append([tempogram_expanded, row['SONG_ID'], beat_times[i]])
                                                    from collections import Counter

                                                    # Twoja lista
                                                    candidates = labels[i - 3:i + 5]
                                                    candidates_flatten = [element.tolist() for element in candidates]

                                                    # Liczenie wystąpień
                                                    counter = Counter(map(tuple, candidates_flatten))

                                                    # Znalezienie elementu z największą liczbą wystąpień
                                                    result_label = max(counter, key=counter.get)
                                                    result_label = np.array(result_label)
                                                    labelsData.append([result_label, row['SONG_ID']])
return cqtData, mfccData, tempogramData, labelsData


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

def getSongTemplate(song_labels, song_beats, song_pred_labels=[]):
    template = []
    if len(song_labels) == len(song_beats):
        from_idx = 0
        previous_label = song_labels[0]
        for index, el in enumerate(song_labels):
            if np.array_equal(song_labels[index][0], previous_label[0]):
                previous_label = song_labels[index]
                continue
            else:
                to_idx = index
                label_value = []
                if len(song_pred_labels) == 0:
                    label_value = song_labels[index - 1]
                else:
                    labels_for_segment = song_pred_labels[from_idx:to_idx]
                    labels_for_segment = [lab[0] for lab in labels_for_segment]

                    arrays_counter = Counter(tuple(array) for array in labels_for_segment)
                    label_value = [np.array(max(arrays_counter, key=arrays_counter.get)), -1]

                label_value_name = ''
                for word, array in labels_coding.items():
                    if np.array_equal(array, label_value[0]):
                        label_value_name = word
                        break

                segment = {
                    "from_idx": from_idx,
                    "to_idx": to_idx,
                    "from_time": song_beats[from_idx],
                    "to_time": song_beats[to_idx],
                    "true_label": label_value[0],
                    "true_label_name": label_value_name
                }
                template.append(segment)
                from_idx = index
                previous_label = song_labels[index]
        return template

def convertPredToOneHot(prediction_labels, song_id):
    output = []
    for pred in prediction_labels:
        max = np.max(pred)
        for i, val in enumerate(pred):
            if val == max:
                pred[i] = 1.
            else:
                pred[i] = 0.
        output.append([pred, song_id])
    return output


features_train_folder = '/Volumes/Czerwony/features/beat_features_train/'
features_val_folder = '/Users/aleksandrafront/Desktop/g/'
features_test_folder = '/Volumes/Czerwony/features/beat_features_test/'
annotations_path = 'annotations/parsed_annotations/'

df = pd.read_csv('metadata_filtered_without_bad_labels.csv')
val_ids = [song_id for song_id in df['SONG_ID'] if os.path.exists(features_test_folder + f'{song_id}_CQT.json') and os.path.exists(annotations_path + f'{song_id}.json')]
df_val = df[df['SONG_ID'].isin(val_ids)]
val_ids_series = pd.Series(val_ids) if isinstance(df_val['SONG_ID'], pd.Series) else pd.DataFrame(val_ids)


train_ids = [song_id for song_id in df['SONG_ID'] if os.path.exists(features_train_folder + f'{song_id}_CQT.json') and os.path.exists(annotations_path + f'{song_id}.json')]
df_train = df[df['SONG_ID'].isin(train_ids)]
train_ids_series = pd.Series(train_ids) if isinstance(df_train['SONG_ID'], pd.Series) else pd.DataFrame(train_ids)

model = load_model('checkpoints/2023-10-23_05-19-46_epoch_57.h5')
model.summary()
cqt, mfcc, tempogram, labels = generateDataset(val_ids_series, features_test_folder)


predictions = model.predict([cqt, tempogram])
y_test = np.argmax(labels, axis=1)
y_pred = np.argmax(predictions, axis=1)
#print(classification_report(y_test, y_pred))

songs = []

for song_id in df['SONG_ID']:
    cqt_for_song = [(index, element) for index, element in enumerate(cqt) if element[1] == song_id]
    if len(cqt_for_song) > 0:
        first_idx_for_song = cqt_for_song[0][0]
        after_last_idx_for_song = cqt_for_song[-1][0] + 1
        cqt_for_song = [row[1] for row in cqt_for_song]

        beats = [element[2] for element in cqt_for_song]
        true_labs = [lab for lab in labels if lab[1] == song_id]
        pred_labs = predictions[first_idx_for_song:after_last_idx_for_song]
        pred_labs = convertPredToOneHot(pred_labs, song_id)
        song_template = getSongTemplate(true_labs, beats)
        song_pred_template = getSongTemplate(true_labs, beats, song_pred_labels=pred_labs)




print('DONE')

