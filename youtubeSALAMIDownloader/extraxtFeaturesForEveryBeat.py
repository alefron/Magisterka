import librosa
import pandas as pd
import os
import json
import pygame
import numpy as np
# Wczytaj plik metadata.csv
metadata = pd.read_csv('metadata_filtered_without_bad_labels.csv')[0:100]
features_folder = "/Volumes/Czerwony/features/beat_features"
records = []
music_folder = 'all_music'
done_sound_path = 'rp1.wav'

# Dla każdego wiersza w metadata
for index, row in metadata.iterrows():
    # Wczytaj plik mp3
    song_id = row['SONG_ID']
    audio_path = os.path.join(music_folder, f'{song_id}.mp3')
    print(f"plik: {song_id}.mp3")
    y, sr = librosa.load(audio_path, sr=22050)
    secs = np.size(y) / sr
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    tempo, beat_frames = librosa.beat.beat_track(y=y_harmonic, sr=sr)
    print('Detected Tempo: ' + str(tempo) + ' bpm')

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_time_diff = np.ediff1d(beat_times)
    beat_nums = np.arange(1, np.size(beat_times))

    cqts = []
    mfccs = []
    tempograms = []

    for index, frame in enumerate(beat_frames):
        start_sample = librosa.frames_to_samples(beat_frames[index])
        end_sample = 0
        if (index + 1) < len(beat_frames):
            end_sample = librosa.frames_to_samples(beat_frames[index + 1])
        if (index + 1) == len(beat_frames):
            end_sample = len(y_harmonic) - 1

        #samples for beat
        fragment = y_harmonic[start_sample:end_sample]

        #adjusting hop length
        hop = 1024
        if len(fragment) <= 2048:
            hop = len(fragment) // 2
            def is_power_of_two(n):
                return n & (n - 1) == 0 and n != 0

            while is_power_of_two(hop):
                hop = hop - 1
                if hop < 2:
                    hop = 8


        #features for beat
        cqt_beat = librosa.cqt(fragment, sr=sr, hop_length=hop, n_bins=84)
        mfcc_beat = librosa.feature.mfcc(y=fragment, sr=sr, n_mfcc=14, hop_length=hop)
        tempogram_beat = librosa.feature.tempogram(y=fragment, sr=sr, win_length=192, hop_length=hop)

        # counting mean within a beat
        cqt_beat_mean = np.mean(cqt_beat, axis=1, keepdims=True)
        mfcc_beat_mean = np.mean(mfcc_beat, axis=1, keepdims=True)
        tempogram_beat_mean = np.mean(tempogram_beat, axis=1, keepdims=True)

        # convert cqt value to object
        cqt_beat_mean = np.array([[{"real": str(x.real), "imag": str(x.imag)} for x in row] for row in cqt_beat_mean])

        #adding to lists of beats features
        cqts.append([beat_times[index], cqt_beat_mean])
        mfccs.append([beat_times[index], mfcc_beat_mean])
        tempograms.append([beat_times[index], tempogram_beat_mean])

    cqts_serializable = [[str(beat[0]), [val[0] for val in beat[1]]] for beat in cqts]
    mfccs_serializable = [[str(beat[0]), [str(val[0]) for val in beat[1]]] for beat in mfccs]
    tempograms_serializable = [[str(beat[0]), [str(val[0]) for val in beat[1]]] for beat in tempograms]

    # beats count
    for feature_name, feature_vectors in [("CQT", cqts_serializable), ("MFCC", mfccs_serializable), ("Tempogram", tempograms_serializable)]:
        record = {
            "SONG_ID": song_id,
            "Feature": {
                "Name": feature_name,
                "Vectors": feature_vectors
            }
        }
        output_filename = f'{song_id}_{feature_name}.json'
        output_path = os.path.join(features_folder, output_filename)
        with open(output_path, 'w') as json_file:
            json.dump(record, json_file)

pygame.mixer.init()
pygame.mixer.music.load(done_sound_path)
pygame.mixer.music.play()
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)















