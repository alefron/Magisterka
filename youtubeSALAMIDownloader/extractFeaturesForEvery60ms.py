import librosa
import pandas as pd
import os
import json
from pydub import AudioSegment
from pydub.playback import play
import pygame
from IPython.display import Audio
import numpy as np
from scipy.io.wavfile import write

# Wczytaj plik metadata.csv
metadata = pd.read_csv('metadata_filtered_without_bad_labels.csv')[525:535]
features_folder = "D:\\features\\2"
records = []
music_folder = 'all_music'
done_sound_path = 'Aleksandra Front - Sekret.wav'

# Dla każdego wiersza w metadata
for index, row in metadata.iterrows():
    # Wczytaj plik mp3
    song_id = row['SONG_ID']
    audio_path = os.path.join(music_folder, f'{song_id}.mp3')
    print(f"plik: {song_id}.mp3")
    y, sr = librosa.load(audio_path, sr=16538)

    # Wyodrębnij cechy: CQT, MFCC, tempogram
    #y_harmonic, y_percussive = librosa.effects.hpss(y)
    n_bins = 84  # Liczba pasm w CQT
    hop_length = 1024

    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=n_bins).tolist()

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=14, hop_length=hop_length).tolist()

    tempogram = librosa.feature.tempogram(y=y, sr=sr, win_length=192, hop_length=hop_length).tolist()

    # Zamiana liczb zespolonych na format {'real': realna_część, 'imag': urojona_część}
    cqt = [[{"real": x.real, "imag": x.imag} for x in row] for row in cqt]
    mfcc = [[{"real": x.real, "imag": x.imag} for x in row] for row in mfcc]
    tempogram = [[{"real": x.real, "imag": x.imag} for x in row] for row in tempogram]

    # Zapisz cechy w formie rekordu
    for feature_name, feature_vectors in [("CQT", cqt), ("MFCC", mfcc), ("Tempogram", tempogram)]:
        transposed_vectors = [[row[i] for row in feature_vectors] for i in range(len(feature_vectors[0]))]
        max_vectors_Count = len(transposed_vectors) // 4 * 4
        transposed_vectors = transposed_vectors[:max_vectors_Count]
        grouped_vectors = [transposed_vectors[i:i+4] for i in range(0, len(transposed_vectors), 1)]
        record = {
            "SONG_ID": song_id,
            "Feature": {
                "Name": feature_name,
                "Vectors": grouped_vectors
            }
        }
        print(len(grouped_vectors))
        output_filename = f'{song_id}_{feature_name}.json'
        output_path = os.path.join(features_folder, output_filename)
        with open(output_path, 'w') as json_file:
            json.dump(record, json_file)

pygame.mixer.init()
pygame.mixer.music.load(done_sound_path)
pygame.mixer.music.play()
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)