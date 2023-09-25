import os
import pandas as pd

# Ścieżka do pliku metadata.csv
metadata_file_path = 'metadata.csv'

# Ścieżki do folderów z plikami MP3
folder1_path = 'salami_youtube_audios'
folder2_path = 'salami_internet_archive_audios'

# Wczytujemy plik metadata.csv
metadata_df = pd.read_csv(metadata_file_path)

# Funkcja sprawdzająca, czy istnieje plik MP3 o danym songID w folderze
def mp3_exists(songID):
    mp3_file1 = os.path.join(folder1_path, f'{songID}.mp3')
    mp3_file2 = os.path.join(folder2_path, f'{songID}.mp3')
    return os.path.exists(mp3_file1) or os.path.exists(mp3_file2)

# Filtrujemy wiersze z pliku metadata.csv
filtered_metadata_df = metadata_df[metadata_df['SONG_ID'].apply(mp3_exists)]

# Ścieżka do nowego pliku CSV
output_csv_file = 'metadata_filtered.csv'

# Zapisujemy przefiltrowane dane do nowego pliku CSV
filtered_metadata_df.to_csv(output_csv_file, index=False)
