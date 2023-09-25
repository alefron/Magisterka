import os
import json
from pydub import AudioSegment
import pydub.playback as playback

# Ścieżka do folderu z plikami JSON
json_folder_path = 'annotations/parsed_annotations'

# Wartość value, którą chcemy wyszukać
preferred_value = 'Solo'

# Numer wystąpienia preferowanej wartości
occurrence = 1  # Możesz zmienić na dowolny numer

# Funkcja do konwersji sekund na format "x minut i y sekund"
def format_seconds(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f'{minutes} minut i {seconds} sekund'


# Funkcja do odtwarzania pliku mp3 od określonej sekundy
def play_mp3_from_second(mp3_file_path, start_second, end_second, song_id):
    audio = AudioSegment.from_mp3(mp3_file_path)
    segment = audio[start_second * 1000:end_second * 1000]
    formatted_start = format_seconds(start_second)
    formatted_end = format_seconds(end_second)
    print(f'Odtwarzanie pliku MP3 o nazwie {song_id}.mp3 od {formatted_start} do {formatted_end} sekundy')
    playback.play(segment)



# Szukamy adnotacji o preferowanej wartości w plikach JSON
found_occurrences = 0

for root, dirs, files in os.walk(json_folder_path):
    for filename in files:
        if filename.endswith('.json'):
            json_file_path = os.path.join(root, filename)

            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)

                for entry in data['data']:
                    if entry['value'] == preferred_value:
                        found_occurrences += 1

                        if found_occurrences == occurrence:
                            # Sprawdzamy, czy plik mp3 istnieje w obu folderach
                            song_id = os.path.splitext(filename)[0]
                            mp3_folder_path1 = 'salami_youtube_audios'
                            mp3_folder_path2 = 'salami_internet_archive_audios'
                            mp3_file_path1 = os.path.join(mp3_folder_path1, f'{song_id}.mp3')
                            mp3_file_path2 = os.path.join(mp3_folder_path2, f'{song_id}.mp3')

                            if os.path.exists(mp3_file_path1):
                                mp3_file_path = mp3_file_path1
                            elif os.path.exists(mp3_file_path2):
                                mp3_file_path = mp3_file_path2
                            else:
                                print(f'Nie znaleziono pliku MP3 o nazwie "{song_id}.mp3".')
                                exit(1)

                            # Odtwarzamy plik MP3
                            print(f'odtwarzanie z folderu: {mp3_file_path}')
                            play_mp3_from_second(mp3_file_path, entry['time'], entry['time'] + entry['duration'], song_id)
                            break

                        # Jeśli znaleziono, przechodzimy do następnego pliku JSON
                        break

            if found_occurrences == occurrence:
                break

# Jeśli nie znaleziono odpowiedniej adnotacji, informujemy użytkownika
if found_occurrences < occurrence:
    print(f'Nie znaleziono wystarczającej liczby adnotacji o wartości "{preferred_value}".')
