import os
import csv
import requests

def download_mp3(url, song_id, output_folder):
    mp3_path = os.path.join(output_folder, f"{song_id}.mp3")
    if os.path.exists(mp3_path):
        print(f"Plik o ID {song_id} już istnieje. Pomijam pobieranie.")
        return

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        if response.status_code == 200:
            mp3_path = os.path.join(output_folder, f"{song_id}.mp3")
            with open(mp3_path, 'wb') as mp3_file:
                mp3_file.write(response.content)
            print(f"Pobrano i zapisano: {mp3_path}")
    except requests.exceptions.RequestException as e:
        print(f"Błąd podczas pobierania pliku o ID: {song_id}. Błąd: {e}")

def download_mp3s_from_csv(csv_file, output_folder):
    with open(csv_file, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            song_id = row['SONG_ID']
            url = row['URL']
            download_mp3(url, song_id, output_folder)

if __name__ == "__main__":
    csv_file = "id_index_internetarchive.csv"
    output_folder = "salami_internet_archive_audios"

    os.makedirs(output_folder, exist_ok=True)
    download_mp3s_from_csv(csv_file, output_folder)
