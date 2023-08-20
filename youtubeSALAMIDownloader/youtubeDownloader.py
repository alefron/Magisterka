from pytube import YouTube
import os
import csv

def download_youtube_audio(url, output_path):
    try:
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()

        print(f"Pobieranie audio dla: {url}...")
        out_file = audio_stream.download(output_path=output_path)

        youtube_id = url.split("=")[-1]
        new_file = os.path.join(output_path, f"{youtube_id}.mp3")
        os.rename(out_file, new_file)

    except Exception as e:
        print("Wystąpił błąd:", e)

if __name__ == "__main__":
    csv_file = "salami_youtube_pairings.csv"
    output_folder = "youtube_audios"

    os.makedirs(output_folder, exist_ok=True)

    with open(csv_file, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            youtube_id = row['youtube_id']
            youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"
            download_youtube_audio(youtube_url, output_folder)
