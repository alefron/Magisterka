import os
import csv
import wave
from pydub import AudioSegment


def get_audio_length(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension == '.wav':
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
    elif file_extension == '.mp3':
        audio = AudioSegment.from_file(file_path)
        duration = len(audio) / 1000  # convert to seconds
    else:
        raise ValueError("Unsupported audio format")

    return duration


def compare_audio_lengths(csv_file, folder_wav, folder_mp3):
    wav_files = os.listdir(folder_wav)
    mp3_files = os.listdir(folder_mp3)
    different_lengths_count = 0

    with open(csv_file, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            youtube_id = row['youtube_id']
            salami_id = row['salami_id']

            if f"{youtube_id}.wav" in wav_files and f"{salami_id}.mp3" in mp3_files:
                wav_path = os.path.join(folder_wav, f"{youtube_id}.wav")
                mp3_path = os.path.join(folder_mp3, f"{salami_id}.mp3")

                wav_duration = get_audio_length(wav_path)
                mp3_duration = get_audio_length(mp3_path)

                if abs(wav_duration - mp3_duration) > 1e-6:  # Comparing floating point numbers with a small epsilon
                    different_lengths_count += 1
                    print(f"Different lengths for YouTube ID: {youtube_id}, Salami ID: {salami_id}")
                    print(f"WAV Duration: {wav_duration} seconds")
                    print(f"MP3 Duration: {mp3_duration} seconds")
                    print("=" * 30)

    print(f"Total number of pairs with different lengths: {different_lengths_count}")


if __name__ == "__main__":
    csv_file = "salami_youtube_pairings.csv"  # Zmień na nazwę swojego pliku CSV
    folder_wav = "youtube_audios_wav"
    folder_mp3 = "salami_youtube_audios"

    compare_audio_lengths(csv_file, folder_wav, folder_mp3)
