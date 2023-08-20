import csv
import os

def count_files_with_high_coverage(csv_file, audio_folder, threshold):
    count = 0

    with open(csv_file, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            youtube_id = row['youtube_id']
            audio_file = os.path.join(audio_folder, f"{youtube_id}.mp3")

            if os.path.exists(audio_file):
                coverage_percent = float(row['coverage_percent'])
                if coverage_percent > threshold:
                    count += 1

    return count

if __name__ == "__main__":
    csv_file = "salami_youtube_pairings.csv"
    audio_folder = "youtube_audios"
    coverage_threshold = 0.98

    num_files_above_threshold = count_files_with_high_coverage(csv_file, audio_folder, coverage_threshold)

    print(f"Ilość plików z pokryciem powyżej {coverage_threshold}%: {num_files_above_threshold}")
