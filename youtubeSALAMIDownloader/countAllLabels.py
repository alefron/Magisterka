import os
import json
import pandas as pd
import csv

# Ścieżka do folderu z przefiltrowanym plikiem metadata.csv
metadata_file_path = 'metadata_filtered.csv'

# Ścieżka do folderu z plikami JSON
json_folder_path = 'annotations/parsed_annotations'

output_csv_file_path = 'labelsStats.csv'

# Wczytujemy przefiltrowany plik metadata.csv
metadata_df = pd.read_csv(metadata_file_path)

# Tworzymy słownik do zliczania wystąpień wartości "value"
value_counts = {}

# Przechodzimy przez wiersze w przefiltrowanym pliku metadata.csv
for index, row in metadata_df.iterrows():
    song_id = row['SONG_ID']
    json_file_path = os.path.join(json_folder_path, f'{song_id}.json')

    # Sprawdzamy, czy plik JSON istnieje
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

            # Przechodzimy przez dane w pliku JSON i zliczamy wystąpienia wartości "value"
            for entry in data['data']:
                value = entry['value']
                if value in value_counts:
                    value_counts[value] += 1
                else:
                    value_counts[value] = 1

# Sortujemy wyniki zliczania od największej do najmniejszej liczby wystąpień
sorted_value_counts = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)

# Wyświetlamy wyniki zliczania
for value, count in sorted_value_counts:
    print(f'Wartość "{value}" występuje {count} razy.')

# Zapisujemy wyniki zliczania do pliku CSV
with open(output_csv_file_path, 'w', newline='') as output_csv:
    csv_writer = csv.writer(output_csv)
    csv_writer.writerow(['Value', 'Count'])  # Nagłówki kolumn

    for value, count in sorted_value_counts:
        csv_writer.writerow([value, count])

