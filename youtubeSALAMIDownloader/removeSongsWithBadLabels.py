import os
import json
import pandas as pd

# Ścieżka do folderu z przefiltrowanym plikiem metadata.csv
metadata_file_path = 'metadata_filtered.csv'

# Ścieżka do folderu z plikami JSON
json_folder_path = 'annotations/parsed_annotations'

# Wartości "value", które chcemy sprawdzić
target_values = ['Secondary-Theme', 'Development', 'Recap', 'variation-2', 'variation', 'variation-1']

# Wczytujemy przefiltrowany plik metadata.csv
metadata_df = pd.read_csv(metadata_file_path)

# Lista indeksów wierszy do usunięcia
rows_to_remove = []

# Przechodzimy przez wiersze w przefiltrowanym pliku metadata.csv
for index, row in metadata_df.iterrows():
    song_id = row['SONG_ID']
    json_file_path = os.path.join(json_folder_path, f'{song_id}.json')

    # Sprawdzamy, czy plik JSON istnieje
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

            # Przechodzimy przez dane w pliku JSON i sprawdzamy, czy którykolwiek "value" jest w docelowych wartościach
            for entry in data['data']:
                if entry['value'] in target_values:
                    rows_to_remove.append(index)
                    break  # Jeśli znaleziono jedną z docelowych wartości, przechodzimy do następnego wiersza
                else:
                    continue  # "else" jest uruchamiane tylko, jeśli pętla "for" nie została przerwana

# Usuwamy wiersze z odpowiadającymi plikami JSON zawierającymi docelowe wartości
filtered_metadata_df = metadata_df.drop(rows_to_remove)

# Zapisujemy nowy przefiltrowany plik metadata.csv
filtered_metadata_file_path = 'metadata_filtered_without_bad_labels.csv'
filtered_metadata_df.to_csv(filtered_metadata_file_path, index=False)

print(f'Nowy przefiltrowany plik CSV został zapisany jako {filtered_metadata_file_path}')
