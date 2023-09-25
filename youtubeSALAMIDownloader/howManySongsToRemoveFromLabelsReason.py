import os
import json

# Ścieżka do folderu z plikami JSON
json_folder_path = 'annotations/parsed_annotations'

# Wartości "value", które chcemy sprawdzić
target_values = ['Secondary-Theme', 'Development', 'Recap', 'variation-2', 'variation', 'variation-1']

# Licznik plików JSON zawierających przynajmniej jedną z docelowych wartości
file_count = 0

# Przechodzimy przez pliki JSON w folderze
for root, dirs, files in os.walk(json_folder_path):
    for filename in files:
        if filename.endswith('.json'):
            json_file_path = os.path.join(root, filename)

            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)

                # Przechodzimy przez dane w pliku JSON i sprawdzamy, czy którykolwiek "value" jest w docelowych wartościach
                for entry in data['data']:
                    if entry['value'] in target_values:
                        file_count += 1
                        break  # Jeśli znaleziono jedną z docelowych wartości, przechodzimy do następnego pliku JSON
                else:
                    continue  # "else" jest uruchamiane tylko, jeśli pętla "for" nie została przerwana, więc plik nie zawierał docelowych wartości

# Wyświetlamy wynik
print(f'Liczba plików JSON zawierających przynajmniej jedną adnotację z docelowymi wartościami: {file_count}')
