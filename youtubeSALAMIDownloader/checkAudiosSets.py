import os

#sprawdza czy zbiory salami_youtube_audios i salami_internet_archive_audios są rozłączne

# Ścieżki do folderów
folder1_path = 'salami_youtube_audios'
folder2_path = 'salami_internet_archive_audios'

# Pobieramy listy plików mp3 w obu folderach
folder1_files = set(file.split('.')[0] for file in os.listdir(folder1_path) if file.endswith('.mp3'))
folder2_files = set(file.split('.')[0] for file in os.listdir(folder2_path) if file.endswith('.mp3'))

# Znajdujemy pliki, które występują w obu folderach
common_files = folder1_files.intersection(folder2_files)

# Wyświetlamy nazwy wspólnych plików
if len(common_files) == 0:
    print("Zbiory sa rozlaczne i to dobrze")
for common_file in common_files:
    print(f'Plik mp3 występuje w obu folderach: {common_file}.mp3')
