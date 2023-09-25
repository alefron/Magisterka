import pandas
import jams
import os
import numpy as np
from collections import Counter

folder_path = "/content/drive/MyDrive/references"
loaded_files = []

for file_name in os.listdir(folder_path):
    if file_name.endswith(".jams"):
        file_path = os.path.join(folder_path, file_name)
        try:
            jam = jams.load(file_path)
            loaded_files.append(jam)
            print("Wczytano plik:", file_path)
        except jams.JAMSError as e:
            print("Błąd wczytywania pliku:", file_path)
            print(e)