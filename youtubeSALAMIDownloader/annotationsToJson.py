import os
import json


def read_music_sections(file_path):
    times = []
    sections = []

    with open(file_path, 'r') as file:
        for line in file:
            time, section = line.strip().split('\t')
            times.append(float(time))
            sections.append(section)

    return times, sections


def convert_to_json(times, sections):
    data = []

    for i in range(len(times)):
        entry = {
            "duration": times[i + 1] - times[i] if i < len(times) - 1 else 0,
            "value": sections[i].replace("_", "-"),
            "time": times[i]
        }
        data.append(entry)

    return data


def save_json_file(data, output_file_path):
    json_data = {"data": data}

    with open(output_file_path, 'w') as output_file:
        json.dump(json_data, output_file, indent=2)


def process_files_in_directory(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        if "parsed" in dirs:
            dirs.remove("parsed")
            parsed_folder = os.path.join(root, "parsed")
            parent_folder = os.path.basename(root)  # Pobieramy nazwę rodzica

            for filename in os.listdir(parsed_folder):
                if filename.endswith("functions.txt"):
                    input_file_path = os.path.join(parsed_folder, filename)
                    output_file_path = os.path.join(output_dir, parent_folder + ".json")

                    times, sections = read_music_sections(input_file_path)
                    json_data = convert_to_json(times, sections)
                    save_json_file(json_data, output_file_path)


input_directory = 'annotations'
output_directory = 'annotations/parsed_annotations'

process_files_in_directory(input_directory, output_directory)
