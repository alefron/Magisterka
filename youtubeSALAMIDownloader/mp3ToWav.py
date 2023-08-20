from moviepy.editor import AudioFileClip
import os

def convert_mp3_to_wav(mp3_path, wav_path):
    audio_clip = AudioFileClip(mp3_path)
    audio_clip.write_audiofile(wav_path, codec='pcm_s16le')

def batch_convert_mp3_to_wav(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mp3_files = [f for f in os.listdir(input_folder) if f.endswith(".mp3")]

    for mp3_file in mp3_files:
        mp3_path = os.path.join(input_folder, mp3_file)
        wav_file = os.path.splitext(mp3_file)[0] + ".wav"
        wav_path = os.path.join(output_folder, wav_file)

        convert_mp3_to_wav(mp3_path, wav_path)
        print(f"Converted {mp3_file} to {wav_file}")

if __name__ == "__main__":
    input_folder = "youtube_audios"
    output_folder = "youtube_audios_wav"

    batch_convert_mp3_to_wav(input_folder, output_folder)
