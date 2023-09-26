import librosa

path = 'all_music/10.mp3'
y, sr = librosa.load(path, sr=16538)

print(librosa.get_duration(y=y, sr=sr) * 1000)
print(len(y))
