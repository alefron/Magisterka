import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

# Definiowanie wejść
input_cqt = Input(shape=(84, 4, 2), name='cqt')
input_mfcc = Input(shape=(14, 4, 2), name='mfcc')
input_tempo = Input(shape=(192, 4, 2), name='tempogram')

# Definiowanie warstw Conv2D z odpowiednimi parametrami
conv_cqt = Conv2D(filters=128, kernel_size=(3, 1), activation='relu')(input_cqt)
conv_mfcc = Conv2D(filters=128, kernel_size=(3, 1), activation='relu')(input_mfcc)
conv_tempo = Conv2D(filters=128, kernel_size=(3, 1), activation='relu')(input_tempo)

# Definiowanie warstw MaxPooling2D
pool_cqt = MaxPooling2D(pool_size=(2, 2))(conv_cqt)
pool_mfcc = MaxPooling2D(pool_size=(2, 2))(conv_mfcc)
pool_tempo = MaxPooling2D(pool_size=(2, 2))(conv_tempo)

# Kolejne warstwy Conv2D
conv_cqt2 = Conv2D(filters=128, kernel_size=(3, 1), activation='relu')(pool_cqt)
conv_mfcc2 = Conv2D(filters=128, kernel_size=(3, 1), activation='relu')(pool_mfcc)
conv_tempo2 = Conv2D(filters=128, kernel_size=(3, 1), activation='relu')(pool_tempo)

# Łączenie równoległych ścieżek
concatenated = Concatenate(axis=1)([conv_cqt2, conv_mfcc2, conv_tempo2])

# Kolejna warstwa Conv2D
conv_final = Conv2D(filters=64, kernel_size=(3, 1), activation='relu')(concatenated)

# Warstwa Flatten
flattened = Flatten()(conv_final)

# Warstwa Dense (wyjście)
output = Dense(10, activation='softmax')(flattened)

# Tworzenie modelu
model = Model(inputs=[input_cqt, input_mfcc, input_tempo], outputs=output)

# Kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# zapisanie modelu
model.save('model_base.h5')

# Wyświetlenie architektury modelu
model.summary()

