import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers.legacy import Adam
from keras import metrics

# Definiowanie wejść
input_cqt = Input(shape=(84, 4, 1), name='cqt')
input_tempo = Input(shape=(192, 4, 1), name='tempogram')

# Definiowanie warstw Conv2D z odpowiednimi parametrami
conv_cqt = Conv2D(filters=128, kernel_size=(3, 1), activation='relu')(input_cqt)
conv_tempo = Conv2D(filters=128, kernel_size=(3, 1), activation='relu')(input_tempo)

conv_cqt_bn = BatchNormalization()(conv_cqt)
conv_tempo_bn = BatchNormalization()(conv_tempo)

# Definiowanie warstw MaxPooling2D
pool_cqt = MaxPooling2D(pool_size=(2, 2))(conv_cqt_bn)
pool_tempo = MaxPooling2D(pool_size=(2, 2))(conv_tempo_bn)

conv_cqt3 = Conv2D(filters=128, kernel_size=(3, 1), activation='relu')(pool_cqt)
conv_tempo3 = Conv2D(filters=128, kernel_size=(3, 1), activation='relu')(pool_tempo)

conv_cqt_bn3 = BatchNormalization()(conv_cqt3)
conv_tempo_bn3 = BatchNormalization()(conv_tempo3)


# Dodanie warstw Dropout
dropout_rate = 0.25  # Tutaj możesz dostosować stopień dropout
dropout_cqt = Dropout(rate=dropout_rate)(conv_cqt_bn3)
dropout_tempo = Dropout(rate=dropout_rate)(conv_tempo_bn3)

# Kolejne warstwy Conv2D
conv_cqt2 = Conv2D(filters=128, kernel_size=(3, 1), activation='relu')(dropout_cqt)
conv_tempo2 = Conv2D(filters=128, kernel_size=(3, 1), activation='relu')(dropout_tempo)

conv_cqt_bn2 = BatchNormalization()(conv_cqt2)
conv_tempo_bn2 = BatchNormalization()(conv_tempo2)

# Łączenie równoległych ścieżek
concatenated = Concatenate(axis=1)([conv_cqt_bn2, conv_tempo_bn2])

# Kolejna warstwa Conv2D
conv_final = Conv2D(filters=64, kernel_size=(3, 1), activation='relu')(concatenated)

batch_final = BatchNormalization()(conv_final)

# Warstwa Flatten
flattened = Flatten()(batch_final)

# Dodanie warstwy Dropout przed warstwą Dense (wyjściową)
dropout_final = Dropout(rate=dropout_rate)(flattened)

# Warstwa Dense (wyjście)
output = Dense(9, activation='softmax')(dropout_final)

# Tworzenie modelu
model = Model(inputs=[input_cqt, input_tempo], outputs=output)

optimizer = Adam(learning_rate=0.01)
#optimizer = Adagrad(learning_rate=0.01)
# Kompilacja modelu
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[
    tf.keras.metrics.CategoricalAccuracy(),
    tf.keras.metrics.TopKCategoricalAccuracy(k=3),
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall()
])

# zapisanie modelu
model.save('model_base3.h5')

# Wyświetlenie architektury modelu
model.summary()

