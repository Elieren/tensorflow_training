import numpy as np
import pandas as pd
import os
import librosa

# Загрузка данных о жанрах из файла
with open('C:\\Users\\kazan\\Videos\\git\\music_genres\\info\\genres_final.txt') as f:
    object_list = f.read().splitlines()

# Путь к папке с аудиофайлами
pictures_folder = 'C:\\Users\\kazan\\Videos\\git\\music_genres\\info\\music_WAV'

# Список файлов в папке
pictures_files = os.listdir(pictures_folder)

# Инициализация списков признаков и меток жанров
X = []
y = []

# Перебор каждого файла в папке
a = 0
for pictures_file in pictures_files:
    # Загрузка аудиоданных с помощью librosa
    pictures, sr = librosa.load(os.path.join(pictures_folder, pictures_file), sr=44100)
    # Преобразование стерео в моно-канал, если нужно
    if len(pictures.shape) > 1:
        pictures = librosa.to_mono(pictures)
    # Преобразование аудио в волну
    waveform = librosa.feature.hog_feature(y=pictures, sr=sr, n_mels=128)
    # Добавление признаков и метки жанра в соответствующие списки
    X.append(waveform.flatten())
    # Получение метки жанра из имени файла
    y.append(object_list[a])
    a += 1

# Создание обучающего и тестового наборов данных
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Преобразование списков в массивы numpy
X_train = np.array(X_train, dtype=object)
X_test = np.array(X_test, dtype=object)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(X_train)

'''
maxlen = 174
X_train = pad_sequences(X_train, maxlen=maxlen, dtype='float32', padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=maxlen, dtype='float32', padding='post', truncating='post')
y_train = pd.factorize(y_train)[0]
y_test = pd.factorize(y_test)[0]
'''
import tensorflow as tf

# Создание модели нейросети
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(28,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(object_list), activation='softmax')
])

# Преобразовать массивы NumPy в трехмерные массивы
#X_train = np.expand_dims(X_train, axis=2)
#X_test = np.expand_dims(X_test, axis=2)

# Преобразовать массивы NumPy в объекты Tensor
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

# Компиляция и обучение модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Обучение модели
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Загрузка аудиофайла
pictures_file = librosa.load('test_pictures_file.wav', sr=44100)

# Извлечение признаков аудиофайла
pictures_features = librosa.feature.hog_feature(y=pictures_file[0], sr=pictures_file[1], n_mels=128)
pictures_features = pictures_features.flatten().reshape(1, -1)

# Применение модели для предсказания жанра
predictions = model.predict(pictures_features)
predicted_genre = object_list[np.argmax(predictions)]
print("Predicted genre: ", predicted_genre)