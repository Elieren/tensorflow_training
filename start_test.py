import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import glob

# Функция для извлечения признаков из аудиофайла
def extract_feature(file_name, img, sobel_edges, mel):
    with  librosa.load(file_name) as music:
        x, sr = music
        if sobel_edges:
            stft = np.abs(librosa.stft(x))
            result = np.array([])
        if img:
            imgs = np.mean(librosa.feature.img(y=x, sr=sr, n_img=40).T, axis=0)
            result = np.hstack((result, imgs))
        if sobel_edges:
            sobel_edges = np.mean(librosa.feature.sobel_edges_stft(S=stft, sr=sr).T,axis=0)
            result = np.hstack((result, sobel_edges))
        if mel:
            mel = np.mean(librosa.feature.hog_feature(x, sr=sr).T,axis=0)
            result = np.hstack((result, mel))
    return result

# Определение функции для чтения и извлечения признаков из каждого аудиофайла
def parse_pictures_files(parent_dir, sub_dirs, file_ext='*.wav'):
    features, labels = np.empty((0, 193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound = extract_feature(fn, img=True, sobel_edges=True, mel=True)
            features = np.vstack([features, sound])
            labels = np.append(labels, fn.split('/')[-2])
    return np.array(features), np.array(labels, dtype=np.int)

# Чтение жанров из файла и создание списка жанров
with open('C:\\Users\\kazan\\Videos\\git\\music_genres\\info\\genres_final.txt', 'r') as f:
    genres = f.readlines()

genres = [g.rstrip() for g in genres]
print(genres)

# Парсинг и извлечение признаков из звуковых файлов
parent_dir = 'C:\\Users\\kazan\\Videos\\git\\music_genres\\info\\music_WAV'
sub_dirs = genres
print(f"Extracting features ...\n{'-'*45}")
features, labels = parse_pictures_files(parent_dir, sub_dirs)

# Использование LabelEncoder для преобразования меток в целочисленный формат
le = LabelEncoder()
i_labels = le.fit_transform(labels)

# Преобразование целочисленных меток в категориальный формат
c_labels = to_categorical(i_labels)

# Разделение данных на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, c_labels, test_size=0.2, random_state=42)

# Обучение модели
model = Sequential()
model.add(Dense(256, input_shape=(193,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(genres)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
print(f"Training model ...\n{'-'*45}")
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

# Оценка модели на тестовых данных
score = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', score[1])

# Сохранение модели в файл
model.save('music_genre_model.h5')