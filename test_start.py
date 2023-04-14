import os
import numpy
from tensorflow import keras
import librosa
from matplotlib import pyplot

def get_mfcc(wav_file_path):
    y, sr = librosa.load(wav_file_path, offset=0, duration=30)
    mfcc = numpy.array(librosa.feature.mfcc(y=y, sr=sr))
    return mfcc

def get_melspectrogram(wav_file_path):
    y, sr = librosa.load(wav_file_path, offset=0, duration=30)
    melspectrogram = numpy.array(librosa.feature.melspectrogram(y=y, sr=sr))
    return melspectrogram

def get_chroma_vector(wav_file_path):
    y, sr = librosa.load(wav_file_path)
    chroma = numpy.array(librosa.feature.chroma_stft(y=y, sr=sr))
    return chroma

def get_tonnetz(wav_file_path):
    y, sr = librosa.load(wav_file_path)
    tonnetz = numpy.array(librosa.feature.tonnetz(y=y, sr=sr))
    return tonnetz

def get_feature(file_path):
    # Extracting MFCC feature
    mfcc = get_mfcc(file_path)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_min = mfcc.min(axis=1)
    mfcc_max = mfcc.max(axis=1)
    mfcc_feature = numpy.concatenate( (mfcc_mean, mfcc_min, mfcc_max) )

    # Extracting Mel Spectrogram feature
    melspectrogram = get_melspectrogram(file_path)
    melspectrogram_mean = melspectrogram.mean(axis=1)
    melspectrogram_min = melspectrogram.min(axis=1)
    melspectrogram_max = melspectrogram.max(axis=1)
    melspectrogram_feature = numpy.concatenate( (melspectrogram_mean, melspectrogram_min, melspectrogram_max) )

    # Extracting chroma vector feature
    chroma = get_chroma_vector(file_path)
    chroma_mean = chroma.mean(axis=1)
    chroma_min = chroma.min(axis=1)
    chroma_max = chroma.max(axis=1)
    chroma_feature = numpy.concatenate( (chroma_mean, chroma_min, chroma_max) )

    # Extracting tonnetz feature
    tntz = get_tonnetz(file_path)
    tntz_mean = tntz.mean(axis=1)
    tntz_min = tntz.min(axis=1)
    tntz_max = tntz.max(axis=1)
    tntz_feature = numpy.concatenate( (tntz_mean, tntz_min, tntz_max) ) 

    feature = numpy.concatenate( (chroma_feature, melspectrogram_feature, mfcc_feature, tntz_feature) )
    return feature

#---------------------------------------------------------------------------------#

genres_1 = ['Rock','Phonk','Synthwave','Jazz','EDM','Metal','Nightcore','Dubstep','Score','Frenchcore','Uptempo']
print(len(genres_1))
features = []
labels = []

with open('C:\\Users\\kazan\\Videos\\git\\music_genres\\info\\genres_final.txt') as f:
    genre_list = f.read().splitlines()

# Путь к папке с аудиофайлами
audio_folder = 'C:\\Users\\kazan\\Videos\\git\\music_genres\\info\\music_WAV'

# Список файлов в папке
audio_files = os.listdir(audio_folder)

# Инициализация списков признаков и меток жанров

# Перебор каждого файла в папке
for a, audio_file in enumerate(audio_files):
    file_path = 'C:\\Users\\kazan\\Videos\\git\\music_genres\\info\\music_WAV\\' + audio_file

    features.append(get_feature(file_path))
    labels.append(genres_1.index(genre_list[a]))
    print(audio_file, genre_list[a], genres_1.index(genre_list[a]))

#-------------------------------------------------------------------------#

permutations = numpy.random.permutation(35)
features = numpy.array(features)[permutations]
labels = numpy.array(labels)[permutations]

features_train = features[0:20]
labels_train = labels[0:20]

features_val = features[20:30]
labels_val = labels[20:30]

features_test = features[30:35]
labels_test = labels[30:35]

inputs = keras.Input(shape=(498), name="feature")
x = keras.layers.Dense(256, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(128, activation="relu", name="dense_2")(x)
outputs = keras.layers.Dense(11, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    # Optimizer
    optimizer=keras.optimizers.RMSprop(),
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
model.fit(x=features_train.tolist(),y=labels_train.tolist(),verbose=1,validation_data=(features_val.tolist() , labels_val.tolist()), epochs=10000)

score = model.evaluate(x=features_test.tolist(),y=labels_test.tolist(), verbose=0)
print(score)
print('Accuracy : ' + str(score[1]*100) + '%')

file_path = "C:\\Users\\kazan\\Desktop\\1.mp4.wav"
feature = get_feature(file_path)
y = model.predict(feature.reshape(1,498))
ind = numpy.argmax(y)
print(genres_1[ind], '=> Phonk (MUKBANG)')

file_path = "C:\\Users\\kazan\\Desktop\\2.wav"
feature = get_feature(file_path)
y = model.predict(feature.reshape(1,498))
ind = numpy.argmax(y)
print(genres_1[ind], '=> ?El Tigro')

file_path = "C:\\Users\\kazan\\Desktop\\3.mp4.wav"
feature = get_feature(file_path)
y = model.predict(feature.reshape(1,498))
ind = numpy.argmax(y)
print(genres_1[ind], '=> EDM (Nana)')

file_path = "C:\\Users\\kazan\\Desktop\\1.wav"
feature = get_feature(file_path)
y = model.predict(feature.reshape(1,498))
ind = numpy.argmax(y)
print(genres_1[ind], '=> ? (Been Good To Know Ya)')

file_path = "C:\\Users\\kazan\\Desktop\\2.mp4.wav"
feature = get_feature(file_path)
y = model.predict(feature.reshape(1,498))
ind = numpy.argmax(y)
print(genres_1[ind], '=> EDM (Zenith)')

'''
example_file = "C:\\Users\\kazan\\Videos\\git\\music_genres\\info\\music_WAV\\Cyberpunk_2077_-_The_Ballad_of_Buck_Ravers_by_SAMURAI_Refused_69859526(rock).wav"
mfcc = get_mfcc(example_file)
pyplot.imshow(mfcc, interpolation='nearest', aspect='auto')
pyplot.show()

melspectrogram = get_melspectrogram(example_file)
pyplot.imshow(melspectrogram, interpolation='nearest', aspect='auto')
pyplot.show()

chroma = get_chroma_vector(example_file)
pyplot.imshow(chroma, interpolation='nearest', aspect='auto')
pyplot.show()

tntz = get_tonnetz(example_file)
pyplot.imshow(tntz , interpolation='nearest', aspect='auto')
pyplot.show()
'''