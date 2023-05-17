import os
import numpy
from tensorflow import keras
import librosa
from matplotlib import pyplot
import pickle

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

genres_1 = ['Rock','Phonk','Synthwave','Jazz','EDM','Metal','Nightcore','Dubstep','Score','Frenchcore','Uptempo','Speedcore','Terror','Synth-rock']
print(len(genres_1))
features = []
labels = []

# Путь к папке с аудиофайлами
audio_folder = './info/music'

# Список файлов в папке
audio_files = os.listdir(audio_folder)

# Инициализация списков признаков и меток жанров

# Перебор каждого файла в папке
for genre_folder in os.listdir(audio_folder):
    genre_path = os.path.join(audio_folder, genre_folder)
    if os.path.isdir(genre_path) and any(substring in genre_folder for substring in genres_1):
        # Перебор каждого WAV файла в папке-жанре
        for audio_file in os.listdir(genre_path):
            if audio_file.endswith('.wav'):
                # Добавление признаков и метки жанра в соответствующие списки
                genre = [substring for substring in genres_1 if substring in genre_folder][0]
                features.append(get_feature(os.path.join(genre_path, audio_file)))
                labels.append(genres_1.index(genre))
                print(f'🟢 {genre} --- 🎶 {audio_file[:-4]}')

#-------------------------------------------------------------------------#

with open('dataset_db/audio/dataset_features.dat', 'wb') as file:
    pickle.dump(features, file)

with open('dataset_db/audio/dataset_labels.dat', 'wb') as file:
    pickle.dump(labels, file)