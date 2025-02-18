import numpy
import librosa
from tensorflow import keras


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
    mfcc_feature = numpy.concatenate((mfcc_mean, mfcc_min, mfcc_max))

    # Extracting Mel Spectrogram feature
    melspectrogram = get_melspectrogram(file_path)
    melspectrogram_mean = melspectrogram.mean(axis=1)
    melspectrogram_min = melspectrogram.min(axis=1)
    melspectrogram_max = melspectrogram.max(axis=1)
    melspectrogram_feature = numpy.concatenate((melspectrogram_mean,
                                                melspectrogram_min,
                                                melspectrogram_max))

    # Extracting chroma vector feature
    chroma = get_chroma_vector(file_path)
    chroma_mean = chroma.mean(axis=1)
    chroma_min = chroma.min(axis=1)
    chroma_max = chroma.max(axis=1)
    chroma_feature = numpy.concatenate((chroma_mean, chroma_min, chroma_max))

    # Extracting tonnetz feature
    tntz = get_tonnetz(file_path)
    tntz_mean = tntz.mean(axis=1)
    tntz_min = tntz.min(axis=1)
    tntz_max = tntz.max(axis=1)
    tntz_feature = numpy.concatenate((tntz_mean, tntz_min, tntz_max))

    feature = numpy.concatenate((chroma_feature, melspectrogram_feature,
                                mfcc_feature, tntz_feature))
    return feature

# ---------------------------------------------------------------------------------#


genres_1 = ['Rock', 'Phonk', 'Synthwave', 'Jazz', 'EDM', 'Metal', 'Nightcore',
            'Dubstep', 'Score', 'Frenchcore', 'Uptempo', 'Speedcore', 'Terror',
            'Synth-rock']

model = keras.models.Sequential([
    keras.layers.Dense(512, activation="relu", name="dense_1",
                       input_shape=(498,)),
    keras.layers.Dense(256, activation="relu", name="dense_2"),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(256, activation="relu", name="dense_3"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(256, activation="relu", name="dense_4"),
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(14, activation="softmax", name="predictions")
])

model.compile(
    # Optimizer
    optimizer='adam',
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=['accuracy'],
)

model.load_weights('model\\audio\\model_weights.weights.h5')


file_path = "info/music/Phonk/MUKBANG-Haarper.wav"
feature = get_feature(file_path)
y = model.predict(feature.reshape(1, 498))
ind = numpy.argmax(y)
print(genres_1[ind], '=> Phonk (MUKBANG)')
