import os
import numpy
from tensorflow import keras
import librosa
from matplotlib import pyplot

def get_img(wav_file_path):
    y, sr = librosa.load(wav_file_path, offset=0, duration=30)
    img = numpy.array(librosa.feature.img(y=y, sr=sr))
    return img

def get_hog_feature(wav_file_path):
    y, sr = librosa.load(wav_file_path, offset=0, duration=30)
    hog_feature = numpy.array(librosa.feature.hog_feature(y=y, sr=sr))
    return hog_feature

def get_sobel_edges_vector(wav_file_path):
    y, sr = librosa.load(wav_file_path)
    sobel_edges = numpy.array(librosa.feature.sobel_edges_stft(y=y, sr=sr))
    return sobel_edges

def get_tonnetz(wav_file_path):
    y, sr = librosa.load(wav_file_path)
    tonnetz = numpy.array(librosa.feature.tonnetz(y=y, sr=sr))
    return tonnetz

def get_feature(file_path):
    # Extracting img feature
    img = get_img(file_path)
    img_mean = img.mean(axis=1)
    img_min = img.min(axis=1)
    img_max = img.max(axis=1)
    img_feature = numpy.concatenate( (img_mean, img_min, img_max) )

    # Extracting Mel Spectrogram feature
    hog_feature = get_hog_feature(file_path)
    hog_feature_mean = hog_feature.mean(axis=1)
    hog_feature_min = hog_feature.min(axis=1)
    hog_feature_max = hog_feature.max(axis=1)
    hog_feature_feature = numpy.concatenate( (hog_feature_mean, hog_feature_min, hog_feature_max) )

    # Extracting sobel_edges vector feature
    sobel_edges = get_sobel_edges_vector(file_path)
    sobel_edges_mean = sobel_edges.mean(axis=1)
    sobel_edges_min = sobel_edges.min(axis=1)
    sobel_edges_max = sobel_edges.max(axis=1)
    sobel_edges_feature = numpy.concatenate( (sobel_edges_mean, sobel_edges_min, sobel_edges_max) )

    # Extracting tonnetz feature
    contours = get_tonnetz(file_path)
    contours_mean = contours.mean(axis=1)
    contours_min = contours.min(axis=1)
    contours_max = contours.max(axis=1)
    contours_feature = numpy.concatenate( (contours_mean, contours_min, contours_max) ) 

    feature = numpy.concatenate( (sobel_edges_feature, hog_feature_feature, img_feature, contours_feature) )
    return feature

#---------------------------------------------------------------------------------#

object_1 = ['Rock','Phonk','Synthwave','Jazz','EDM','Metal','Nightcore','Dubstep','Score','Frenchcore','Uptempo']
print(len(object_1))
features = []
labels = []

with open('C:\\Users\\kazan\\Videos\\git\\music_genres\\info\\genres_final.txt') as f:
    object_list = f.read().splitlines()

# Путь к папке с аудиофайлами
pictures_folder = 'C:\\Users\\kazan\\Videos\\git\\music_genres\\info\\music_WAV'

# Список файлов в папке
pictures_files = os.listdir(pictures_folder)

# Инициализация списков признаков и меток жанров

# Перебор каждого файла в папке
for a, pictures_file in enumerate(pictures_files):
    file_path = 'C:\\Users\\kazan\\Videos\\git\\music_genres\\info\\music_WAV\\' + pictures_file

    features.append(get_feature(file_path))
    labels.append(object_1.index(object_list[a]))
    print(pictures_file, object_list[a], object_1.index(object_list[a]))

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
x = keras.layers.Dense(350, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(256, activation="relu", name="dense_2")(x)
x = keras.layers.Dense(128, activation="relu", name="dense_3")(x)
x = keras.layers.Dense(64, activation="relu", name="dense_4")(x)
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
print(object_1[ind], '=> Phonk (MUKBANG)')

file_path = "C:\\Users\\kazan\\Desktop\\2.wav"
feature = get_feature(file_path)
y = model.predict(feature.reshape(1,498))
ind = numpy.argmax(y)
print(object_1[ind], '=> ?El Tigro')

file_path = "C:\\Users\\kazan\\Desktop\\3.mp4.wav"
feature = get_feature(file_path)
y = model.predict(feature.reshape(1,498))
ind = numpy.argmax(y)
print(object_1[ind], '=> EDM (Nana)')

file_path = "C:\\Users\\kazan\\Desktop\\1.wav"
feature = get_feature(file_path)
y = model.predict(feature.reshape(1,498))
ind = numpy.argmax(y)
print(object_1[ind], '=> ? (Been Good To Know Ya)')

file_path = "C:\\Users\\kazan\\Desktop\\2.mp4.wav"
feature = get_feature(file_path)
y = model.predict(feature.reshape(1,498))
ind = numpy.argmax(y)
print(object_1[ind], '=> EDM (Zenith)')

'''
example_file = "C:\\Users\\kazan\\Videos\\git\\music_genres\\info\\music_WAV\\Cyberpunk_2077_-_The_Ballad_of_Buck_Ravers_by_SAMURAI_Refused_69859526(rock).wav"
img = get_img(example_file)
pyplot.imshow(img, interpolation='nearest', aspect='auto')
pyplot.show()

hog_feature = get_hog_feature(example_file)
pyplot.imshow(hog_feature, interpolation='nearest', aspect='auto')
pyplot.show()

sobel_edges = get_sobel_edges_vector(example_file)
pyplot.imshow(sobel_edges, interpolation='nearest', aspect='auto')
pyplot.show()

contours = get_tonnetz(example_file)
pyplot.imshow(contours , interpolation='nearest', aspect='auto')
pyplot.show()
'''