import os
import numpy
from tensorflow import keras
import tensorflow
import librosa
from matplotlib import pyplot
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.feature._hog import hog
from skimage.measure import find_contours
from PIL import Image
import cv2
from io import BytesIO

def load_image(file_path):
    img = Image.open(file_path)
    img = img.convert('L') # конвертация в оттенки серого
    img = img.resize((60, 60)) # изменение размера до 100x100
    buf = BytesIO() # создание байтового буфера
    img.save(buf, format='JPEG') # сохранение изображения в формате JPEG
    file_bytes = buf.getvalue() # получение байтов из буфера
    img = cv2.imdecode(numpy.frombuffer(file_bytes, numpy.uint8), cv2.IMREAD_GRAYSCALE) # декодирование JPEG в изображение cv2 в оттенках серого
    img = tensorflow.image.convert_image_dtype(img, tensorflow.float64) # преобразование изображения в формат float64
    img = numpy.array(img)
    return img

# Функция для вычисления гистограммы направленных градиентов (HOG)
#def get_hog_feature(img):
    img = io.imread(file_path)
    hog_feature, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), visualize=True)
    return hog_feature

# Функция для вычисления границ объектов с помощью оператора Собеля
def get_sobel_edges(img):
    img = Image.open(file_path)
    img = img.resize((60, 60)) # изменение размера до 100x100
    buf = BytesIO() # создание байтового буфера
    img.save(buf, format='JPEG') # сохранение изображения в формате JPEG
    file_bytes = buf.getvalue() # получение байтов из буфера
    img = cv2.imdecode(numpy.frombuffer(file_bytes, numpy.uint8), cv2.IMREAD_COLOR) # декодирование JPEG в изображение cv2 (цветное)
    img = tensorflow.image.convert_image_dtype(img, tensorflow.float64) # преобразование изображения в формат float64
    sobel_edges = sobel(rgb2gray(img))
    sobel_edges = numpy.array(sobel_edges)
    return sobel_edges

# Функция для вычисления контуров объектов 
def get_contours(img):
    my_photo = cv2.imread(img)
    scale_percent = 99
    k = [60, 60]
    width = int(my_photo.shape[1] * scale_percent / 100)
    height = int(my_photo.shape[0] * scale_percent / 100)
    while True:
        if (k[0] < width) or (k[1] < height):
            scale_percent -= 1
            width = int(my_photo.shape[1] * scale_percent / 100)
            height = int(my_photo.shape[0] * scale_percent / 100)
        else:
            break
    dim = (width, height)
    my_photo = cv2.resize(my_photo, dim)
    filterd_image  = cv2.medianBlur(my_photo,7)
    img_grey = cv2.cvtColor(filterd_image,cv2.COLOR_BGR2GRAY)
    #set a thresh
    thresh = 100
    #get threshold image
    ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    #find contours
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #create an empty image for contours
    img_contours = numpy.uint8(numpy.zeros((my_photo.shape[0],my_photo.shape[1])))
    cv2.drawContours(img_contours, contours, -1, (255,255,255), 1)
    return img_contours

#----------------------------------------------------------------------------------#

def get_feature(file_path):
    # Extracting img feature
    img = load_image(file_path)
    img_mean = img.mean(axis=1)
    img_min = img.min(axis=1)
    img_max = img.max(axis=1)
    img_feature = numpy.concatenate( (img_mean, img_min, img_max) )

    # Extracting Mel Spectrogram feature
    #hog_feature = get_hog_feature(file_path)
    #hog_feature_mean = hog_feature.mean(axis=1)
    #hog_feature_min = hog_feature.min(axis=1)
    #hog_feature_max = hog_feature.max(axis=1)
    #hog_feature_feature = numpy.concatenate( (hog_feature_mean, hog_feature_min, hog_feature_max) )

    # Extracting sobel_edges vector feature
    sobel_edges = get_sobel_edges(file_path)
    sobel_edges_mean = sobel_edges.mean(axis=1)
    sobel_edges_min = sobel_edges.min(axis=1)
    sobel_edges_max = sobel_edges.max(axis=1)
    sobel_edges_feature = numpy.concatenate( (sobel_edges_mean, sobel_edges_min, sobel_edges_max) )

    # Extracting tonnetz feature
    #contours = get_contours(file_path)
    #contours_array = numpy.array(contours)
    #contours_mean = contours_array.mean(axis=0)
    #contours_min = contours_array.min(axis=0)
    #contours_max = contours_array.max(axis=0)
    #contours_feature = numpy.concatenate( (contours_mean, contours_min, contours_max) ) 

    feature = numpy.concatenate( (img_feature, sobel_edges_feature) )
    return feature

#---------------------------------------------------------------------------------#

object_1 = ['Cat','Dog','Mouse','Snake']
print(len(object_1))
features = []
labels = []

# Путь к папке с аудиофайлами
audio_folder = 'C:\\Users\\kazan\\Videos\\git\\music_genres\\info\\Pictures'

# Список файлов в папке
audio_files = os.listdir(audio_folder)

# Инициализация списков признаков и меток жанров

# Перебор каждого файла в папке
for genre_folder in os.listdir(audio_folder):
    genre_path = os.path.join(audio_folder, genre_folder)
    if os.path.isdir(genre_path) and any(substring in genre_folder for substring in genres_1):
        # Перебор каждого WAV файла в папке-жанре
        for audio_file in os.listdir(genre_path):
            if audio_file.endswith('.jpg'):
                # Добавление признаков и метки жанра в соответствующие списки
                genre = [substring for substring in genres_1 if substring in genre_folder][0]
                features.append(get_feature(os.path.join(genre_path, audio_file)))
                labels.append(genres_1.index(genre))

#-------------------------------------------------------------------------#

# Предполагая, что `features` - это список списков разной длины
max_length = max(len(x) for x in features)
features_array = numpy.zeros((len(features), max_length), dtype=numpy.float64)
for i, row in enumerate(features):
    features_array[i,:len(row)] = row

permutations = numpy.random.permutation(len(features))
features = features_array[permutations]

labels = numpy.array(labels)[permutations]

features_train = features[0:50]
labels_train = labels[0:50]

features_val = features[50:65]
labels_val = labels[50:65]

features_test = features[65:85]
labels_test = labels[65:85]

inputs = keras.Input(shape=(360), name="feature")
x = keras.layers.Dense(1024, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(512, activation="relu", name="dense_2")(x)
x = keras.layers.Dense(350, activation="relu", name="dense_3")(x)
x = keras.layers.Dense(256, activation="relu", name="dense_4")(x)
x = keras.layers.Dense(128, activation="relu", name="dense_5")(x)
x = keras.layers.Dense(64, activation="relu", name="dense_6")(x)
x = keras.layers.Dense(32, activation="relu", name="dense_7")(x)
outputs = keras.layers.Dense(4, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    # Optimizer
    optimizer=keras.optimizers.RMSprop(),
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

features_train = numpy.array(features_train)
labels_train = numpy.array(labels_train)

features_train = tensorflow.convert_to_tensor(features_train, dtype=tensorflow.float32)
labels_train = tensorflow.convert_to_tensor(labels_train, dtype=tensorflow.float32)

model.fit(x=features_train,y=labels_train,verbose=1,validation_data=(features_val , labels_val), epochs=1000)

score = model.evaluate(x=features_test,y=labels_test, verbose=0)
print(score)
print('Accuracy : ' + str(score[1]*100) + '%')


file_path = "C:\\Users\\kazan\\Desktop\\Dog\\0.jpg"
feature = get_feature(file_path)
y = model.predict(feature.reshape(1,360))
ind = numpy.argmax(y)
print(object_1[ind], '=> Dog')

file_path = "C:\\Users\\kazan\\Desktop\\1.jpg"
feature = get_feature(file_path)
y = model.predict(feature.reshape(1,360))
ind = numpy.argmax(y)
print(object_1[ind], '=> Cat')

file_path = "C:\\Users\\kazan\\Desktop\\Snake\\26.jpg"
feature = get_feature(file_path)
y = model.predict(feature.reshape(1,360))
ind = numpy.argmax(y)
print(object_1[ind], '=> Shake')