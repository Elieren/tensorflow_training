import pickle
import os
import numpy
import tensorflow
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.feature._hog import hog
from PIL import Image
import cv2
from io import BytesIO

scale = 128
quantity_image = 3141  # 3141


def load_image(file_path):
    global scale
    img = Image.open(file_path)
    img = img.convert('L')  # конвертация в оттенки серого
    img = img.resize((scale, scale))
    buf = BytesIO()  # создание байтового буфера
    img.save(buf, format='JPEG')  # сохранение изображения в формате JPEG
    file_bytes = buf.getvalue()  # получение байтов из буфера
    # декодирование JPEG в изображение cv2 в оттенках серого
    img = cv2.imdecode(numpy.frombuffer(file_bytes, numpy.uint8),
                       cv2.IMREAD_GRAYSCALE)
    # преобразование изображения в формат float64
    img = tensorflow.image.convert_image_dtype(img, tensorflow.float32)
    img = numpy.array(img)
    img = img / 255
    img = img.reshape(scale, scale, 1)
    return img


# Функция для вычисления гистограммы направленных градиентов (HOG)
def get_hog_feature(img):
    global scale
    img = Image.open(img)
    img = img.resize((scale, scale))
    buf = BytesIO()  # создание байтового буфера
    img.save(buf, format='JPEG')  # сохранение изображения в формате JPEG
    file_bytes = buf.getvalue()  # получение байтов из буфера
    # декодирование JPEG в изображение cv2 (цветное)
    img = cv2.imdecode(numpy.frombuffer(file_bytes, numpy.uint8),
                       cv2.IMREAD_COLOR)
    # преобразование изображения в формат float64
    img = tensorflow.image.convert_image_dtype(img, tensorflow.float32)
    hog_feature, hog_image = hog(img,
                                 orientations=9, pixels_per_cell=(8, 8),
                                 cells_per_block=(2, 2), visualize=True,
                                 channel_axis=2)
    hog_image = hog_image.reshape(scale, scale, 1)
    hog_image = numpy.array(hog_image)
    return hog_image


# Функция для вычисления границ объектов с помощью оператора Собеля
def get_sobel_edges(img):
    global scale
    img = Image.open(img)
    img = img.resize((scale, scale))
    buf = BytesIO()  # создание байтового буфера
    img.save(buf, format='JPEG')  # сохранение изображения в формате JPEG
    file_bytes = buf.getvalue()  # получение байтов из буфера
    # декодирование JPEG в изображение cv2 (цветное)
    img = cv2.imdecode(numpy.frombuffer(file_bytes, numpy.uint8),
                       cv2.IMREAD_COLOR)
    # преобразование изображения в формат float64
    img = tensorflow.image.convert_image_dtype(img, tensorflow.float32)
    sobel_edges = sobel(rgb2gray(img))
    sobel_edges = sobel_edges.reshape(scale, scale, 1)
    sobel_edges = numpy.array(sobel_edges)
    return sobel_edges


# Функция для вычисления контуров объектов
def get_contours(img):
    global scale
    my_photo = cv2.imread(img)
    my_photo = cv2.resize(my_photo, (scale, scale))
    filterd_image = cv2.medianBlur(my_photo, 7)
    img_grey = cv2.cvtColor(filterd_image, cv2.COLOR_BGR2GRAY)
    # set a thresh
    thresh = 100
    # get threshold image
    ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    # find contours
    contours, hierarchy = cv2.findContours(thresh_img,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # create an empty image for contours
    img_contours = numpy.uint8(numpy.zeros(
        (my_photo.shape[0], my_photo.shape[1]))
        )
    cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 1)
    img_contours = img_contours.reshape(scale, scale, 1)
    return img_contours

# ----------------------------------------------------------------------------------#


def get_feature(file_path, X, i):
    # Extracting img feature
    img = load_image(file_path)

    # Extracting Mel Spectrogram feature
    hog_feature = get_hog_feature(file_path)

    # Extracting sobel_edges vector feature
    sobel_edges = get_sobel_edges(file_path)

    features = numpy.concatenate((img, hog_feature, sobel_edges), axis=-1)
    X[i, :, :, :] = features
    return X

# ---------------------------------------------------------------------------------#


object_1 = ['Cat', 'Dog']
print(len(object_1))
features = []
labels = []

# Путь к папке с аудиофайлами
audio_folder = 'info/Pictures'

# Список файлов в папке
audio_files = os.listdir(audio_folder)

# Инициализация списков признаков и меток жанров
X = numpy.zeros((quantity_image, scale, scale, 3))

i = 0
# Перебор каждого файла в папке
for genre_folder in os.listdir(audio_folder):
    genre_path = os.path.join(audio_folder, genre_folder)
    if os.path.isdir(genre_path) and any(substring in genre_folder
                                         for substring in object_1):
        # Перебор каждого WAV файла в папке-жанре
        for audio_file in os.listdir(genre_path):
            if audio_file.endswith('.jpg'):
                # Добавление признаков и метки жанра в соответствующие списки
                genre = [substring for substring in object_1
                         if substring in genre_folder][0]
                get_feature(os.path.join(genre_path, audio_file), X, i)
                labels.append(object_1.index(genre))
                print(genre, i)
                i += 1

# -------------------------------------------------------------------------#

with open('dataset_db/pictures/dataset_features.dat', 'wb') as file:
    pickle.dump(X, file)

with open('dataset_db/pictures/dataset_labels.dat', 'wb') as file:
    pickle.dump(labels, file)
