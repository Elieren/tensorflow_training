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
from tensorflow.keras.models import load_model

def load_image(file_path):
    img = Image.open(file_path)
    img = img.convert('L') # конвертация в оттенки серого
    buf = BytesIO() # создание байтового буфера
    img.save(buf, format='JPEG') # сохранение изображения в формате JPEG
    file_bytes = buf.getvalue() # получение байтов из буфера
    img = cv2.imdecode(numpy.frombuffer(file_bytes, numpy.uint8), cv2.IMREAD_GRAYSCALE) # декодирование JPEG в изображение cv2 в оттенках серого
    img = tensorflow.image.convert_image_dtype(img, tensorflow.float64) # преобразование изображения в формат float64
    img = numpy.array(img)
    return img

# Функция для вычисления гистограммы направленных градиентов (HOG)
def get_hog_feature(img):
    img = Image.open(img)
    buf = BytesIO() # создание байтового буфера
    img.save(buf, format='JPEG') # сохранение изображения в формате JPEG
    file_bytes = buf.getvalue() # получение байтов из буфера
    img = cv2.imdecode(numpy.frombuffer(file_bytes, numpy.uint8), cv2.IMREAD_COLOR) # декодирование JPEG в изображение cv2 (цветное)
    img = tensorflow.image.convert_image_dtype(img, tensorflow.float64) # преобразование изображения в формат float64
    hog_feature, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), visualize=True, channel_axis=2)
    hog_image = numpy.array(hog_image)
    return hog_image

# Функция для вычисления границ объектов с помощью оператора Собеля
def get_sobel_edges(img):
    img = Image.open(img)
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
    hog_feature = get_hog_feature(file_path)
    hog_feature_mean = hog_feature.mean(axis=1)
    hog_feature_min = hog_feature.min(axis=1)
    hog_feature_max = hog_feature.max(axis=1)
    hog_feature_feature = numpy.concatenate( (hog_feature_mean, hog_feature_min, hog_feature_max) )

    # Extracting sobel_edges vector feature
    sobel_edges = get_sobel_edges(file_path)
    sobel_edges_mean = sobel_edges.mean(axis=1)
    sobel_edges_min = sobel_edges.min(axis=1)
    sobel_edges_max = sobel_edges.max(axis=1)
    sobel_edges_feature = numpy.concatenate( (sobel_edges_mean, sobel_edges_min, sobel_edges_max) )

    # Extracting tonnetz feature
    contours = get_contours(file_path)
    contours_array = numpy.array(contours)
    contours_mean = contours_array.mean(axis=0)
    contours_min = contours_array.min(axis=0)
    contours_max = contours_array.max(axis=0)
    contours_feature = numpy.concatenate( (contours_mean, contours_min, contours_max) ) 

    feature = numpy.concatenate( (img_feature, sobel_edges_feature, hog_feature_feature, contours_feature) )
    return feature

#---------------------------------------------------------------------------------#

object_1 = ['Cat','Dog','Mouse','Snake']

loaded_model = load_model('my_model.h5')

file_path = "C:\\Users\\kazan\\Desktop\\1.jpg"
img = io.imread(file_path)
print(img.shape)
feature = get_feature(file_path)
y = loaded_model.predict(feature.reshape(1,1536))
ind = numpy.argmax(y)
print(object_1[ind], '=> Dog')

file_path = "C:\\Users\\kazan\\Desktop\\2.jpg"
img = io.imread(file_path)
print(img.shape)
feature = get_feature(file_path)
y = loaded_model.predict(feature.reshape(1,1536))
ind = numpy.argmax(y)
print(object_1[ind], '=> Cat')

file_path = "C:\\Users\\kazan\\Desktop\\4.jpg"
img = io.imread(file_path)
print(img.shape)
feature = get_feature(file_path)
y = loaded_model.predict(feature.reshape(1,1536))
ind = numpy.argmax(y)
print(object_1[ind], '=> Shake')