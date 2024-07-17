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

import boto3
from botocore.client import Config
from dotenv import load_dotenv


load_dotenv()

# ------------------------AWS-S3-Pictures-------------------------#

endpoint_url = os.environ['URL_HOST']
aws_access_key_id = os.environ['ACCESS_KEY']
aws_secret_access_key = os.environ['SECRET_ACCESS_KEY']
bucket = os.environ['BUCKET']

#  ---------------------------------------------------------------#

s3 = boto3.client('s3',
                  endpoint_url=endpoint_url,
                  aws_access_key_id=aws_access_key_id,
                  aws_secret_access_key=aws_secret_access_key,
                  config=Config(signature_version='s3v4'))

scale = 128


def load_image(img):
    global scale
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
    img = cv2.resize(img, (scale, scale))
    filterd_image = cv2.medianBlur(img, 7)
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
        (img.shape[0], img.shape[1]))
        )
    cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 1)
    img_contours = img_contours.reshape(scale, scale, 1)
    return img_contours

# ----------------------------------------------------------------------------------#


def get_feature(key, X, i):

    image_object = s3.get_object(Bucket=bucket, Key=key)
    image_data = image_object['Body'].read()

    img = Image.open(BytesIO(image_data))

    # Extracting img feature
    img_gray = load_image(img)

    # Extracting Mel Spectrogram feature
    hog_feature = get_hog_feature(img)

    # Extracting sobel_edges vector feature
    sobel_edges = get_sobel_edges(img)

    features = numpy.concatenate((img_gray, hog_feature, sobel_edges), axis=-1)
    X[i, :, :, :] = features
    # return X

# ---------------------------------------------------------------------------------#


files = []
objects = []

labels = []


paginator = s3.get_paginator('list_objects')

for page in paginator.paginate(Bucket=bucket):
    # print(page)
    _ = [files.append(s['Key']) for s in page['Contents']]


quantity_image = len(files)

X = numpy.zeros((quantity_image, scale, scale, 3))

_ = [objects.append(x.split('/')[0])
     for x in files if x.split('/')[0] not in objects]

for i, x in enumerate(files):
    genre = x.split('/')[0]
    get_feature(x, X, i)
    labels.append(objects.index(genre))
    print(genre, i)


# -------------------------------------------------------------------------#

with open('dataset_db/pictures/dataset_features.dat', 'wb') as file:
    pickle.dump(X, file)

with open('dataset_db/pictures/dataset_labels.dat', 'wb') as file:
    pickle.dump(labels, file)
