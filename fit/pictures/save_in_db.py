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

from skimage.transform import rotate
from skimage.transform import warp, SimilarityTransform

# import matplotlib.pyplot as plt


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

# ----------------------------------------------------------------------------------#


def random_rotate(img):
    img = numpy.array(img)
    angle = numpy.random.uniform(-35, 35)
    rotated_img = rotate(img, angle)
    return Image.fromarray((rotated_img * 255).astype(numpy.uint8))


def random_shift(img):
    img = numpy.array(img)
    # Измените масштабы сдвига, если нужно
    # Базовое значение сдвига (20% от размера изображения)
    shift_y, shift_x = numpy.array(img.shape[:2]) / 5

    # Рандомизация величины сдвига (от -shift до +shift)
    shift_y = numpy.random.uniform(-shift_y, shift_y)
    shift_x = numpy.random.uniform(-shift_x, shift_x)

    tf_shift = SimilarityTransform(translation=(shift_x, shift_y))
    # Здесь мы используем mode='wrap', чтобы части изображения перемещались на другую сторону
    shifted_img = warp(img, tf_shift, mode='wrap')
    return Image.fromarray((shifted_img * 255).astype(numpy.uint8))


# ----------------------------------------------------------------------------------#


def load_image(file_bytes):
    # декодирование JPEG в изображение cv2 в оттенках серого
    img = cv2.imdecode(numpy.frombuffer(file_bytes, numpy.uint8),
                       cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # преобразование изображения в формат float64
    img = tensorflow.image.convert_image_dtype(img, tensorflow.float32)
    img = numpy.array(img)
    # plt.imshow(img)
    # plt.show()
    return img


# Функция для вычисления гистограммы направленных градиентов (HOG)
def get_hog_feature(file_bytes):
    # декодирование JPEG в изображение cv2 (цветное)
    img = cv2.imdecode(numpy.frombuffer(file_bytes, numpy.uint8),
                       cv2.IMREAD_COLOR)
    # преобразование изображения в формат float64
    img = tensorflow.image.convert_image_dtype(img, tensorflow.float32)
    hog_feature, hog_image = hog(img,
                                 orientations=9, pixels_per_cell=(8, 8),
                                 cells_per_block=(2, 2), visualize=True,
                                 channel_axis=2)
    hog_image = numpy.array(hog_image)
    return hog_image


# Функция для вычисления границ объектов с помощью оператора Собеля
def get_sobel_edges(file_bytes):
    # декодирование JPEG в изображение cv2 (цветное)
    img = cv2.imdecode(numpy.frombuffer(file_bytes, numpy.uint8),
                       cv2.IMREAD_COLOR)
    # преобразование изображения в формат float64
    img = tensorflow.image.convert_image_dtype(img, tensorflow.float32)
    sobel_edges = sobel(rgb2gray(img))
    sobel_edges = numpy.array(sobel_edges)
    return sobel_edges


# ----------------------------------------------------------------------------------#


def get_feature(key, X, i):

    image_object = s3.get_object(Bucket=bucket, Key=key)
    image_data = image_object['Body'].read()

    img_orig = Image.open(BytesIO(image_data))

    # Добавляем случайные повороты и сдвиги
    img_rotated1 = random_rotate(img_orig)
    # plt.imshow(img_rotated1)
    # plt.show()
    img_rotated2 = random_rotate(img_orig)
    img_shifted1 = random_shift(img_orig)
    # plt.imshow(img_shifted1)
    # plt.show()
    img_shifted2 = random_shift(img_orig)

    for idx, img in enumerate([img_orig, img_rotated1,
                               img_rotated2, img_shifted1,
                               img_shifted2]):

        img = img.resize((scale, scale))
        buf = BytesIO()  # создание байтового буфера
        img.save(buf, format='JPEG')  # сохранение изображения в формате JPEG
        file_bytes = buf.getvalue()  # получение байтов из буфера

        # Extracting img feature
        img_color = load_image(file_bytes)

        # Extracting Mel Spectrogram feature
        hog_feature = get_hog_feature(file_bytes)

        # Extracting sobel_edges vector feature
        sobel_edges = get_sobel_edges(file_bytes)

        hog_feature = numpy.expand_dims(hog_feature, axis=-1)
        sobel_edges = numpy.expand_dims(sobel_edges, axis=-1)

        features = numpy.concatenate(
            (img_color, hog_feature, sobel_edges), axis=-1)
        X[i * 5 + idx, :, :, :] = features

# ---------------------------------------------------------------------------------#


files = []
objects = []

labels = []


paginator = s3.get_paginator('list_objects')

for page in paginator.paginate(Bucket=bucket):
    # print(page)
    _ = [files.append(s['Key']) for s in page['Contents']]


quantity_image = len(files) * 5

X = numpy.zeros((quantity_image, scale, scale, 5), dtype=numpy.float32)

_ = [objects.append(x.split('/')[0])
     for x in files if x.split('/')[0] not in objects]

for i, x in enumerate(files):
    genre = x.split('/')[0]
    get_feature(x, X, i)

    for _ in range(5):
        labels.append(objects.index(genre))
    print(genre, i)


# -------------------------------------------------------------------------#

with open('dataset_db/pictures/dataset_features_new(128_5_15615).dat', 'wb') as file:
    pickle.dump(X, file)

with open('dataset_db/pictures/dataset_labels_new(128_5_15615).dat', 'wb') as file:
    pickle.dump(labels, file)

# -------------------------------------------------------------------------#


# dump(X, 'dataset_db/pictures/dataset_features_new(640_1).joblib')

# dump(labels, 'dataset_db/pictures/dataset_labels_new(640_1).joblib')
