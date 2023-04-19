import cv2
import pixellib
from pixellib.instance import instance_segmentation
from PIL import Image
import cv2
from io import BytesIO
import numpy
from tensorflow import keras
import tensorflow

def load_image(file_path):
    global scale
    img = cv2.imread(file_path)
    height, width, channels = img.shape

    # задание новых размеров картинки
    new_height = 550
    new_width = int(new_height * width / height)

    # изменение размера картинкиs
    img = cv2.resize(img, (new_width, new_height))
    return img

# Загрузка изображения
image = load_image('C:\\Users\\kazan\\Desktop\\3.jpg')

# Загрузка натренированной модели нейронной сети из файла .h5
model = instance_segmentation()

# Загрузка весов модели из файла .h5
model.load_model('fit\\pictures\\100_my_model.h5')

# Обнаружение объектов на изображении
result = model.segmentFrame(image, show_bboxes=True)

# Отображение изображения с выделенными объектами и рамками вокруг них
image = result[1]
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()