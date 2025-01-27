import numpy
import tensorflow
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.feature._hog import hog
from PIL import Image
import cv2
from io import BytesIO
import tensorflow as tf

scale = 128


def load_image(file_bytes):
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


def get_feature(file_path, X):

    img = Image.open(file_path)
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
    X[0, :, :, :] = features

# ---------------------------------------------------------------------------------#


object_1 = ['Cat', 'Dog']

# Создание той же архитектуры модели
loaded_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (2, 2), activation='relu',
                           input_shape=(128, 128, 5)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(2, activation="softmax")
])

# Компиляция модели перед загрузкой весов
loaded_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Загрузка весов
loaded_model.load_weights('model\\pictures\\model_weights.weights.h5')


file_path = "test\pictures\dog.jpg"
X = numpy.zeros((1, scale, scale, 5), dtype=numpy.float32)
get_feature(file_path, X)
feature = X
y = loaded_model.predict(feature)
ind = numpy.argmax(y)
print(object_1[ind], '=> Dog')

file_path = "test\pictures\cat.jpg"
X = numpy.zeros((1, scale, scale, 5), dtype=numpy.float32)
get_feature(file_path, X)
feature = X
y = loaded_model.predict(feature)
ind = numpy.argmax(y)
print(object_1[ind], '=> Cat')
