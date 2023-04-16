import numpy
from tensorflow import keras
import tensorflow
import pickle

shape = ''

object_1 = ['Cat','Dog','Mouse','Snake']
print(len(object_1))
features = []
labels = []

with open('dataset_features.dat', 'rb') as file:
	features = pickle.load(file)

with open('dataset_labels.dat', 'rb') as file:
	labels = pickle.load(file)

features_train = features[0:78]
labels_train = labels[0:78]

features_val = features[78:156]
labels_val = labels[78:156]

features_test = features[156:186]
labels_test = labels[156:186]

tensorflow.device('/device:GPU:0')

inputs = keras.Input(shape=(shape), name="feature")
x = keras.layers.Dense(8192, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(4096, activation="relu", name="dense_2")(x)
x = keras.layers.Dense(4096, activation="relu", name="dense_3")(x)
x = keras.layers.Dense(1024, activation="relu", name="dense_4")(x)
x = keras.layers.Dense(256, activation="relu", name="dense_5")(x)
x = keras.layers.Dense(350, activation="relu", name="dense_6")(x)
x = keras.layers.Dense(256, activation="relu", name="dense_7")(x)
x = keras.layers.Dense(1024, activation="relu", name="dense_8")(x)
x = keras.layers.Dense(128, activation="relu", name="dense_9")(x)
x = keras.layers.Dense(64, activation="relu", name="dense_10")(x)
x = keras.layers.Dense(32, activation="relu", name="dense_11")(x)
x = keras.layers.Dense(16, activation="relu", name="dense_12")(x)
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

model.save('my_model.h5')