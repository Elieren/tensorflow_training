import numpy
from tensorflow import keras
import tensorflow
import pickle
import matplotlib.pyplot as plt

epoch = 10

features = []
labels = []

with open('dataset_features.dat', 'rb') as file:
	features = pickle.load(file)

with open('dataset_labels.dat', 'rb') as file:
	labels = pickle.load(file)

permutations = numpy.random.permutation(83)
features = numpy.array(features)[permutations]
labels = numpy.array(labels)[permutations]
labels = keras.utils.to_categorical(labels, num_classes=128)
labels = keras.utils.to_categorical(labels, num_classes=128)

features_train = features[0:76]
labels_train = labels[0:76]

features_test = features[76:83]
labels_test = labels[76:83]

features_train = numpy.array(features_train)
labels_train = numpy.array(labels_train)

features_train = tensorflow.convert_to_tensor(features_train, dtype=tensorflow.float32)
labels_train = tensorflow.convert_to_tensor(labels_train, dtype=tensorflow.int32)

#tensorflow.device('/device:GPU:0')


model = keras.models.Sequential([
    keras.layers.Dense(350, activation="relu", name="dense_1", input_shape=(498,)),
    keras.layers.Dense(256, activation="relu", name="dense_2"),
    keras.layers.Dense(128, activation="relu", name="dense_3"),
    keras.layers.Dense(64, activation="relu", name="dense_4"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(14, activation="softmax", name="predictions")
])

model.compile(
    # Optimizer
    optimizer='adam',
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
model.fit(x=features_train.tolist(),y=labels_train.tolist(),verbose=1, epochs=100)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(model.history.history['loss'])

score = model.evaluate(x=features_test.tolist(),y=labels_test.tolist(), verbose=0)
print(score)
print('Accuracy : ' + str(score[1]*100) + '%')

model.save('my_model_music.h5')