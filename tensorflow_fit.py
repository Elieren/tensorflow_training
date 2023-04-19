import numpy
from tensorflow import keras
import tensorflow
import pickle
import matplotlib.pyplot as plt

shape = 480
epoch = 100

features = []
labels = []

with open('dataset_features.dat', 'rb') as file:
	features = pickle.load(file)

with open('dataset_labels.dat', 'rb') as file:
	labels = pickle.load(file)

permutations = numpy.random.permutation(265)
features = numpy.array(features)[permutations]
labels = numpy.array(labels)[permutations]

features_train = features[0:250]
labels_train = labels[0:250]

#features_val = features[126:250]
#labels_val = labels[126:250]

features_test = features[252:265]
labels_test = labels[252:265]

features_train = numpy.array(features_train)
labels_train = numpy.array(labels_train)

features_train = tensorflow.convert_to_tensor(features_train, dtype=tensorflow.float32)
labels_train = tensorflow.convert_to_tensor(labels_train, dtype=tensorflow.float32)

tensorflow.device('/device:GPU:0')


model = keras.models.Sequential([
    keras.layers.Dense(256, activation="relu", name="dense_1",input_shape=(shape,)),
    keras.layers.Dense(128, activation="relu", name="dense_2"),
    keras.layers.Dense(128, activation="relu", name="dense_3"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4, activation="softmax", name="predictions")
])

model.compile(
    # Optimizer
    optimizer='adam',
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=["accuracy"],
)



model.fit(x=features_train,y=labels_train,verbose=1, epochs=epoch)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(model.history.history['loss'])
plt.show()

score = model.evaluate(x=features_test,y=labels_test, verbose=0)
print(score)
print('Accuracy : ' + str(score[1]*100) + '%')