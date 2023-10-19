import numpy
from tensorflow import keras
import tensorflow
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

features = []
labels = []

with open('dataset_db/audio/dataset_features.dat', 'rb') as file:
    features = pickle.load(file)

with open('dataset_db/audio/dataset_labels.dat', 'rb') as file:
    labels = pickle.load(file)

permutations = numpy.random.permutation(len(features))
features = numpy.array(features)[permutations]
labels = numpy.array(labels)[permutations]
# labels = keras.utils.to_categorical(labels, num_classes=128)
# labels = keras.utils.to_categorical(labels, num_classes=128)


features_train, X_other, labels_train, y_other = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Разделение остатка на test и val
features_test, features_val, labels_test, labels_val = train_test_split(
    X_other, y_other, test_size=0.5, random_state=42
)


features_train = numpy.array(features_train)
labels_train = numpy.array(labels_train)

features_train = tensorflow.convert_to_tensor(features_train,
                                              dtype=tensorflow.float32)
labels_train = tensorflow.convert_to_tensor(labels_train,
                                            dtype=tensorflow.int32)

features_val = numpy.array(features_val)
labels_val = numpy.array(labels_val)

features_val = tensorflow.convert_to_tensor(features_val,
                                            dtype=tensorflow.float32)
labels_val = tensorflow.convert_to_tensor(labels_val, dtype=tensorflow.int32)

# tensorflow.device('/device:GPU:0')


model = keras.models.Sequential([
    keras.layers.Dense(350, activation="relu", name="dense_1",
                       input_shape=(498,)),
    keras.layers.Dense(128, activation="relu", name="dense_2"),
    keras.layers.Dense(256, activation="relu", name="dense_3"),
    keras.layers.Dense(128, activation="relu", name="dense_4"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(14, activation="softmax", name="predictions")
])

model.compile(
    # Optimizer
    optimizer='adam',
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=['accuracy'],
)
model.fit(features_train, labels_train,
          validation_data=(features_val, labels_val), verbose=1, epochs=200)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.show()

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.show()

score = model.evaluate(x=features_test.tolist(),
                       y=labels_test.tolist(), verbose=0)
print(score)
print('Accuracy : ' + str(score[1]*100) + '%')

model.save('model/audio/my_model_music.keras')
