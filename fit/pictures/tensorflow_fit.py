import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import matplotlib.pyplot as plt

# load features and labels
with open('dataset_db/pictures/dataset_features.dat', 'rb') as file:
    X = pickle.load(file)

with open('dataset_db/pictures/dataset_labels.dat', 'rb') as file:
    y = pickle.load(file)

# convert labels to categorical
y = keras.utils.to_categorical(y, num_classes=4)

# shuffle and split data into train/validation/test sets
permutations = np.random.permutation(len(X))
X = np.array(X)[permutations]
y = np.array(y)[permutations]

X_train, y_train = X[:2500], y[:2500]
X_val, y_val = X[2500:3000], y[2500:3000]
X_test, y_test = X[3000:], y[3000:]

# define model
model = keras.Sequential([
    keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(256, 256, 4)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (5, 5), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4, activation="softmax")
])

# compile model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

# fit model on training set
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(model.history.history['loss'])

# evaluate model on test set
score = model.evaluate(X_test, y_test)
print(score)
print('Accuracy : ' + str(score[1]*100) + '%')

model.save('model/pictures/my_model_pictures.h5')