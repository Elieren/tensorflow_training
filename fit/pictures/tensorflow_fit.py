import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# load features and labels
with open('dataset_db/pictures/dataset_features_new(128_5_15615).dat', 'rb') as file:
    X = pickle.load(file)

with open('dataset_db/pictures/dataset_labels_new(128_5_15615).dat', 'rb') as file:
    y = pickle.load(file)

# convert labels to categorical
y = tf.keras.utils.to_categorical(y, num_classes=2)

# shuffle and split data into train/validation/test sets
permutations = np.random.permutation(len(X))
X = np.array(X)[permutations]
y = np.array(y)[permutations]


X_train, X_other, y_train, y_other = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Разделение остатка на test и val
X_test, X_val, y_test, y_val = train_test_split(
    X_other, y_other, test_size=0.75, random_state=42
)

X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

# define model
model = tf.keras.Sequential([
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

# compile model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# fit model on training set
model.fit(X_train, y_train, epochs=20, batch_size=128,
          validation_data=(X_val, y_val))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.show()

# evaluate model on test set
score = model.evaluate(X_test, y_test)
print(score)
print('Accuracy : ' + str(score[1]*100) + '%')

model.save_weights('model/pictures/model_weights.h5')
