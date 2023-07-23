import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import sqlite3
import pickle

# Подключение к базе данных
conn = sqlite3.connect('./dataset_db/text/base.db')
cursor = conn.cursor()
cursor.execute("SELECT text, class FROM text_ai")
texts = []
labels = []
for row in cursor.fetchall():
    text = row[0]
    label = row[1]
    texts.append(text)
    labels.append(label)
cursor.close()
conn.close()


text = []
for i in texts:
    i = i.replace('\xa0', ' ').lower()
    text.append(i)

# Предварительная обработка и разделение данных
# Замените 'texts' и 'labels' на вашу переменную, содержащую тексты и метки из базы данных
# Осуществите предварительную обработку данных по своему усмотрению
# и разделите данные на обучающие и тестовые выборки

X_train, X_other, y_train, y_other = train_test_split(text, labels, test_size=0.2, random_state=42)

# Разделение остатка на test и val
X_test, X_val, y_test, y_val = train_test_split(X_other, y_other, test_size=0.5, random_state=42)

# Создание словаря
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1  # размер словаря

with open('./dataset_db/text/word_index.pickle', 'wb') as handle:
    pickle.dump(tokenizer.word_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Преобразование текста в последовательности чисел
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)
sequences_val = tokenizer.texts_to_sequences(X_val) ##############

y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.int32) #####################

# Представление последовательностей фиксированной длины
max_length = 120  # максимальная длина последовательности
X_train_padded = pad_sequences(sequences_train, maxlen=max_length, padding='pre')
X_test_padded = pad_sequences(sequences_test, maxlen=max_length, padding='pre')
X_val_padded = pad_sequences(sequences_val, maxlen=max_length, padding='pre') ###################################

# Создание модели TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 50, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Тренировка модели
model.fit(X_train_padded, y_train, epochs=10, validation_data=(X_test_padded, y_test))

score = model.evaluate(x=X_val_padded,y=y_val, verbose=0)
print(score)
print('Accuracy : ' + str(score[1]*100) + '%')

# Сохранение модели
model.save('./model/text/my_model.h5')