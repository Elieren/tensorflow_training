from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import pickle

# Загрузка предобученной модели
model = load_model('./model/text/my_model.h5')

with open('./dataset_db/text/word_index.pickle', 'rb') as handle:
    word_index = pickle.load(handle)

# Пример новых текстов для классификации
new_texts = [""]  # Замените на ваши тексты

# Предобработка новых текстов
tokenizer = Tokenizer()
tokenizer.word_index = word_index
new_sequences = tokenizer.texts_to_sequences(new_texts)
max_length = 120  # Размер последовательности, используемый при обучении
new_padded_sequences = pad_sequences(new_sequences, maxlen=max_length,
                                     padding='pre')

# Классификация новых текстов
predictions = model.predict(new_padded_sequences)

# Принятие решения на основе порога (например, 0.5)
predicted_classes = (predictions > 0.5).astype("int32")

# Вывод предсказанных классов
for i, text in enumerate(new_texts):
    print(f"Текст: {text}")
    print(f"Предсказанный класс: {predicted_classes[i]}")
    print("-" * 40)
