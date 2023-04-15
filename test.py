import os

genres_1 = ['Rock','Phonk','Synthwave','Jazz','EDM','Metal','Nightcore','Dubstep','Score','Frenchcore','Uptempo']
audio_folder = 'C:\\Users\\kazan\\Videos\\git\\music_genres\\info\\music'

# Инициализация списков признаков и меток жанров
features = []
labels = []

# Перебор каждого WAV файла в папке
# Перебор каждой папки-жанра внутри директории
for genre_folder in os.listdir(audio_folder):
    genre_path = os.path.join(audio_folder, genre_folder)
    if os.path.isdir(genre_path) and any(substring in genre_folder for substring in genres_1):
        # Перебор каждого WAV файла в папке-жанре
        for audio_file in os.listdir(genre_path):
            if audio_file.endswith('.wav'):
                # Добавление признаков и метки жанра в соответствующие списки
                genre = [substring for substring in genres_1 if substring in genre_folder][0]
                print(os.path.join(genre_path, audio_file))
                print(genres_1.index(genre))
