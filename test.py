import os

# Путь к папке с аудиофайлами
pictures_folder = 'C:\\Users\\kazan\\Videos\\git\\music_genres\\info\\music_WAV'

# Список файлов в папке
pictures_files = os.listdir(pictures_folder)

print(pictures_files)