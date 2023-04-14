import os

# Путь к папке с аудиофайлами
audio_folder = 'C:\\Users\\kazan\\Videos\\git\\music_genres\\info\\music_WAV'

# Список файлов в папке
audio_files = os.listdir(audio_folder)

print(audio_files)