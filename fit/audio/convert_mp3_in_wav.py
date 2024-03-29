import os
from pydub import AudioSegment

# Пути к папке с файлами MP3 и папке, куда сохранять файлы WAV
mp3_folder = '12'
wav_folder = '13'

# Создание папки для сохранения WAV файлов, если она не существует
if not os.path.exists(wav_folder):
    os.makedirs(wav_folder)

# Перебор каждого файла MP3 в указанной папке
for file_name in os.listdir(mp3_folder):
    if file_name.endswith('.mp3'):
        # Загрузка файла MP3 с помощью библиотеки pydub
        pictures = AudioSegment.from_mp3(os.path.join(mp3_folder, file_name))

        # Сохранение файла WAV с тем же названием в другую папку
        file_name = os.path.splitext(file_name)[0] + '.wav'
        pictures.export(os.path.join(wav_folder, file_name), format='wav')
