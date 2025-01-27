import os
import numpy
import librosa
import pickle
import soundfile as sf
from io import BytesIO

import boto3
from botocore.client import Config
from dotenv import load_dotenv


load_dotenv()

# ------------------------AWS-S3-Pictures-------------------------#

endpoint_url = os.environ['URL_HOST']
aws_access_key_id = os.environ['ACCESS_KEY']
aws_secret_access_key = os.environ['SECRET_ACCESS_KEY']
bucket = "music-dataset"

#  ---------------------------------------------------------------#

botocore_config = Config(
    read_timeout=120,  # Увеличение времени ожидания
    retries={
        'max_attempts': 10,  # Увеличение количества попыток
    },
    signature_version='s3v4'  # Добавляем параметр версии подписи
)

s3 = boto3.client('s3',
                  endpoint_url=endpoint_url,
                  aws_access_key_id=aws_access_key_id,
                  aws_secret_access_key=aws_secret_access_key,
                  config=botocore_config)


def get_mfcc(y, sr):
    mfcc = numpy.array(librosa.feature.mfcc(y=y, sr=sr))
    return mfcc


def get_melspectrogram(y, sr):
    melspectrogram = numpy.array(librosa.feature.melspectrogram(y=y, sr=sr))
    return melspectrogram


def get_chroma_vector(y, sr):
    chroma = numpy.array(librosa.feature.chroma_stft(y=y, sr=sr))
    return chroma


def get_tonnetz(y, sr):
    tonnetz = numpy.array(librosa.feature.tonnetz(y=y, sr=sr))
    return tonnetz


def get_feature(key):

    audio_object = s3.get_object(Bucket=bucket, Key=key)
    audio_data = BytesIO()

    # Чтение данных по частям
    for chunk in audio_object['Body'].iter_chunks(chunk_size=1024):
        audio_data.write(chunk)

    audio_data.seek(0)

    data, samplerate = sf.read(audio_data)
    audio_data.seek(0)
    y, sr = librosa.load(audio_data, sr=samplerate, mono=True)

    # Extracting MFCC feature
    mfcc = get_mfcc(y, sr)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_min = mfcc.min(axis=1)
    mfcc_max = mfcc.max(axis=1)
    mfcc_feature = numpy.concatenate((mfcc_mean, mfcc_min, mfcc_max))

    # Extracting Mel Spectrogram feature
    melspectrogram = get_melspectrogram(y, sr)
    melspectrogram_mean = melspectrogram.mean(axis=1)
    melspectrogram_min = melspectrogram.min(axis=1)
    melspectrogram_max = melspectrogram.max(axis=1)
    melspectrogram_feature = numpy.concatenate((melspectrogram_mean,
                                                melspectrogram_min,
                                                melspectrogram_max))

    # Extracting chroma vector feature
    chroma = get_chroma_vector(y, sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_min = chroma.min(axis=1)
    chroma_max = chroma.max(axis=1)
    chroma_feature = numpy.concatenate((chroma_mean, chroma_min, chroma_max))

    # Extracting tonnetz feature
    tntz = get_tonnetz(y, sr)
    tntz_mean = tntz.mean(axis=1)
    tntz_min = tntz.min(axis=1)
    tntz_max = tntz.max(axis=1)
    tntz_feature = numpy.concatenate((tntz_mean, tntz_min, tntz_max))

    feature = numpy.concatenate((chroma_feature, melspectrogram_feature,
                                 mfcc_feature, tntz_feature))
    return feature

# ---------------------------------------------------------------------------------#


files = []
objects = []

features = []
labels = []


paginator = s3.get_paginator('list_objects')

for page in paginator.paginate(Bucket=bucket):
    # print(page)
    _ = [files.append(s['Key']) for s in page['Contents']]


_ = [objects.append(x.split('/')[0])
     for x in files if x.split('/')[0] not in objects]

for x in files:
    try:
        genre = x.split('/')[0]
        features.append(get_feature(x))
        labels.append(objects.index(genre))
        print(f"🟢 {genre} --- {x.split('/')[1]}")
    except Exception as e:
        print(e)

# -------------------------------------------------------------------------#

with open('dataset_db/audio/dataset_features.dat', 'wb') as file:
    pickle.dump(features, file)

with open('dataset_db/audio/dataset_labels.dat', 'wb') as file:
    pickle.dump(labels, file)
