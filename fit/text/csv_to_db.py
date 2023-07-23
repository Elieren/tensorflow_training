import sqlite3
import csv
import random


value = 10000 + 1

def a(data):
    for i in data:
        print(i)
        print(f'test: {data_1.index(i)}/{value - 1}\n')

        while True:
            text = input(': ')
            if ((int(text) == 0) or (int(text) == 1)):
                cursor.execute("INSERT INTO text_ai (text, class) VALUES (?, ?)",(i, text))
                connect.commit()
                break
            else:
                print('\n', i)
                print(f'test: {data_1.index(i)}/{value - 1}\n')

        print('-' * 100)

data = []
with open('./info/lenta-ru-news.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for x, i in enumerate(reader):
        if x < value:
            data.append(i[1])
        else:
            break
data_1 = data
data = data[1:]

connect = sqlite3.connect('./dataset_db/text/base.db')
cursor = connect.cursor()

cursor.execute('SELECT * FROM text_ai;')

result = cursor.fetchall()

if len(result) == 0:

    a(data)

else:
    x = len(result)
    data = data[x:]

    a(data)

cursor.close()
connect.close()