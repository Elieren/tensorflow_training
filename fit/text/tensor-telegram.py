from dotenv.main import load_dotenv
import telebot
import os
import sqlite3


def load_news(value):
    connect = sqlite3.connect('base.db')  # Database creation
    cursor = connect.cursor()

    cursor.execute('SELECT text FROM text_ai')
    result = cursor.fetchall()
    data = []
    for x in result:
        data.append(x[0])

    cursor.close()
    connect.close()

    return data


def load_db(id_tele):
    connect = sqlite3.connect('base.db')
    cursor = connect.cursor()

    id_tele = 'id_' + str(id_tele)

    cursor.execute(f'SELECT {id_tele} FROM text_ai;')

    result = cursor.fetchall()

    cursor.close()
    connect.close()
    key = 0
    for i, x in enumerate(result):
        if x[0] == None:
            key = 1
            break

    if key == 0:
        i = 'stop'

    return i


def article_text(data, quantity, index):
    try:
        i = data[index]
        text = f'{i}\ntext: {index + 1}/{quantity - 1}'
        return text
    except Exception:
        return 'end'


def write_in_db(id_tele, text, i, ids):
    id_tele = 'id_' + str(id_tele)
    with sqlite3.connect('base.db') as connect:
        cursor = connect.cursor()
        cursor.execute(f"UPDATE text_ai SET {id_tele} = {i} WHERE id = {ids};")
        connect.commit()


def user_db(id_tele):
    id_tele = 'id_' + str(id_tele)
    with sqlite3.connect('base.db') as connect:
        cursor = connect.cursor()
        cursor.execute("SELECT * FROM text_ai")
        columns = [description[0] for description in cursor.description]

        if id_tele not in columns:
            cursor.execute(f"ALTER TABLE text_ai ADD COLUMN {id_tele} int;")
        connect.commit()


load_dotenv()
token = os.environ['TOKEN']
quantity = int(os.environ['VALUE'])

return_data_news = load_news(quantity)

bot = telebot.TeleBot(token)
markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
itembtn1 = telebot.types.KeyboardButton('Интересно')
itembtn2 = telebot.types.KeyboardButton('Неинтересно')
markup.add(itembtn1, itembtn2)


@bot.message_handler(commands=['start'])
def send_welcome(message):
    # bot.send_message(message.chat.id, "Добрый день.")
    user_db(message.chat.id)
    return_data_db = load_db(message.chat.id)
    if return_data_db == 0:
        value = 0
    else:
        value = 1
    index = 0
    text = ''
    if value == 0:
        text = article_text(return_data_news, quantity, 0)
    elif value == 1:
        index = return_data_db
        text = article_text(return_data_news, quantity, index)

    bot.reply_to(message, text, reply_markup=markup)


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    return_data_db = load_db(message.chat.id)
    if return_data_db != 'stop':
        index = return_data_db
        text = article_text(return_data_news, quantity, index)
        if (message.text == 'Интересно') or (message.text == 'Неинтересно'):
            if message.text == 'Интересно':
                write_in_db(message.chat.id, text, 1, index + 1)
            elif message.text == 'Неинтересно':
                write_in_db(message.chat.id, text, 0, index + 1)

            index += 1
            text = article_text(return_data_news, quantity, index)
            if text == 'end':
                text = 'На этом всё. Спасибо за участие.'
        else:
            text = article_text(return_data_news, quantity, index)
            text = "Не верное значение. Введите ещё раз.\n\n" + text
        bot.send_message(message.chat.id, text, reply_markup=markup)


# Запускаем телеграм-бота
print('SERVER START')
bot.polling(none_stop=True)
