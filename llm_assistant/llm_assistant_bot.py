import telebot
from telebot import types


from model_wrapper import ModelWrapper

"""
get_text_messages - обработка любого текстового сообщения, в том числе того, что отправился при нажатии кнопки.

Методы, реализующие одноименные команды телеграм-боту:
start
help
generate
checkmodel
model
"""

TOKEN = "..."
bot = telebot.TeleBot(TOKEN)

model_wrapper = ModelWrapper() # внутри класса описание

@bot.message_handler(commands=['help'])
def help(message):
    help_message = """Доступны следующие команды:
/start старт бота
/model выбор модели
/checkmodel посмотреть, как модель сейчас загружена
/generate сгенерировать текст по контексту (можно использовать без введения команды)
"""
    bot.send_message(message.from_user.id, help_message)


@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.from_user.id, "Привет! Для знакомства с доступными командами введите /help")


@bot.message_handler(commands=['model'])
def model(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("StatLM")
    btn2 = types.KeyboardButton("GPT")
    btn3 = types.KeyboardButton("Llama")
    markup.add(btn1, btn2, btn3)
    bot.send_message(message.from_user.id, "Выберите модель для генерации", reply_markup=markup)


@bot.message_handler(commands=['checkmodel'])
def checkmodel(message):
    bot.send_message(message.from_user.id, f"Текущая модель: {str(model_wrapper.current_model_name)}")


@bot.message_handler(commands=['generate'])
def generate(message):
    bot.send_message(message.from_user.id,
                     "Введите текст (вопрос, на который нужно ответить, либо текст, который нужно продолжить)")


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    print(f'<{message.text}>')
    if message.text in ['StatLM', 'GPT', 'Llama']:
        print(f'@{message.text}@')
        status, result = model_wrapper.load(message.text, test_inference=True)
        if status:
            bot.send_message(message.from_user.id, "Подгружено")
        else:
            bot.send_message(message.from_user.id, f"Проблемы с загрузкой модели, ниже описаны ошибки.\n{result}")
    else:
        status, result = model_wrapper.generate(message.text)
        if status:
            bot.send_message(message.from_user.id, result)
        else:
            bot.send_message(message.from_user.id, f"Проблемы с генерацией, ниже описаны ошибки.\n{result}")


bot.polling(none_stop=True, interval=0) #обязательная для работы бота часть
# TODO: сделайте логирование запросов с указанием модели и параметров - это полезно
