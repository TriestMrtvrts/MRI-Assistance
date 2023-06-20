#!/usr/bin/python3

from aiogram import Bot
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor

from bot.states import States
from handlers import welcome, image_handler, user_agreement, validating_answer_of_model, correcting_answer_from_model

from torch import nn


bot = Bot(token="6045456181:AAEhTGp24GRONqZTtoDk5irffoCLAV_wpIE")
dp = Dispatcher(bot, storage=MemoryStorage())

dp.register_message_handler(welcome, commands=["start"], state="*")
dp.register_message_handler(user_agreement, state=States.work)
dp.register_message_handler(image_handler, state=States.image, content_types=['photo', 'document'])
dp.register_message_handler(validating_answer_of_model, state=States.validating)
dp.register_message_handler(correcting_answer_from_model, state=States.correcting)

if __name__ == "__main__":
    executor.start_polling(dp)
