from aiogram.dispatcher.filters.state import State, StatesGroup


class States(StatesGroup):
    work = State()
    image = State()
    validating = State()
    correcting = State()
