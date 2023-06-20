import cv2
import numpy as np
import torch
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

device = 'cpu'


def prepare_image(path):
    img = cv2.cvtColor(cv2.resize(cv2.imread(path, 0), (180, 180)), cv2.COLOR_GRAY2RGB)
    img_tensor = torch.from_numpy(np.array(img).astype(np.float32)).to(device)
    img_tensor = img_tensor.permute(2, 0, 1) / 255

    return img_tensor.unsqueeze(0)


def form_reply_keyboard(buttons_info):
    keyboard = ReplyKeyboardMarkup(one_time_keyboard=True)
    buttons = list(map(lambda x: KeyboardButton(x), buttons_info))

    if len(buttons) > 2:
        for button in buttons:
            keyboard.add(button)
    else:
        keyboard.row(*buttons)

    return keyboard
