import os
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops

from config import *


def draw_text(text, font_name, size):
    font_size = 10
    img = Image.new('L', size, color=0)
    font = ImageFont.truetype(f'{FONTS_FOLDER}/{font_name}', font_size)
    draw = ImageDraw.Draw(img)
    w, h = draw.textsize(text, font=font)
    while w < size[0] and h < size[1]:
        font_size += 1
        font = ImageFont.truetype(f'{FONTS_FOLDER}/{font_name}', font_size)
        w, h = draw.textsize(text, font=font)
    x = (size[0] - w) / 2
    y = (size[1] - h) / 2
    draw.text((x, y), text, font=font, fill=255)
    return img


def create_path(path, force_delete=True):
    if force_delete:
        shutil.rmtree(path, ignore_errors=True)
    try:
        os.makedirs(path)
    except Exception:
        pass


def create_images(size=SIZE):
    fonts = os.listdir(FONTS_FOLDER)
    for symbol in TARGET_ARRAY:
        create_path(f'{DATASET_FOLDER}/{symbol}', False)
        for font in fonts:
            items = []
            if type(symbol) is str and (symbol.upper() != symbol or symbol.lower() != symbol):
                items.append({'name': f'{symbol}.lower', 'value': symbol.lower()})
                items.append({'name': f'{symbol}.upper', 'value': symbol.upper()})
            else:
                items.append({'name': f'{symbol}', 'value': str(symbol)})
            for item in items:
                img = draw_text(item['value'], font, size)
                img_arr = np.array(img)
                while len(img_arr) > 0 and np.max(img_arr[0]) == np.min(img_arr[0]):
                    img_arr = np.delete(img_arr, 0, 0)
                if len(img_arr) == 0:
                    print(f'{font} generating empty {symbol}')
                    continue
                rotate_angle = 10
                x_offset = 2
                y_offset = 1
                img_arr = np.insert(img_arr, 0, 0, axis=0)
                img_arr = np.insert(img_arr, img_arr.shape[0], 0, axis=0)
                img = Image.fromarray(cv2.resize(img_arr, size, interpolation=cv2.INTER_AREA))
                img.save(f'{DATASET_FOLDER}/{symbol}/{font}.{item["name"]}.png')
                img.rotate(rotate_angle, fillcolor='black').save(f'{DATASET_FOLDER}/{symbol}/{font}.{item["name"]}.rotate+.png')
                img.rotate(-rotate_angle, fillcolor='black').save(f'{DATASET_FOLDER}/{symbol}/{font}.{item["name"]}.rotate-.png')
                ImageChops.offset(img, x_offset, 0).save(f'{DATASET_FOLDER}/{symbol}/{font}.{item["name"]}.offset_x+.png')
                ImageChops.offset(img, -x_offset, 0).save(f'{DATASET_FOLDER}/{symbol}/{font}.{item["name"]}.offset_x-.png')
                ImageChops.offset(img, 0, y_offset).save(f'{DATASET_FOLDER}/{symbol}/{font}.{item["name"]}.offset_y+.png')
                ImageChops.offset(img, 0, -y_offset).save(f'{DATASET_FOLDER}/{symbol}/{font}.{item["name"]}.offset_y-.{font}.png')


if __name__ == '__main__':
    create_path(DATASET_FOLDER)
    create_images()
