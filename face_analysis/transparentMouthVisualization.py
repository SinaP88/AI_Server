import cv2
import numpy as np
import os
import logging
import tensorflow as tf
from PIL import Image
import io


def predict_mouth(model, input_img, x, y, w, h):
    print(f"w: {w}, h: {h}")
    input_s = cv2.resize(input_img[y:y + h, x:x + w], (512, 512))
    input_s = input_s / 255.0
    input_s = np.array(input_s, dtype=float)
    input_s = np.expand_dims(input_s, axis=0)
    result = model.predict(input_s)
    msk = np.zeros([input_img.shape[0], input_img.shape[1]])
    # for face in face from other module
    p_msk = np.reshape(result * 255, (512, 512))
    msk[y:y + h, x:x + w] = cv2.resize(p_msk, (w, h))
    inverse_msk = 255 - msk
    return inverse_msk

# def prepareInput(image_path, image_channel=3, save_path=None):
#     if image_channel !=3:
#         image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 0)
#     else:
#         image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
#     if image is None:
#         raise FileNotFoundError
#     # print('Image original shape:', image.shape)
#     image = cv2.resize(image, (512, 512))
#     if save_path:
#         cv2.imwrite(save_path, image)
#     # image = np.expand_dims(image, axis=-1)
#     image = image / 255.0
#     # print('Resized image shape:', image.shape)
#     image = np.array(image, dtype=float)
#     image = np.expand_dims(image, axis=0)
#     print('Input tensor shape:', image.shape)
#     return image


def transparentImage(input_image, r_mask):
    transparent = Image.new('RGBA', (input_image.shape[1], input_image.shape[0]), (0, 0, 0, 0))
    im = Image.fromarray(input_image.astype('uint8'), 'RGB')
    # im.save("image.png", format="png")
    mask = Image.fromarray(np.uint8(r_mask), 'L')
    # mask.save("mask.png", format="png")
    # msk = Image.open(, 'r')
    # mask = Image.open(r_mask, 'r')
    transparent.paste(im, (0, 0), mask=mask)
    # transparent.save('transparent.png', format="png")
    return transparent


