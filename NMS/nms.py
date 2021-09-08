from PIL import Image
import numpy as np
from sympy import symbols, solve
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt


def do_filter(img):
    horizontal = np.array([(-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)])
    horizontalmade = np.zeros(shape=(img.shape[0], img.shape[1]))
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            horizontalmade[i, j] = np.sum(np.multiply(img[i - 1: i + 2, j - 1: j + 2], horizontal))
    result = Image.fromarray(horizontalmade)
    if result.mode != 'L':
        result = result.convert('L')
    result.save("horizontal.jpg")
    vertical = np.array([(-1.0, -2.0, -1.0), (0.0, 0.0, 0.0), (1.0, 2.0, 1.0)])
    verticalmade = np.zeros(shape=(img.shape[0], img.shape[1]))
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            verticalmade[i, j] = np.sum(np.multiply(img[i - 1: i + 2, j - 1: j + 2], vertical))
    result = Image.fromarray(verticalmade)
    if result.mode != 'L':
        result = result.convert('L')
    result.save("vertical.jpg")
    totalmade = np.sqrt(np.square(horizontalmade) + np.square(verticalmade))
    result = Image.fromarray(totalmade)
    if result.mode != 'L':
        result = result.convert('L')
    result.save("total.jpg")
    img = totalmade
    angles = np.rad2deg(np.arctan2(verticalmade, horizontalmade))
    new_img = np.zeros(shape=(img.shape[0], img.shape[1]))
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if (angles[i, j] < 0):
                angles[i, j] += 180
            if (22.5 <= angles[i, j] < 67.5):
                val = max(img[i - 1, j - 1], img[i + 1, j + 1])
            elif (67.5 <= angles[i, j] < 112.5):
                val = max(img[i, j - 1], img[i, j + 1])
            elif (112.5 <= angles[i, j] < 157.5):
                val = max(img[i + 1, j - 1], img[i - 1, j + 1])
            else:
                val = max(img[i - 1, j], img[i + 1, j])
            if img[i, j] >= val:
                new_img[i, j] = img[i, j]
    result = Image.fromarray(new_img)
    if result.mode != 'L':
        result = result.convert('L')
    result.save("nms.jpg")


data = np.array(Image.open("image.jpg").convert('L'))
do_filter(data)
