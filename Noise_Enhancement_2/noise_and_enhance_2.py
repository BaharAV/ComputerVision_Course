from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib as mpl
from random import randint
import math

mpl.use('TkAgg')
import matplotlib.pyplot as plt


# def variance_(data):
#     mean = mean_(data)
#     var = 0
#     for i in range(0, data.shape[0]):
#         for j in range(0, data.shape[1]):
#             var += ((data[i][j] - mean) ** 2)
#     return var / (data.shape[0] * data.shape[1])


def mean_(data):
    mean = 0
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            mean += data[i][j]
    return mean / (data.shape[0] * data.shape[1])


def median_(data):
    array = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            array.append(data[i, j])
    array.sort()
    # print(array)
    med = array[(int)(len(array) / 2)]
    return med

def make_gaussian(image):
    img = cv.imread(image, 0)
    noise_sigma = 35
    temp = np.float64(np.copy(img))
    noise = np.random.randn(temp.shape[0], temp.shape[1]) * noise_sigma
    noisy_image = temp + noise
    cv.imwrite("gaussian.jpg", noisy_image)


def mean(image, which):
    data = np.array(Image.open(image).convert('L'))
    w = 2
    m = data.shape[0]
    n = data.shape[1]
    img_new = np.zeros([m, n])
    for i in range(2, m):
        for j in range(2, n):
            img_new[i, j] = mean_(data[i - w:i + w, j - w:j + w])
    img_new = img_new.astype(np.uint8)
    if (which == 0):
        cv.imwrite('saltandpepper_mean.png', img_new)
    else:
        cv.imwrite('gaussian_mean.png', img_new)


def median(image, which):
    data = np.array(Image.open(image).convert('L'))
    w = 2
    m = data.shape[0]
    n = data.shape[1]
    img_new = np.zeros([m, n])
    for i in range(2, m):
        for j in range(2, n):
            img_new[i, j] = median_(data[i - w:i + w, j - w:j + w])
    if (which == 0):
        cv.imwrite('saltandpepper_median.png', img_new)
    else:
        cv.imwrite('gaussian_median.png', img_new)


########################################################################################################################
make_gaussian("picture_noise.jpg")
median("gaussian.jpg", 1)
mean("gaussian.jpg", 1)
