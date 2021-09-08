from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib as mpl
from random import randint
import math

mpl.use('TkAgg')
import matplotlib.pyplot as plt

# calculating variance for a windoe #####################################################################################################
def variance_(data):
    mean = mean_(data)
    var = 0
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            var += ((data[i][j] - mean) ** 2)
    return var / (data.shape[0] * data.shape[1])

# calculating mean for a windoe #####################################################################################################
def mean_(data):
    mean = 0
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            mean += data[i][j]
    return mean / (data.shape[0] * data.shape[1])

# calculating median for a windoe #####################################################################################################
def median_(data):
    array = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            array.append(data[i, j])
    array.sort()
    # print(array)
    med = array[(int)(len(array) / 2)]
    return med

# global equalize #####################################################################################################
def global_equalize_(image):
    data = np.array(Image.open(image).convert('L'))
    array = np.zeros(256)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            array[data[i, j]] += 1
    # print(array)
    plt.plot(array, color='blue')
    plt.title("picture_histogram")
    plt.show()
    min = 255
    max = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if (data[i, j] < min):
                min = data[i, j]
            if (data[i, j] > max):
                max = data[i, j]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j] -= min
    ranges = max - min
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j] = int((data[i, j] / ranges) * 255)
    result = Image.fromarray(data)
    result.save('globalequalize.jpg')
    array = np.zeros(256)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            array[data[i, j]] += 1
    # print(array)
    plt.plot(array, color='blue')
    plt.title("globalequalize_histogram")
    plt.show()
    return data;

# local equalize parts #####################################################################################################
# same as global equalize but without saving the picture
# used for each part of the picture in local equalize
def local_equalize_parts_(data):
    # data = np.array(Image.open(image).convert('L'))
    min = 255
    max = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if (data[i, j] < min):
                min = data[i, j]
            if (data[i, j] > max):
                max = data[i, j]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j] -= min
    ranges = max - min
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j] = int((data[i, j] / ranges) * 255)
    # result = Image.fromarray(data)
    # result.save('globalequalize.jpg')
    array = np.zeros(256)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            array[data[i, j]] += 1
    return data;

# local equalize #####################################################################################################
def local_equalize_(image, m, n):
    data = np.array(Image.open(image).convert('L'))
    x = int(data.shape[0] / m)
    y = int(data.shape[1] / n)
    for r in range(0, x * m, x):
        for c in range(0, y * n, y):
            # cv.imwrite(f"img{r}_{c}.png", data[r:r + x, c:c + y])
            # data[r:r + x, c:c + y] = local_equalize_parts_(f"img{r}_{c}.png")
            data[r:r + x, c:c + y] = local_equalize_parts_(data[r:r + x, c:c + y])
    result = Image.fromarray(data)
    result.save('localequalization.jpg')
    # result.show()
    newarray = np.zeros(256)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            newarray[data[i, j]] += 1
    # plt.plot(previousarray, color='green')
    plt.plot(newarray, color='blue')
    plt.title("localequalization_histogram")
    plt.show()

# local enhancement #####################################################################################################
def local_enhancement_(image):
    data = np.array(Image.open(image).convert('L'))
    mainmean = mean_(data)
    k = 0.5
    w = 2
    m=data.shape[0]
    n=data.shape[1]
    for r in range(w, m):
        for c in range(w, n):
            var = variance_((data[r - w:r + w, c - w:c + w]))
            mean = mean_(data[r - w:r + w, c - w:c + w])
            if (var == 0):
                continue;
            A = (k * mainmean) / var
            # data[r, c] = A * (data[r, c] - mean) + mean
            for i in range(0, data[r - w:r + w, c - w:c + w].shape[0]):
                for j in range(0, data[r - w:r + w, c - w:c + w].shape[1]):
                    data[r - w + i, c - w + j] = A * (data[r - w + i, c - w + j] - mean) + mean
    result = Image.fromarray(data)
    result.save('localenahncement.png')
    # result.show()
    array = np.zeros(256)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            array[data[i, j]] += 1
    plt.plot(array, color='blue')
    plt.title("localenahncement_histogram")
    plt.show()

# making salt and pepper noise #####################################################################################################
def make_saltandpepper(image):
    data = np.array(Image.open(image).convert('L'))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = randint(0, 50)
            # print(value)
            if (value > 0 and value <= 1):
                data[i, j] = 0
            elif (value > 1 and value <= 2):
                data[i, j] = 255
            else:
                data[i, j] = data[i, j]
    result = Image.fromarray(data)
    result.save('saltandpepper.jpg')
    # result.show()

# # making gaussian noise #####################################################################################################
# def make_gaussian(image):
#     img = cv.imread(image, 0)
#     noise_sigma = 35
#     temp = np.float64(np.copy(img))
#     noise = np.random.randn(temp.shape[0], temp.shape[1]) * noise_sigma
#     noisy_image = temp + noise
#     cv.imwrite("gaussian.jpg", noisy_image)

# smoothing filter #####################################################################################################
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

# median filter #####################################################################################################
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
global_equalize_("picture.jpg")
local_equalize_("picture.jpg", 3, 3)
local_enhancement_("picture.jpg")
#
make_saltandpepper("picture_noise.jpg")
median("saltandpepper.jpg", 0)
mean("saltandpepper.jpg", 0)
#
# make_gaussian("picture_noise.jpg")
# median("gaussian.jpg", 1)
# mean("gaussian.jpg", 1)
