from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib as mpl
from random import randint
import math

mpl.use('TkAgg')
import matplotlib.pyplot as plt


def mean_(matrix):
    mean = 0
    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            mean = mean + matrix[i][j]
    mean = float((mean) / (matrix.shape[0] * matrix.shape[1]))
    return mean


def correlation(x, y):
    result = 0
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            result = result + (x[i][j] * y[i][j])
    result = (result * 1.0) / (x.shape[0] * y.shape[1])
    return result


data = np.array(Image.open("image.jpg").convert('RGB'))
# print(data.shape[0])
# print(data.shape[1])
# print(data.shape[2])
Rs = data[:, :, 0]
Rsmean = mean_(Rs)
Rsnew = Rs - Rsmean

Gs = data[:, :, 1]
Gsmean = mean_(Gs)
Gsnew = Gs - Gsmean

Bs = data[:, :, 2]
Bsmean = mean_(Bs)
Bsnew = Bs - Bsmean

shape = (3, 3)
CO = np.zeros(shape)
CO[0][0] = correlation(Rsnew, Rsnew)
CO[0][1] = correlation(Rsnew, Gsnew)
CO[0][2] = correlation(Rsnew, Bsnew)

CO[1][0] = correlation(Gsnew, Rsnew)
CO[1][1] = correlation(Gsnew, Gsnew)
CO[1][2] = correlation(Gsnew, Bsnew)

CO[2][0] = correlation(Bsnew, Rsnew)
CO[2][1] = correlation(Bsnew, Gsnew)
CO[2][2] = correlation(Bsnew, Bsnew)

print("correlation matrix")
print(CO)
w, v = np.linalg.eig(CO)

A = np.zeros((3, 3), np.float64)
for i in range(0, 3):
    for j in range(0, 3):
        A[i][j] = v[j][i]

print("A")
print(A)

P1 = data[0:, 0:, 0]
P2 = data[0:, 0:, 1]
P3 = data[0:, 0:, 2]
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        data[i, j] = np.dot(A, data[i, j])
        P1[i, j] = data[i, j, 0]
        P2[i, j] = data[i, j, 1]
        P3[i, j] = data[i, j, 2]

img = Image.fromarray(P1, 'L')
img.save('R.png')
# img.show()
img = Image.fromarray(P2, 'L')
img.save('G.png')
# img.show()
img = Image.fromarray(P3, 'L')
img.save('B.png')
# img.show()
img = Image.fromarray(data, 'RGB')
img.save('new_image.png')
img.show()
