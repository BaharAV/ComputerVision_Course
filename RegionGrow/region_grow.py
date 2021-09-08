import numpy as np
import cv2 as cv
import sys

sys.setrecursionlimit(10 ** 9)
from matplotlib import pyplot as plt


def regiongrow(image, tr):
    height, weight = image.shape
    imagemap = np.zeros(image.shape)
    seedlist = []
    seedlist.append((0, 0))
    label = 1
    connects = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    while (len(seedlist) > 0):
        currentPoint = seedlist.pop(0)
        imagemap[currentPoint[0], currentPoint[1]] = label
        for i in range(8):
            tempx = currentPoint[0] + connects[i][0]
            tempy = currentPoint[1] + connects[i][1]
            if tempx < 0 or tempy < 0 or tempx >= height or tempy >= weight:
                continue
            grayDiff = abs(int(image[currentPoint[0], currentPoint[1]]) - int(image[tempx, tempy]))
            if grayDiff < tr and imagemap[tempx, tempy] == 0:
                imagemap[tempx, tempy] = label
                seedlist.append((tempx, tempy))
    return imagemap

img = cv.imread('skincell.jfif', 0)
newImage = regiongrow(img, 10)
cv.imshow('new-image', newImage)
cv.waitKey(0)
img = cv.imread('coin.png', 0)
newImage = regiongrow(img, 10)
cv.imshow('new-image', newImage)
cv.waitKey(0)
