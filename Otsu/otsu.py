from PIL import Image
import numpy as np
from sympy import symbols, solve
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt


def givteta(t):
    px = 0
    for x in range(t):
        px += (array[x] / size)
    return px


def givemean(t):
    mean = 0
    for x in range(t):
        mean += (x * (array[x] / size))
    return mean


def givemax(t):
    meant = givemean(t)
    tetat = givteta(t)
    if (tetat != 0) and ((1 - tetat) != 0):
        return (np.power((meant - (tetat * mean_255)), 2) / (tetat * (1 - tetat)))
    else:
        return 0


data = np.array(Image.open("image.jpg").convert('L'))
array = np.zeros(256)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        array[data[i, j]] += 1

plt.plot(array, color='blue')
plt.title("histogram")
plt.show()
size = data.shape[0] * data.shape[1]
maxes = 0
maxes_value = 0
mean_255 = givemean(255)
for i in range(0, 255):
    newmaxes = givemax(i)
    if (newmaxes > maxes_value):
        maxes_value = newmaxes
        maxes = i
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if data[i, j] > maxes:
            data[i, j] = 255
        else:
            data[i, j] = 0
print(maxes)
result = Image.fromarray(data)
result.save("new-image.jpg")
