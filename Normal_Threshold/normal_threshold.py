from PIL import Image
import numpy as np
from sympy import symbols, solve
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt


def findx(image):
    data = np.array(Image.open(image).convert('L'))
    array = np.zeros(256)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            array[data[i, j]] += 1
    # plt.plot(array, color='blue')
    # plt.title("histogram")
    # plt.show()
    ao = np.max(array)
    muo = np.where(array == ao)[0][0]
    ao = ao / (data.shape[0] * data.shape[1])
    sigmao = 1 / (ao * np.sqrt(2 * np.pi))

    zeroo = muo + 3 * sigmao
    zeroo = int(zeroo)
    if (zeroo > 255):
        ab = np.max(array[0:muo - 1])
        mub = np.where(array == ab)[0][0]
        ab = ab / (data.shape[0] * data.shape[1])
        sigmab = 1 / (ab * np.sqrt(2 * np.pi))
    else:
        zerob = np.argmin(array[zeroo:255])
        zerob = zeroo + zerob
        zerob = int(zerob)
        mub = np.argmax(array[zeroo:zerob])
        sigmab = (zerob - mub) / 3
        ab = np.max(array[zeroo:255])
        ab = ab / (data.shape[0] * data.shape[1])

    teta = (ao * sigmao) / ((ao * sigmao) + (ab * sigmab))

    x = symbols('x', real=True)
    part1 = (np.power((x - mub), 2) / 2 * np.power(sigmab, 2))
    part2 = (np.power((x - muo), 2) / 2 * np.power(sigmao, 2))
    part3 = np.log(((1 - teta) * sigmao) / ((teta) * sigmab))
    x = solve((part1 - part2 - part3).rewrite(np.piecewise))
    x = [i for i in x if i > 0] or None
    print(x)
    savedata = np.array(data, copy=True)
    for z in range(len(x)):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if (data[i, j] > x[z]):
                    data[i, j] = 255
                else:
                    data[i, j] = 0
        result = Image.fromarray(data)
        result.save("new-" + str(z + 1) + "-" + image)
        data = savedata


findx("image-1.jpg")
findx("image-2.jpg")
