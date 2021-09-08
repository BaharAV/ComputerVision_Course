from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt


########################################################################################################################
# drawing the histogram
def draw_histogram(image):
    data = np.array(Image.open(image).convert('L'))
    array = np.zeros(256)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            array[data[i, j]] += 1
    plt.plot(array, color='blue')
    plt.title("histogram")
    plt.show()


########################################################################################################################
# make histogram flat
def make_histogram_flat(image):
    data = np.array(Image.open(image).convert('L'))
    total = data.shape[0] * data.shape[1]
    print(total)
    eachofnumbers = int(total / 256)
    print(eachofnumbers)
    for i in range(255):  # 0 to 254 because the last one should be corrected by itself
        array = np.zeros(256)
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                array[data[x, y]] += 1
        if (array[i] > eachofnumbers):
            print("model 1")
            howmuch = int(array[i] - eachofnumbers)
            j = 0
            go = 1
            for x in range(data.shape[0]):
                if (go == 1):
                    for y in range(data.shape[1]):
                        if (go == 1):
                            if (data[x, y] == i):
                                data[x, y] = i + 1
                                j += 1
                                if (j >= howmuch):
                                    go = 0
                                    break
                        else:
                            break
                else:
                    break
        else:
            if (array[i] < eachofnumbers):
                print("model 2")
                howmuch = int(eachofnumbers - array[i])
                j = 0
                number = 1
                while (j < howmuch):
                    for x in range(data.shape[0]):
                        for y in range(data.shape[1]):
                            if (data[x, y] == (i + number)):
                                data[x, y] = i
                                j += 1
                                if (j >= howmuch):
                                    break
                        if (j >= howmuch):
                            break
                    if (j >= howmuch):
                        break
                    else:
                        number += 1
    #first method
    result = Image.fromarray(data)
    result.save('newpicture-1.jpg')  # this left the extra part in pixel 255
    result.show()
    array = np.zeros(256)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            array[data[i, j]] += 1
    print(array)
    plt.plot(array, color='blue')
    plt.title("histogram")
    plt.show()
    #second method
    extra = int(array[255] - eachofnumbers)
    print(extra)
    for x in range(extra):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if (data[i, j] == 255):
                    data[i, j] = x
                    x = x + 1
                    if (x >= extra):
                        break
            if (x >= extra):
                break
        if (x >= extra):
            break

    result = Image.fromarray(data)
    result.save('newpicture-2.jpg')  # this left the extra part in pixel 0 to pixel n instead of pixel 255
    result.show()
    array = np.zeros(256)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            array[data[i, j]] += 1
    print(array)
    plt.plot(array, color='blue')
    plt.title("histogram")
    plt.show()


########################################################################################################################
# main
make_histogram_flat("picture.jpg")
