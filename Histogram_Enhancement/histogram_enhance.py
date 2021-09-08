from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt


########################################################################################################################
# part 1
# enhance the image without using a library
def enhance_without_library(image):
    data = np.array(Image.open(image).convert('L'))
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
    result.save('newpicture-withoutlibrary.jpg')
    # show the new picture
    result.show()
    # draw combined diagrams
    imageone = np.array(Image.open(image).convert('L'))
    previousarray = np.zeros(256)
    for i in range(imageone.shape[0]):
        for j in range(imageone.shape[1]):
            previousarray[imageone[i, j]] += 1
    imagetwo = data
    newarray = np.zeros(256)
    for i in range(imagetwo.shape[0]):
        for j in range(imagetwo.shape[1]):
            newarray[imagetwo[i, j]] += 1
    plt.plot(previousarray, color='green')
    plt.plot(newarray, color='blue')
    plt.title("comparsion between previous and new histogram without library")
    plt.show()


########################################################################################################################
# part 1-prime
# enhance the image using opencv library of python
def enhance_with_library(image):
    previousimage = cv.imread(image, 0)
    newimage = cv.equalizeHist(previousimage)
    result = np.hstack([newimage])
    # save the new picture
    cv.imwrite('newpicture-withlibrary.jpg', result)
    # show the new picture
    result = Image.open('newpicture-withlibrary.jpg').convert('L')
    result.show()
    # draw combined diagrams
    imageone = np.array(Image.open(image).convert('L'))
    previousarray = np.zeros(256)
    for i in range(imageone.shape[0]):
        for j in range(imageone.shape[1]):
            previousarray[imageone[i, j]] += 1
    imagetwo = newimage
    newarray = np.zeros(256)
    for i in range(imagetwo.shape[0]):
        for j in range(imagetwo.shape[1]):
            newarray[imagetwo[i, j]] += 1
    plt.plot(previousarray, color='green')
    plt.plot(newarray, color='blue')
    plt.title("comparsion between previous and new histogram with library")
    plt.show()


########################################################################################################################
# part 2
# drawing the histogram - to show seprately
def draw_histogram_firstmethod(image, title):
    data = np.array(Image.open(image).convert('L'))
    array = np.zeros(256)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            array[data[i, j]] += 1
    x_axis = range(0, 256)
    y_axis = array.data
    plt.title(title)
    plt.scatter(x_axis, y_axis, color='blue', marker='.', label="histogram")
    plt.xlabel("value")
    plt.ylabel("number")
    plt.grid(True)
    plt.legend()
    plt.show()


########################################################################################################################
# main
enhance_without_library("picture.jpg")  # question 1 - without the help of library
enhance_with_library("picture.jpg")  # question 1 - with the help of library
draw_histogram_firstmethod('picture.jpg', 'previous histogram')  # question 2
draw_histogram_firstmethod('newpicture-withlibrary.jpg', 'new histogram with library')  # question 2
draw_histogram_firstmethod('newpicture-withoutlibrary.jpg', 'new histogram without library')  # question 2
# question 2 is also solved with a combined model in the codes
# the two diagrams are a bit different before and after saving the picture but the main job is the same
