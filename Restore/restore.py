from PIL import Image
import numpy as np
from sympy import symbols, Eq, solve
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt

data = np.array(Image.open("image.jpg").convert('L'))
#first we should make the problem
for i in range(1, data.shape[0]):
    for j in range(1, data.shape[1], 10):
        data[i - 1, j] = data[i, j]
result = Image.fromarray(data)
result.save("new-image-1.jpg")
# to make and solve the equations, we should choode some points
# (1,1)->(0,1)
# (1,11)->(0,11)
# (3,1)->(2,1)
# (3,11)->(2,11)
# (2,1)->(?,?)
#then we use these points to make the equations
a1, a2, a3, a4, a5, a6, a7, a8 = symbols('a1, a2, a3, a4, a5, a6, a7,a8')
eq1 = Eq((a1 + a2 + a3 + a4), 0)
eq2 = Eq((a5 + a6 + a7 + a8), 1)
eq3 = Eq((a1 + 11 * a2 + 11 * 1 * a3 + a4), 0)
eq4 = Eq((a5 + 11 * a6 + 11 * 1 * a7 + a8), 11)
eq5 = Eq((3 * a1 + a2 + 3 * 1 * a3 + a4), 2)
eq6 = Eq((3 * a5 + a6 + 3 * 1 * a7 + a8), 1)
eq7 = Eq((3 * a1 + 11 * a2 + 3 * 11 * a3 + a4), 2)
eq8 = Eq((3 * a5 + 11 * a6 + 3 * 11 * a7 + a8), 11)
#then we use solve to solve the equation
x = solve((eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8), (a1, a2, a3, a4, a5, a6, a7, a8))
#now equation is solved and we can see the result
print(x)
#finally we use the results to solve the problem 
for i in range(1, data.shape[0]):
    for j in range(1, data.shape[1], 10):
        data[i, j] = data[i - 1, j]
result = Image.fromarray(data)
result.save("new-image-2.jpg")
