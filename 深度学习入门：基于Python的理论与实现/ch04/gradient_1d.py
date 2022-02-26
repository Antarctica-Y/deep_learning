# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


'''导数函数'''
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

'''函数'''
def function_1(x):
    return 0.01*x**2 + 0.1*x 

'''切线'''
def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y
     
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf_1 = tangent_line(function_1, 5)
y1 = tf_1(x)
tf_2 = tangent_line(function_1, 10)
y2 = tf_2(x)

plt.plot(x, y)
plt.plot(x, y1)
plt.plot(x, y2)

plt.show()
