import numpy as np
import matplotlib.pylab as plt


def numerical_diff(func, x):
    h = 1e-4
    return (func(x + h) - func(x - h)) / (2 * h)


def function_1(x):
    return 0.01 * x**2 + 0.1 * x


def function_2(x):
    return x[0]**2 + x[1]**2


def function_2_x0(x):
    return function_2([x, 4.0])


def function_2_x1(x):
    return function_2([3.0, x])


def run():
    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)
    plt.xlabel("x")
    plt.ylabel("function_1(x)")
    plt.plot(x, y)
    # plt.show()
    print("function_1(5) の微分: " + str(numerical_diff(function_1, 5)))
    print("function_1(10) の微分: " + str(numerical_diff(function_1, 10)))
    print("function_2(x0 = 3.0) の偏微分: " +
          str(numerical_diff(function_2_x0, 3.0)))
    print("function_2(x1 = 4.0) の偏微分: " +
          str(numerical_diff(function_2_x1, 4.0)))


run()
