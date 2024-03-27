import numpy as np

from differentiation import function_2


def numberical_gradient(func, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for index in range(x.size):
        org_val = x[index]

        x[index] = org_val + h
        f_x_plus_h = func(x)

        x[index] = org_val - h
        f_x_minus_h = func(x)

        grad[index] = (f_x_plus_h - f_x_minus_h) / (2 * h)
        x[index] = org_val

    return grad


def gradient_descent(func, init_x, lr=0.01, step_num=100):
    x = init_x

    for index in range(step_num):
        grad = numberical_gradient(func, x)
        x -= lr * grad

    return x


def run():
    print("function_2([3.0, 4.0]) の勾配: " +
          str(numberical_gradient(function_2, np.array([3.0, 4.0]))))
    print("function_2([0.0, 2.0]) の勾配: " +
          str(numberical_gradient(function_2, np.array([0.0, 2.0]))))
    print("function_2([3.0, 0.0]) の勾配: " +
          str(numberical_gradient(function_2, np.array([3.0, 0.0]))))
    print("function_2([-3.0, 4.0]) の勾配降下の結果: " + str(gradient_descent(function_2,
          np.array([-3.0, 4.0]), 0.1, 100)))


run()
