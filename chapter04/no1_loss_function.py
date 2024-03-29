import numpy as np


def sum_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def run():
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    print("sum_squared_error: " + str(sum_squared_error(np.array(y), np.array(t))))
    print("cross_entropy_error: " +
          str(cross_entropy_error(np.array(y), np.array(t))))

    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    print("sum_squared_error: " + str(sum_squared_error(np.array(y), np.array(t))))
    print("cross_entropy_error: " +
          str(cross_entropy_error(np.array(y), np.array(t))))


run()
