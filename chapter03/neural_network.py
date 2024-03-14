import numpy as np


def neural_network(x1: np.int32, x2: np.int32) -> (np.int32, np.int32):
    x = np.array([x1, x2])
    w1 = np.array([[1, 2, 3], [4, 5, 6]])
    layer1 = np.dot(x, w1)  # layer1.shape = (3,)
    w2 = np.array([[7, 8], [9, 10], [11, 12]])
    layer2 = np.dot(layer1, w2)  # layer2.shape = (2,)
    w3 = np.array([[13, 14], [15, 16]])
    layer3 = np.dot(layer2, w3)  # layer3.shape = (2,)
    return (layer3[0], layer3[1])


print("neural_network(1, 2) => " + str(neural_network(1, 2)))
