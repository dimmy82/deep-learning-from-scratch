import numpy as np
from activation_function import identity_function, sigmoid_function


def neural_network(x1: np.float32, x2: np.float32) -> (np.float32, np.float32):
    x = np.array([x1, x2])
    w1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    b1 = np.array([0.1, 0.2, 0.3])
    layer1 = sigmoid_function(np.dot(x, w1) + b1)  # layer1.shape = (3,)
    w2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    b2 = np.array([0.1, 0.2])
    layer2 = sigmoid_function(np.dot(layer1, w2) + b2)  # layer2.shape = (2,)
    w3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    b3 = np.array([0.1, 0.2])
    layer3 = identity_function(np.dot(layer2, w3) + b3)  # layer3.shape = (2,)
    return (layer3[0], layer3[1])


def run():
    print("neural_network(1.0, 0.5) => " + str(neural_network(1.0, 0.5)))


run()
