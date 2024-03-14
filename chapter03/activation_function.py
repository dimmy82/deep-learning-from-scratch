import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int32)


x1 = np.array([-1.0, 1.0, 2.0])
print("x1: " + x1.__str__())
print("step_function(x1): " + step_function(x1).__str__())

x2 = np.arange(-5.0, 5.0, 0.1)
y = step_function(x2)
plt.plot(x2, y)
plt.ylim(-0.1, 1.1)
# plt.show()


def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


x11 = np.array([-1.0, 1.0, 2.0])
print("x11: " + x11.__str__())
print("sigmoid(x11): " + sigmoid_function(x11).__str__())

x12 = np.arange(-5.0, 5.0, 0.1)
y = sigmoid_function(x12)
plt.plot(x12, y)
plt.ylim(-0.1, 1.1)

plt.show()

