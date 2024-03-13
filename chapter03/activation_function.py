import numpy as np


def step_function(x):
    return np.array(x > 0, dtype=np.int32)


x1 = np.array([-1.0, 1.0, 2.0])
print("x1: " + x1.__str__())
print("step_function(x1): " + step_function(x1).__str__())
