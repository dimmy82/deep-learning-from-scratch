import numpy as np

a1 = np.array([1, 2, 3])
print(a1)
a2 = np.array([0.1, 0.2, 0.3])
print(a2)
print(a1 + a2)
print(a1 - a2)
print(a1 * a2)
print(a1 / a2)

a3 = np.array([[1, 5, 9], [4, 2, 8], [9, 1, 4]])
print(a3 > 7)
print(a3[a3 > 7])
