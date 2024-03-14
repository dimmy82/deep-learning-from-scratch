import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
# 多次元配列の積
print("A * B:")
print(np.dot(A, B))
print("B * A:")
print(np.dot(B, A))

C = np.array([[1, 2, 3], [4, 5, 6]])
D = np.array([[1, 2], [3, 4], [5, 6]])
print("C.sharp: " + str(C.shape))
print("D.sharp: " + str(D.shape))
print("C * D:")
print(np.dot(C, D))
print("D * C:")
print(np.dot(D, C))

E = np.array([[1, 2, 3], [4, 5, 6]])
F = np.array([1, 2])
print("E.sharp: " + str(E.shape))
print("F.sharp: " + str(F.shape))
print("E * F: エラーになる")
# print(np.dot(E, F))
print("F * E:")
print(np.dot(F, E))

G = np.array([[1, 2], [3, 4], [5, 6]])
print("G * F: エラーにはならない！")
print(np.dot(G, F))
