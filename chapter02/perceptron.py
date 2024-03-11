from typing import Literal
import numpy as np

x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7
print(w*x)
print(np.sum(w*x))
print(np.sum(w*x)+b)


def logic_circuit(type: Literal["AND", "NAND", "OR"], x1: float, x2: float) -> float:
    b: float
    w: np.ndarray[float]
    x = np.array([x1, x2])
    if type == "AND":
        w = np.array([0.5, 0.5])
        b = -0.7
    elif type == "NAND":
        w = np.array([-0.5, -0.5])
        b = 0.7
    elif type == "OR":
        w = np.array([0.5, 0.5])
        b = -0.2
    else:
        raise ValueError("type must be one of 'AND', 'NAND', 'OR'")
    if np.sum(w * x) + b > 0:
        return 1
    else:
        return 0


print("AND[0, 0]: "+str(logic_circuit("AND", 0, 0)))
print("AND[0, 1]: "+str(logic_circuit("AND", 0, 1)))
print("AND[1, 0]: "+str(logic_circuit("AND", 1, 0)))
print("AND[1, 1]: "+str(logic_circuit("AND", 1, 1)))
print("NAND[0, 0]: "+str(logic_circuit("NAND", 0, 0)))
print("NAND[0, 1]: "+str(logic_circuit("NAND", 0, 1)))
print("NAND[1, 0]: "+str(logic_circuit("NAND", 1, 0)))
print("NAND[1, 1]: "+str(logic_circuit("NAND", 1, 1)))
print("OR[0, 0]: "+str(logic_circuit("OR", 0, 0)))
print("OR[0, 1]: "+str(logic_circuit("OR", 0, 1)))
print("OR[1, 0]: "+str(logic_circuit("OR", 1, 0)))
print("OR[1, 1]: "+str(logic_circuit("OR", 1, 1)))
