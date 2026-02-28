import numpy as np

def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5

coord_x = np.arange(-4.0, 6.0, 0.1)
x_train = np.array([[_x**i for i in range(4)] for _x in coord_x])
y_train = func(coord_x)

# Решение нормального уравнения
XTX = np.dot(x_train.T, x_train)
XTy = np.dot(x_train.T, y_train)
w = np.linalg.solve(XTX, XTy)

# Эмпирический риск
a_train = np.dot(x_train, w)
Q = np.mean((y_train - a_train)**2)
