import numpy as np


def func(x):
    return 2 * x + 0.1 * x ** 3 + 2 * np.cos(3*x)

# здесь объявляйте функцию df (производную) и продолжайте программу
def df(x):
    return 2 + 0.3 * x ** 2 - 6 * np.sin(3 * x)

eta = 0.5
x = 4.0
N = 200
a = 0.8
G = 0
e = 0.01

for i in range(N):
    G = a * G + (1 - a) * df(x) * df(x)
    x -= x - eta * (df(x) / (G ** 0.5 + e))

print(x, G)