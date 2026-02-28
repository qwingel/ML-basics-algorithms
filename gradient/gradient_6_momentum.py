import numpy as np


def func(x):
    return -0.5 * x + 0.2 * x ** 2 - 0.01 * x ** 3 - 0.3 * np.sin(4*x)


# здесь объявляйте функцию df (производную) и продолжайте программу
def df(x):
    return -0.5 + 0.4 * x - 0.03 * x ** 2 - 1.2 * np.cos(4*x)


eta = 0.1
x = -3.5
N = 200
r = 0.8
u = 0

for i in range(N):
    u = r * u + (1 - r) * eta * df(x)
    x = x - u

print(x, u)
