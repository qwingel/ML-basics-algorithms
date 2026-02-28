import numpy as np


def func(x):
    return 0.4 * x + 0.1 * np.sin(2*x) + 0.2 * np.cos(3*x)

# здесь объявляйте функцию df (производную) и продолжайте программу
def df(x):
    return 0.4 + 0.2 * np.cos(2*x) - 0.6 * np.sin(3 * x)

eta = 1.0
x = 4.0
N = 500
r = 0.7
u = 0

for i in range(N):
    u = r * u + (1 - r) * eta * df(x - r * u)
    x -= u
