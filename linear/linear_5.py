import numpy as np

np.random.seed(0)
x = np.arange(-1.0, 1.0, 0.1)

model_a = lambda xx, ww: (ww[0] + ww[1] * xx)
Y = -5.2 + 0.7 * x + np.random.normal(0, 0.1, len(x))

ones = np.ones(len(x))
X = np.column_stack((ones, x))

XT = X.T
w = np.linalg.inv(XT @ X) @ XT @ Y
