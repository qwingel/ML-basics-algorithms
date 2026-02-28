import numpy as np

w = np.array([20, -1, 3, 5, -0.15])
model = lambda a, x: a.T @ x


print(1 / (1 + np.exp(-1 * (w.T @ [0, 0, 0, 0, 0]))))
