import numpy as np

w = np.array([-330, 1, 3, 2, -0.15])
x = np.array([1,240,80,1,1000])

print(1 / (1 + np.exp(-1 * (w.T @ x))))