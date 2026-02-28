import numpy as np

x_test = [(9, 6), (2, 4), (-3, -1), (3, -2), (-3, 6), (7, -3), (6, 2)]
sign_x = lambda x: -1 if x < 0 else 1
w = np.array([14/5, -7/5, 1])
predict = [sign_x(np.array([1, i[0], i[1]]) @ w.T) for i in x_test]