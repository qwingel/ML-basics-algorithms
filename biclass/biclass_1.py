import numpy as np
x_test = [(5, -3), (-3, 8), (3, 6), (0, 0), (5, 3), (-3, -1), (-3, 3)]

sign_x = lambda x: -1 if x < 0 else 1

w = np.array([-33/13, 9/13, 1])

predict = [sign_x(np.array([1, i[0], i[1]]) @ w.T) for i in x_test]
print(predict)
