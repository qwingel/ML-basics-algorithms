import numpy as np

x_test = np.array([(-5, 2), (-4, 6), (3, 2), (3, -3), (5, 5), (5, 2), (-1, 3)])
y_test = np.array([1, 1, 1, -1, -1, -1, -1])
w = np.array([-8/3, -2/3, 1])

Q = 0
for i in range(len(x_test)):
    a_x = [1, x_test[i][0], x_test[i][1]] @ w.T
    M = a_x * y_test[i]
    if M < 0:
        Q += 1

print(Q)