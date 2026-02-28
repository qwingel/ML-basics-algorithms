import numpy as np

w = np.array([15/7, -9/7, -1])
x_test = np.array([(1, -8, -4), (1, -2, 2), (1, 4, 8), (1, 6, 3)]) # задайте самостоятельно (признаки образов: x0, x1, x2)
y_test = np.array([1, 1, -1, -1]) # задайте самостоятельно (метки класса)

margin = [([1, x_test[i][0], x_test[i][1]] @ w.T) * y_test[i] for i in range(len(x_test))]
print(margin)