import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
mean1 = np.array([1, -2])
mean2 = np.array([-3, -1])
mean3 = np.array([1, 2])

r = 0.5
D = 1.0
V = [[D, D * r], [D*r, D]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T
x3 = np.random.multivariate_normal(mean3, V, N).T

x_train = np.hstack([x1, x2, x3]).T
y_train = np.hstack([np.zeros(N), np.ones(N), np.ones(N) * 2])

# здесь вычисляйте векторы математических ожиданий и ковариационную матрицу по выборке x1, x2, x3
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)
mm3 = np.mean(x3.T, axis=0)

VV1 = np.cov(x1)
VV2 = np.cov(x2)
VV3 = np.cov(x3)

VV = (np.cov(x1) + np.cov(x2) + np.cov(x3)) / 3
# параметры для линейного дискриминанта Фишера
Py1, Py2, Py3 = 0.2, 0.4, 0.4
L1, L2, L3 = 1, 1, 1

# функции для LDA
a1 = np.linalg.inv(VV) @ mm1
a2 = np.linalg.inv(VV) @ mm2
a3 = np.linalg.inv(VV) @ mm3

b1 = np.log(L1 * Py1) - 0.5 * mm1.T @ np.linalg.inv(VV) @ mm1
b2 = np.log(L2 * Py2) - 0.5 * mm2.T @ np.linalg.inv(VV) @ mm2
b3 = np.log(L3 * Py3) - 0.5 * mm3.T @ np.linalg.inv(VV) @ mm3

# Правильное предсказание для ВСЕХ точек x_train
scores1 = x_train @ a1 + b1
scores2 = x_train @ a2 + b2
scores3 = x_train @ a3 + b3

predict = np.argmax([scores1, scores2, scores3], axis=0)
Q = np.sum(predict != y_train)
print(Q, predict)