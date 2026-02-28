import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
mean1 = np.array([1, -2, 0])
mean2 = np.array([1, 3, 1])
r = 0.7
D = 2.0
V = [[D, D * r, D*r*r], [D*r, D, D*r], [D*r*r, D*r, D]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T

x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.zeros(N), np.ones(N)])

# здесь вычисляйте векторы математических ожиданий и ковариационную матрицу по выборке x1, x2
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

VV = (np.cov(x1) + np.cov(x2))/2
# параметры для линейного дискриминанта Фишера
Py1, L1 = 0.5, 1  # вероятности появления классов
Py2, L2 = 1 - Py1, 1  # и величины штрафов неверной классификации

# здесь продолжайте программу
a = lambda v, m: np.linalg.inv(v) @ m.T
b = lambda l, py, m, v: np.log(l * py) - 0.5 * m.T @ np.linalg.inv(v) @ m

alpha1, alpha2 = a(VV, mm1), a(VV, mm2)
beta1, beta2 = b(L1, Py1, mm1, VV), b(L2, Py2, mm2, VV)