import numpy as np


# исходная функция
def func(x):
    return -0.7 * x - 0.2 * x ** 2 + 0.05 * x ** 3 - 0.2 * np.cos(3 * x) + 2


def Q(w, X, y):
    return np.mean(np.square(X @ w - y))


def dQdw(w, X, y):
    return X.T @ (X @ w - y) * (2 / X.shape[0])


coord_x = np.arange(-4.0, 6.0, 0.1)  # значения по оси абсцисс [-4; 6] с шагом 0.1
coord_y = func(coord_x)  # значения функции по оси ординат

sz = len(coord_x)  # количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001])  # шаг обучения для каждого параметра w0, w1, w2, w3
w = np.array([0., 0., 0., 0.])  # начальные значения параметров модели
N = 500  # число итераций алгоритма SGD
lm = 0.02  # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 20  # размер мини-батча (величина K = 20)
gamma = 0.8  # коэффициент гамма для вычисления импульсов Нестерова
v = np.zeros(len(w))  # начальное значение [0, 0, 0, 0]

# создание матрицы X и вектора y
X = np.array([[1, x, x ** 2, x ** 3] for x in coord_x])
y = np.array(coord_y)

# начальное значение Qe
Qe = Q(w, X, y)
np.random.seed(0)  # фиксация случайного генератора

for _ in range(N):
    k = np.random.randint(0, sz - batch_size - 1)
    batch_interval = np.arange(k, k + batch_size)
    Xk = X[batch_interval]
    yk = y[batch_interval]

    # пересчет Qe
    Qe = lm * Q(w, Xk, yk) + (1 - lm) * Qe

    # пересчет импульса и весов
    v = gamma * v + (1 - gamma) * eta * dQdw(w - gamma * v, Xk, yk)
    w -= v

# финальный расчет Q для всей выборки
Q = Q(w, X, y)