import numpy as np


def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5


def model(x, w):
    """Полином 3-й степени"""
    return w[0] + w[1] * x + w[2] * x ** 2 + w[3] * x ** 3


def loss(y_true, y_pred):
    return (y_true - y_pred) ** 2


def grad_loss(x_batch, y_batch, w):
    """Градиент для мини-батча"""
    batch_size = len(x_batch)
    grad = np.zeros(4)
    for i in range(batch_size):
        y_pred = model(x_batch[i], w)
        error = y_pred - y_batch[i]
        grad += error * np.array([1, x_batch[i], x_batch[i] ** 2, x_batch[i] ** 3])
    return 2 * grad / batch_size


# Инициализация
coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)
sz = len(coord_x)
eta = np.array([0.1, 0.01, 0.001, 0.0001])
w = np.array([0., 0., 0., 0.])
N = 500
lm = 0.02
batch_size = 50

Qe = np.mean([loss(coord_y[i], model(coord_x[i], w)) for i in range(sz)])
np.random.seed(0)

for _ in range(N):
    k = np.random.randint(0, sz - batch_size)
    x_k = coord_x[k:k + batch_size]
    y_k = coord_y[k:k + batch_size]

    grad_Qk = grad_loss(x_k, y_k, w)  # Градиент батча
    w -= eta * grad_Qk  # Обновление весов

    Qk = np.mean([loss(y_k[i], model(x_k[i], w)) for i in range(batch_size)])
    Qe = lm * Qk + (1 - lm) * Qe

Q = np.mean([loss(coord_y[i], model(coord_x[i], w)) for i in range(sz)])
print(w, Qe, Q)