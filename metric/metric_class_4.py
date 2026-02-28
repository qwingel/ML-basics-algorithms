import numpy as np

# координаты четырех точек
x = np.array([0, 1, 2, 3])
y = np.array([0.5, 0.8, 0.6, 0.2])

x_est = np.arange(0, 3.1, 0.1) # множество точек для промежуточного восстановления функции

# здесь продолжайте программу
h = 1.0

K = lambda r: np.abs(1 - r) * bool(r <= 1)
ro = lambda xx, xi: np.abs(xx - xi)     # метрика
w = lambda xx, xi: K(ro(xx, xi) / h)    # веса

y_est = []
for xx in x_est:
    ww = np.array([w(xx, xi) for xi in x])
    yy = np.dot(ww, y) / sum(ww)            # формула Надарая-Ватсона
    y_est.append(yy)
