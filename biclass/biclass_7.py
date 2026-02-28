import numpy as np


def func(x):
    return 0.1 * x**2 - np.sin(x) + 0.1 * np.cos(x * 5) + 1.


# здесь объявляйте дополнительные функции (если необходимо)


coord_x = np.arange(-5.0, 5.0, 0.1) # значения отсчетов по оси абсцисс
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x) # общее число отсчетов

# здесь продолжайте программу
w = np.array([1.11, -0.26, 0.061, 0.0226, 0.00178])
y_pred = w[0] + w[1]*coord_x + w[2]*coord_x**2 + w[3]*coord_x**3 + w[4]*coord_x**4
losses = (y_pred - coord_y)**2
Q = np.sum(losses) / sz