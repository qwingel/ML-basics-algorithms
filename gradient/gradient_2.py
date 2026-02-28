import numpy as np

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.5 * x**2 - 0.1 * 1/np.exp(-x) + 0.5 * np.cos(2*x) - 2.


# здесь объявляйте необходимые функции

coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.01, 0.001, 0.0001, 0.01, 0.01]) # шаг обучения для каждого параметра w0, w1, w2, w3, w4
w = np.array([0., 0., 0., 0., 0.]) # начальные значения параметров модели
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего

Qe = sum((w.T @ np.array([1, coord_x[i], coord_x[i] * coord_x[i], np.cos(2 * coord_x[i]), np.sin(2 * coord_x[i])]) - coord_y[i]) ** 2 for i in range(sz)) / sz # начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел

for i in range(N):
    k = np.random.randint(0, sz - 1)
    xk = np.array([1, coord_x[k], coord_x[k] * coord_x[k], np.cos(2 * coord_x[k]), np.sin(2 * coord_x[k])])
    E = (w.T @ xk - coord_y[k])
    w = w - 2 * eta * E * xk
    Qe = lm * E ** 2 + (1 - lm) * Qe

Q = sum((w.T @ np.array([1, coord_x[i], coord_x[i] * coord_x[i], np.cos(2 * coord_x[i]), np.sin(2 * coord_x[i])]) - coord_y[i]) ** 2 for i in range(sz)) / sz