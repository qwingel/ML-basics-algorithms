import numpy as np

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.02 * np.exp(-x) - 0.2 * np.sin(3 * x) + 0.5 * np.cos(2 * x) - 7


# здесь объявляйте необходимые функции
def df(w, x, y):
    return 2 * (w @ x - y) * x

def loss(w, x, y):
    return (w @ x - y) ** 2

coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.01, 1e-3, 1e-4, 1e-5, 1e-6]) # шаг обучения для каждого параметра w0, w1, w2, w3, w4
w = np.array([0., 0., 0., 0., 0.]) # начальные значения параметров модели
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего

Qe = np.mean([loss(w, np.array([1, x, x ** 2, x ** 3, x ** 4]), y) for x, y in zip(coord_x, coord_y)]) # начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел

# здесь продолжайте программу
for i in range(N):
    k = np.random.randint(0, sz - 1)  # sz - размер выборки (массива coord_x)
    xk = np.array([1, coord_x[k], coord_x[k] ** 2, coord_x[k] ** 3, coord_x[k] ** 4])
    Qk = loss(w, xk, coord_y[k])
    Qe = lm * Qk + (1 - lm) * Qe
    w = w - eta * df(w, xk, coord_y[k])

Q = np.mean([loss(w, np.array([1, x, x ** 2, x ** 3, x ** 4]), y) for x, y in zip(coord_x, coord_y)])
print(w, Qe, Q)