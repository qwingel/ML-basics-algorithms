import numpy as np

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.1 * x**2 - np.sin(x) + 5.

# здесь объявляйте необходимые функции
def df(x):
    return 0.2 * x - np.cos(x)

x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
y = func(x) # значения функции по оси ординат

sz = len(x)	# количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001]) # шаг обучения для каждого параметра w0, w1, w2, w3
w = np.array([0., 0., 0., 0.]) # начальные значения параметров модели
N = 200 # число итераций градиентного алгоритма

# здесь продолжайте программу

def ax(w, s):
    return w.T @ s

def Qf(w):
    return sum((ax(w, [1, i, i * i, i * i * i]) - func(i)) ** 2 for i in x) / sz

def Wf(w):
    grad = np.zeros_like(w, dtype=float)
    for i in x:
        s_i = np.array([1.0, i, i*i, i*i*i])
        grad += (w @ s_i - func(i)) * s_i
    return (2.0 / sz) * grad

for j in range(N):
    w = w - eta * Wf(w)
    Q = Qf(w)