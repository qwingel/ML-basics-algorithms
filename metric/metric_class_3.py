import numpy as np
from sklearn.model_selection import train_test_split


np.random.seed(0)
n_feature = 2 # число признаков

# исходные параметры для формирования образов обучающей выборки
r1 = 0.7
D1 = 3.0
mean1 = [3, 3]
V1 = [[D1 * r1 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

r2 = 0.5
D2 = 2.0
mean2 = [1, 1] # + np.array(range(1, n_feature+1)) * 0.5
V2 = [[D2 * r2 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

r3 = -0.7
D3 = 1.0
mean3 = [-2, -2]  # + np.array(range(1, n_feature+1)) * -0.5
V3 = [[D3 * r3 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

# моделирование обучающей выборки
N1, N2, N3 = 200, 150, 190
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T
x3 = np.random.multivariate_normal(mean3, V3, N3).T

data_x = np.hstack([x1, x2, x3]).T
data_y = np.hstack([np.zeros(N1), np.ones(N2), np.ones(N3) * 2])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.5, shuffle=True)

# здесь продолжайте программу
h = 1.0

def K(r):
    # Гауссовское ядро из условия
    return (1.0 / (2.0 * np.pi)) * np.exp(-(r**2) / 2.0)

predict = []
y_train_int = y_train.astype(int)

for x in x_test:
    # rho(x, x(i)) для всех i
    dist = np.sum(np.abs(x_train - x), axis=1)

    # веса K(dist / h)
    w = K(dist / h)

    # суммы по классам: sum_{i: y_i=y} w_i
    scores = np.bincount(y_train_int, weights=w, minlength=3)

    # argmax по классам 0,1,2
    predict.append(int(np.argmax(scores)))

# качество Q(a, X) = доля ошибок
predict = list(predict)
Q = float(np.mean(np.array(predict) != y_test))
