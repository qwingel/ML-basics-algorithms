import numpy as np

np.random.seed(0)

n_total = 1000 # число образов выборки
n_features = 200 # число признаков

table = np.zeros(shape=(n_total, n_features))

for _ in range(100):
    i, j = np.random.randint(0, n_total), np.random.randint(0, n_features)
    table[i, j] = np.random.randint(1, 10)

# матрицу table не менять

# здесь продолжайте программу
F = table.T @ table / table.shape[0]

# для симметричной матрицы используем eigh
L, W = np.linalg.eigh(F)              # L по возрастанию
idx = np.argsort(L)[::-1]             # по убыванию
L = L[idx]
WW = W[:, idx]                        # 200x200, столбцы — векторы

data_x = table @ WW                   # 1000x200

# удалить последние k признаков, где lambda < 0.01
k = np.sum(L < 0.01)
if k > 0:
    data_x = data_x[:, :-k]




