import numpy as np

rub_usd = np.array([75, 76, 79, 82, 85, 81, 83, 86, 87, 85, 83, 80, 77, 79, 78, 81, 84], dtype=float)

h = 3.0

def K(r):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * r * r)

predict = []
series = rub_usd.tolist()

for _ in range(10):
    x_new = len(series)                 # 17, 18, 19, ...
    print(x_new)
    xs = np.arange(len(series))         # 0..(len-1)
    print(xs)
    w = K(np.abs(x_new - xs) / h)       # веса по расстоянию индексов
    y_new = np.dot(w, series) / w.sum() # Надарая–Ватсон
    predict.append(y_new)
    series.append(y_new)                # важно: используем прогноз дальше
