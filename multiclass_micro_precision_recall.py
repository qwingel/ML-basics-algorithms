import numpy as np

# micro
TP = [45, 37, 51, 47]
TN = [36, 37, 29, 28]
FP = [18, 21, 15, 17]
FN = [8, 11, 9, 5]

precision = np.mean(TP) / (np.mean(TP) + np.mean(FP))
recall = np.mean(TP) / (np.mean(TP) + np.mean(FN))