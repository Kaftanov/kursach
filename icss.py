import numpy as np

from algorithms.algorithms import ICSS
import matplotlib.pyplot as plt


a0 = 0.2
a1 = 0.5
b1 = 0.3

n = 10000
w = np.random.normal(size=n)
eps = np.zeros_like(w)
sigsq = np.zeros_like(w)

for i in range(1, n):
    sigsq[i] = a0 + a1 * (eps[i - 1] ** 2) + b1 * sigsq[i - 1]
    eps[i] = w[i] * np.sqrt(sigsq[i])

eps[:2500] = eps[:2500] + 4
eps[2500:5000] = eps[2500:5000] + 8
eps[5000:7500] = eps[5000:7500] + 12
eps[7500:10000] = eps[7500:10000] + 16


obj = ICSS(time_series=eps)
obj.run()
obj.plot()
plt.show()
