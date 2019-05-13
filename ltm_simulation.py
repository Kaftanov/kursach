import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt

from models.LMT import LMT


np.random.seed(2)

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

eps[:5000] = eps[:5000] + 4

am = arch_model(eps, vol='Garch', p=1, q=1)
res = am.fit(update_freq=5)
plt.plot(am.y)

model_ = LMT(res.resid, res.conditional_volatility)
model_.evaluate()
print(model_)
plt.show()