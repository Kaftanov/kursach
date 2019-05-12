import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt

from models.KL import KL


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

am = arch_model(eps)
res = am.fit(update_freq=5)
print(res.summary())

print("Fitting KL")
model = KL(am.y)
model.evaluate()

print(f"Results:")
print("=" * 30)
print(f"Tau: {model.tau}")
print("=" * 30)
print(f"V: {model.v}")
print("=" * 30)
print(f"KL: {model.kl_result}")
print("=" * 30)
print(f"SBC: {model.structural_break_factor}")
print("=" * 30)

plt.plot(am.y)
k_min = model.compute_tau_glob()
plt.plot([k_min for _ in range(10)], np.linspace(np.min(am.y), np.max(am.y), 10))
plt.show()
