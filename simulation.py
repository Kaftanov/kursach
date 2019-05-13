import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

from models.KL import KL

data = pd.read_csv("data\\ciqpriceequity_201905122302.csv")

data.loc[3000::, "priceclose"] = data.loc[3000:, "priceclose"] + 100

am = arch_model(data["priceclose"], p=1, q=1)
res = am.fit(update_freq=10)

model = KL(time_series=res.conditional_volatility)
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

plt.plot(res.conditional_volatility)
plt.plot([model.tau for _ in range(700)], [i for i in range(700)])
plt.show()