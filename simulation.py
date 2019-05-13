import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

from models.KL import KL
from models.LMT import LMT
from models.IT import IT


data = pd.read_csv("data\\ciqpriceequity_201905122302.csv")

data.loc[3000::, "priceclose"] = data.loc[3000:, "priceclose"] + 400

am = arch_model(data["priceclose"], p=1, q=1, vol="GARCH")
res = am.fit(update_freq=10)

model = IT(res.resid, res.conditional_volatility)
model.evaluate()
print(model)

plt.plot(data["priceclose"])
plt.plot([model.tau for _ in range(100)], [i for i in range(100)])
plt.show()
