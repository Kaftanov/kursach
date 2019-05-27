import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

from methods.methods import KL, LMT, IT
from algorithms.algorithms import ICSS


data = pd.read_csv("data\\ciqpriceequity_201905122302.csv")


# am = arch_model(data["priceclose"], p=1, q=1, vol="GARCH")
# res = am.fit(update_freq=10)
# model = IT(res.resid, res.conditional_volatility)
model = ICSS(np.log(data["priceclose"].values))
model.run()
model.plot()
plt.show()
print(model.breaks)
