import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithms.algorithms import ICSS
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import scipy.stats as scs
import statsmodels.tsa.api as smt


ds_name = "qiwi_data.csv"

data = pd.read_csv(f"data\\{ds_name}", parse_dates=["pricingdate"], index_col="pricingdate")

plt.plot(data.index, data["priceclose"].values, color='black', linestyle='-', marker='.')
plt.xlabel("Time (Year period)")
plt.ylabel("Value")
plt.grid()
plt.savefig(f"figures\\source_ts_{ds_name.split('.')[0].split('_')[0]}.png", format='png', dpi=150, quality=100)
plt.show()

sample = np.log((data["priceclose"] / data["priceclose"].shift(1)).dropna(axis=0))

print(f'Adfuller Test: {adfuller(sample.values)[1]}')


model = ICSS(sample)
model.evaluate()
model.print_breakpoints()
model.plot()
plt.savefig(f"figures\\ticker_simulation_{ds_name.split('.')[0].split('_')[0]}.png", format='png', dpi=150, quality=100)
plt.show()
