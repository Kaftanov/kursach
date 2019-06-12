import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithms.algorithms import ICSS

ds_name = "yndx_data.csv"

data = pd.read_csv(f"data\\{ds_name}", parse_dates=["pricingdate"], index_col="pricingdate")
sample = np.log((data["priceclose"] / data["priceclose"].shift(1)).dropna(axis=0))
# sample = 100 * data["priceclose"].pct_change().dropna()

plt.plot(data.index, data["priceclose"].values, color='black', linestyle='-', marker='.')
plt.xlabel("Time (Year period)")
plt.ylabel("Value")
plt.grid()
plt.savefig(f"figures\\en_source_ts_{ds_name.split('.')[0].split('_')[0]}.png", format='png', dpi=150, quality=100)
plt.show()


model = ICSS(sample)
model.evaluate()
model.print_breakpoints()
model.plot()
plt.savefig(f"figures\\ticker_simulation_{ds_name.split('.')[0].split('_')[0]}.png", format='png', dpi=150, quality=100)
plt.show()
