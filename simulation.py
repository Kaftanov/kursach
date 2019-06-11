import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithms.algorithms import ICSS

ds_name = "gaz_data.csv"

data = pd.read_csv(f"data\\{ds_name}", parse_dates=["pricingdate"], index_col="pricingdate")
sample = 100 * data["priceclose"].pct_change().dropna()
model = ICSS(sample)
model.evaluate()
model.print_breakpoints()
model.plot()
plt.savefig(f"figures\\ticker_simulation_{ds_name.split('.')[0].split('_')[0]}.png", format='png', dpi=150, quality=100)
plt.show()
