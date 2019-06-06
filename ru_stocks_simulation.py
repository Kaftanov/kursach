import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from algorithms.algorithms import ICSS

ds_name = "GAZP_790601_190606.csv"
# https://ria.ru/20110126/326591553.html
# https://ria.ru/20120214/565530019.html
# https://ria.ru/20120216/567428395.html

data = pd.read_csv(f"data\\{ds_name}")
sample = np.log((data["<CLOSE>"] / data["<CLOSE>"].shift(1)).dropna(axis=0))
base = datetime.datetime.today()
sample.index = [base - datetime.timedelta(days=x) for x in range(0, sample.shape[0])][::-1]

model = ICSS(sample)
model.evaluate()
model.print_breakpoints()
model.plot()
plt.savefig(f"figures\\ru_stock_simulation_{ds_name.split('.')[0].split('_')[0]}.png", format='png', dpi=150, quality=100)
plt.show()
