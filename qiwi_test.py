import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithms.algorithms import ICSS
from statsmodels.tsa.stattools import adfuller


plt.rcParams['figure.figsize'] = 15, 5

ds_name = "QIWI_790101_190612.csv"


data = pd.read_csv(f"data\\{ds_name}").iloc[:1401]
data["<DATE>"] = pd.to_datetime(data["<DATE>"].astype(str), format=r'%Y%m%d')
data.set_index("<DATE>", inplace=True)


print(f'Min date: {min(data.index)}')
print(f'Max date: {max(data.index)}')
print(f'Data length: {data.shape[0] - 1}')
sample = np.log((data["<CLOSE>"] / data["<CLOSE>"].shift(1)).dropna(axis=0))

print(f'Adfuller Test: {adfuller(sample.values, maxlag=1)[1]}')

plt.plot(data.index, data["<CLOSE>"].values, color='black', linestyle='-', marker='.')
plt.xlabel("Time (Year period)")
plt.ylabel("Value")
plt.grid()
plt.savefig(f"figures\\source_ts_{ds_name.split('.')[0].split('_')[0]}.png", format='png', dpi=150, quality=100)
plt.show()

plt.plot(sample.index, sample.values, color='black', linestyle='-', marker='.')
plt.xlabel("Time (Year period)")
plt.ylabel("Value (ln)")
plt.grid()
plt.savefig(f"figures\\source_ln_ts_{ds_name.split('.')[0].split('_')[0]}.png", format='png', dpi=150, quality=100)
plt.show()

model = ICSS(sample, method="KL")
model.evaluate()
model.print_breakpoints()
model.plot()
plt.savefig(f"figures\\ru_stock_simulation_{ds_name.split('.')[0].split('_')[0]}.png", format='png', dpi=150, quality=100)
plt.show()

plt.plot(data["<CLOSE>"].index, data["<CLOSE>"].values, color='black', linestyle='-', marker='.')
plt.xlabel("Time (Year period)")
plt.ylabel("Value (ln)")
plt.grid()
for break_index in model.breaks:
    plt.axvline(x=data["<CLOSE>"].index[break_index], color='black', linestyle='solid')
plt.show()