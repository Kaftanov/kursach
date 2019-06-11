import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from algorithms.algorithms import ICSS

plt.rcParams['figure.figsize'] = 15, 5

ds_name = "LKOH_790601_190606.csv"
# https://ria.ru/20110126/326591553.html
# https://ria.ru/20120214/565530019.html
# https://ria.ru/20120216/567428395.html
# https://oilcapital.ru/news/export/03-02-2012/eksport-rossiyskih-nefteproduktov-cherez-rpk-vysotsk-lukoyl-ii-za-yanvar-2012-goda-snizilsya-na-6-do-907-tys-tonn

data = pd.read_csv(f"data\\{ds_name}")
base = datetime.datetime.today()
data.index = [base - datetime.timedelta(days=x) for x in range(0, data.shape[0])][::-1]
sample = np.log((data["<CLOSE>"] / data["<CLOSE>"].shift(1)).dropna(axis=0))


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
