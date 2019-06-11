import pandas as pd
import datetime
import matplotlib.pyplot as plt

from algorithms.algorithms import ICSS
from methods.methods import KL


sample = pd.read_csv("data\\test_data.csv")["Val"]
base = datetime.datetime.today()
sample.index = [base - datetime.timedelta(days=x) for x in range(0, sample.shape[0])]

a = KL(sample.values[0:1099])
a.evaluate()
print(a)

model = ICSS(sample)
model.evaluate()
model.print_breakpoints()
model.plot()
plt.savefig("figures\\test.png", format='png', dpi=150, quality=100)
plt.show()
