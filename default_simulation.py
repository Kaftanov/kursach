import pandas as pd
import datetime
import matplotlib.pyplot as plt

from algorithms.algorithms import ICSS

sample = pd.read_csv("data\\test.csv")
base = datetime.datetime.today()
sample.index = [base - datetime.timedelta(days=x) for x in range(0, sample.shape[0])]
model = ICSS(sample)
model.evaluate()
model.print_breakpoints()
model.plot()
plt.savefig("figures\\test.svg", format='svg', dpi=150, quality=100)
plt.show()
