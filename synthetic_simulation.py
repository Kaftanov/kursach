import datetime
import pandas as pd
import matplotlib.pyplot as plt

from algorithms.simulate import GARCH
from algorithms.algorithms import ICSS

sample = pd.Series(GARCH().generate_sample(breaks=5, length=3000, break_weight=0.5))
base = datetime.datetime.today()
sample.index = [base - datetime.timedelta(days=x) for x in range(0, sample.shape[0])]
model = ICSS(sample)
model.evaluate()
model.print_breakpoints()
model.plot()
plt.savefig("figures\\synthetic_garch_test.svg", format='svg', dpi=150, quality=100)
plt.show()
