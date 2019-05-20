#  Check  Kolmogorov–Smirnov theorem

import numpy as np
import matplotlib.pyplot as plt



def edf(x_array, t):
    """
    Empirical distribution function
    :param x_array: np.array
    :param t: argument of F(x)
    :return:
    """
    return np.sum(x_array[x_array <= t]) / x_array.shape[0]


def dist(x_array, y_array, upper_bound=None, complexity_coef=1000):
    if upper_bound:
        t_range = upper_bound
    else:
        t_range = np.linspace(min(x_array.min(), y_array.min()), max(x_array.max(), y_array.max()),
                              (x_array.shape[0] + y_array.shape[0]) * complexity_coef)

    result = np.zeros(t_range.shape[0])

    for index, t in enumerate(t_range):
        result[index] = np.abs(edf(x_array, t) - edf(y_array, t))
    return result


def split_series(series, grid_coef=3):
    best_splitter = np.random.randint(low=int((series.shape[0]/grid_coef)//1),
                                      high=series.shape[0] - int((series.shape[0]/grid_coef)//1))
    print(best_splitter)
    return series[:best_splitter], series[best_splitter:]


class KS:
    def __init__(self, data_array):
        self.y = data_array.copy()
        self.length = data_array.shape[0]

        self.d_l = None
        self.d_r = None

    def evaluate(self):
        series1, series2 = split_series(self._y, grid_coef=3)
        sub_series1, sub_series2 = split_series(series1, grid_coef=2)
        sub_series3, sub_series4 = split_series(series2, grid_coef=2)

        self.d_l = dist(sub_series1, sub_series2)
        self.d_r = dist(sub_series3, sub_series4)


X = np.random.normal(0, 1, 100)
Y = np.random.normal(1, 2, 100)
plt.plot(X)
x1, x2 = split_series(X, grid_coef=3)
plt.plot([x1.shape[0] for _ in range(10)], [i for i in range(-5, 5)])
x3, x4 = split_series(x1, grid_coef=2)
plt.plot([x3.shape[0] for _ in range(10)], [i for i in range(-5, 5)])
plt.show()
