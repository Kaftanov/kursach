#  Check  Kolmogorov–Smirnov theorem

import numpy as np
from methods.methods import KL
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


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
    return np.max(result)


class KS:
    def __init__(self, data_array):
        self.y = data_array.copy()
        self.length = data_array.shape[0]

        self.d1, self.d2 = np.random.randint(low=1, high=self.length - 1, size=2)
        self.d_l = None
        self.d_r = None

        self.tau_ks = None
        self.tau_l = None
        self.tau_r = None

    def split(self, series, k, type_=1):
        if type_ == 1:
            return series[:k], series[k:]
        elif type_ == 2:
            return series[:int((k / 2) // 1)], series[int((k / 2) // 1) + 1:]

    def step_eval(self, k):
        series1, series2 = self.split(self.y, k, type_=1)
        sub_series1, sub_series2 = self.split(series1, k, type_=2)
        sub_series3, sub_series4 = self.split(series2, k, type_=2)

        self.d_l = dist(sub_series1, sub_series2)
        self.d_r = dist(sub_series3, sub_series4)
        return self.d_l, self.d_r

    def evaluate(self):
        tau_ks = 0
        min_ = -1
        print(self.d1, self.length - self.d1)
        for k in range(self.d1, self.length - self.d1):
            d_l, d_r = self.step_eval(k)
            print(d_l + d_r)
            if d_l + d_r <= min_:
                min_ = d_l + d_r
                tau_ks = k
        self.tau_ks = tau_ks
        self.tau_l = max(tau_ks - self.d2, self.d1)
        self.tau_r = min(tau_ks + self.d2, self.d1)
        return self.tau_ks, self.tau_l, self.tau_r

    def simulate(self):
        X = np.random.normal(0, 1, 1000)
        model = KS(data_array=X)
        print(model.evaluate())


class ICSS:
    def __init__(self, time_series):
        self.name = 'ICSS procedure'
        self.y = time_series
        self.T = self.y.shape[0]
        self.breaks = []

    def left_search(self, t_left, t_right):
        left_bound = t_left
        right_bound = t_right
        for _ in range(self.T):
            # print(t1, t2)
            model = KL(self.y[left_bound:right_bound])
            if model.evaluate():
                # print(model.tau)
                right_bound = model.tau
            else:
                # print(f"l_b = {left_bound} r_b = {right_bound}")
                return left_bound + right_bound if left_bound + right_bound < self.T else 0

    def right_search(self, t_left, t_right):
        left_bound = t_left
        right_bound = t_right
        for _ in range(self.T):
            # print(t1, t2)
            model = KL(self.y[left_bound:right_bound])
            if model.evaluate():
                # print(model.tau)
                left_bound += model.tau + 1
                # print(f"l = {left_bound}")
            else:
                return left_bound - 1

    def search(self, t_init_left, t_init_right):
        t_first = self.left_search(t_left=t_init_left, t_right=t_init_right)
        t_last = self.right_search(t_left=t_first + 1, t_right=t_init_right)
        results = [t_init_left, t_first, t_last, t_init_right]
        while t_first < t_last:
            t_first = self.left_search(t_left=t_first, t_right=t_last)
            t_last = self.right_search(t_left=t_first + 1, t_right=t_last)
            results.append(t_first)
            results.append(t_last)
        return sorted(list(set(results)))

    def select_breaks(self, breaks):
        result = [breaks[0]]
        for i in range(1, len(breaks) - 1):
            model = KL(time_series=self.y[breaks[i - 1] + 1: breaks[i + 1]])
            if model.evaluate():
                result.append(breaks[i])
        result.append(breaks[-1])
        return result

    def run(self, clean=False):
        self.breaks = self.search(0, self.T)
        if clean:
            prev = self.select_breaks(self.select_breaks(self.breaks))
            curr = self.select_breaks(self.breaks)
            for i in range(100):
                if len(prev) == len(curr):
                    break
                else:
                    prev = self.select_breaks(prev)
                    curr = self.select_breaks(curr)
            print(prev, curr)
            self.breaks = sorted(list(set(prev + curr)))[1:-1]
            return self.breaks
        else:
            self.breaks = self.breaks[1:-1]
            return self.breaks

    def plot(self):
        plt.plot(self.y)
        if not self.breaks:
            plt.title("No breaks found")
        else:
            plt.title(f"Found {len(self.breaks)} breaks")
        for val in self.breaks:
            plt.axvline(x=val, color='r')
        plt.xlabel("Time (t)")
        plt.ylabel("Value (val)")
        plt.grid()
