import numpy as np
from methods.methods import KL
import matplotlib.pyplot as plt
from arch import arch_model
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
    def __init__(self, data):
        self.y = data.values.copy()
        self.length = self.y.shape[0]

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


class ICSS:
    def __init__(self, data, method=None):
        """

        :param data: pd.Series object or object with values and index attributes
        :param method: method class
        """
        self.y = data.values.copy()
        if method:
            if method == "KL":
                self.core = KL
            else:
                self.core = method
                am = arch_model(data.values)
                res = am.fit(update_freq=5)
                self.y = res.resid / res.conditional_volatility
        else:
            self.core = KL
        self.name = 'ICSS procedure'

        self.index = data.index
        self.T = self.y.shape[0]

        self.d1 = None
        self.d2 = None

        self.breaks = []

    def left_search(self, t_left, t_right):
        left_bound = t_left
        right_bound = t_right
        for _ in range(self.T):
            model = self.core(self.y[left_bound:right_bound])
            if model.evaluate():
                right_bound = model.tau
            else:
                return left_bound + right_bound if left_bound + right_bound < self.T else 0

    def right_search(self, t_left, t_right):
        left_bound = t_left
        right_bound = t_right
        for _ in range(self.T):
            model = self.core(self.y[left_bound:right_bound])
            if model.evaluate():
                left_bound += model.tau + 1
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
            model = self.core(time_series=self.y[breaks[i - 1] + 1: breaks[i + 1]])
            if model.evaluate():
                result.append(breaks[i])
        result.append(breaks[-1])
        return result

    def evaluate(self, clean=False, exact=False):
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
            self.breaks = sorted(list(set(prev + curr)))[1:-1]
        else:
            self.breaks = self.breaks[1:-1]
        if exact:
            pass
        else:
            return self.breaks

    def print_breakpoints(self):
        for break_index in self.breaks:
            print(f'structural break at = {self.index[break_index]} ({break_index})')

    def likelihood(self):
        # TODO add code
        pass

    def plot(self):
        plt.plot(self.index, self.y, color='silver', linestyle='-', marker='.')
        if not self.breaks:
            plt.title("No breaks found")
        else:
            plt.title(f"Found {len(self.breaks)} breaks")
        for break_index in self.breaks:
            plt.axvline(x=self.index[break_index], color='black', linestyle='solid')
        plt.xlabel("Time (Year period)")
        plt.ylabel("Value")
        plt.grid()
