import numpy as np


class KL:
    def __init__(self, time_series, disp=False):
        """

        :param time_series:
        """
        self.__name = "KL-CUSUM"
        self.__q99 = 1.628
        self.info = disp
        self.y = time_series.copy()
        self.y_len = self.y.shape[0]
        self.y_mean = np.mean(self.y)
        self.y_mean_sq = np.power(self.y_mean, 2)
        self.r_estimate = int(np.sqrt(self.y_len) // 1)

        self.tau = None
        self.v = None
        self.kl_result = None
        self.structural_break_factor = None

    def compute_kl(self, k):
        return (np.sum(np.power(self.y[:k], 2)) - k * np.sum(np.power(self.y, 2)) / self.y_len) / np.sqrt(self.y_len)

    def compute_kl_glob(self):
        if self.info:
            print("Computing KL for all observations")
        a1 = 1 / np.sqrt(self.y_len)
        s2 = np.sum(np.power(self.y, 2))
        res = np.zeros(self.y_len)
        for i in range(1, self.y_len + 1):
            a2 = i / self.y_len
            s1 = np.sum(np.power(self.y[:i], 2))
            res[i - 1] = (s1 - a2 * s2) * a1
        return res if res else np.array([0])

    def compute_tau(self):
        res = np.zeros(self.y_len)
        for i in range(1, self.y_len + 1):
            res[i] = np.abs(self.compute_kl(i))
        return np.argmax(np.abs(res)) + 1

    def compute_tau_glob(self):
        if self.info:
            print("Searching index of shift")
        return np.argmax(np.abs(self.compute_kl_glob())) + 1

    def compute_w(self, j):
        return 1 - (j / (self.r_estimate + 1))

    def compute_c(self, j):
        temp_sum = 0
        for i in range(self.y_len - j):
            temp_sum += (np.power(self.y[i], 2) - self.y_mean_sq) * (np.power(self.y[i + j], 2) - self.y_mean_sq)
        return temp_sum / self.y_len

    def compute_coefficients(self, j):
        return self.compute_w(j) * self.compute_c(j)

    def compute_v(self):
        if self.info:
            print("Computing v")
        result = np.zeros(self.r_estimate)
        for i in range(self.r_estimate):
            result[i] = self.compute_coefficients(i)
        return np.sqrt(np.sum(result))

    def evaluate(self):
        self.tau = self.compute_tau_glob()
        self.v = self.compute_v()
        self.kl_result = np.abs(self.compute_kl(self.tau))
        self.structural_break_factor = self.kl_result / self.v
        if self.structural_break_factor >= self.__q99:
            if self.info:
                print(f"tau it's structural break {self.structural_break_factor} >= {self.__q99}")
            return self.structural_break_factor

    def __str__(self):
        return f"{'=' * 50} \nTau: {self.tau} \nStructural Break Factor {self.structural_break_factor} \n{'=' * 50}"
