import numpy as np


class IT:
    def __init__(self, resid, conditional_volatility):
        self.name = "IT method (Incl´an, Tiao, 1994)"
        self.__q99 = 1.628

        self.std_residuals = np.power(resid / conditional_volatility, 2)  # should power 2 but I don't know
        self.len_ = self.std_residuals.shape[0]

        self.tau = None
        self.structural_break_factor = None

    def compute_it(self, k):
        return (np.sum(self.std_residuals[:k]) / np.sum(self.std_residuals)) - (k / self.len_)

    def compute_it_glob(self):
        result = np.zeros(self.len_)
        sum_ = np.sum(self.std_residuals)
        for k in range(1, self.len_ + 1):
            result[k - 1] = (np.sum(self.std_residuals[:k]) / sum_) - (k / self.len_)
        return result

    def compute_tau(self):
        return np.argmax(np.abs(self.compute_it_glob())) + 1

    def evaluate(self):
        self.tau = self.compute_tau()
        self.structural_break_factor = np.sqrt(self.len_ / 2) * np.abs(self.compute_it(self.tau))
        if self.structural_break_factor >= self.__q99:
            print(f"tau it's structural break {self.structural_break_factor} >= {self.__q99}")
            return self.structural_break_factor

    def __str__(self):
        return f"{'=' * 50} \nTau: {self.tau} \nStructural Break Factor {self.structural_break_factor} \n{'=' * 50}"


class KL:
    def __init__(self, time_series, info=False):
        """

        :param time_series:
        """
        self._name = "KL-CUSUM"
        self._q99 = 1.628
        self._q95 = 1.358
        self.info = info
        self.y = time_series.copy()
        self.y_len = self.y.shape[0]
        self.y_mean = np.mean(self.y)
        self.y_mean_sq = np.power(self.y_mean, 2)
        self.r_estimate = int(np.sqrt(self.y_len) // 1)  # int(np.log(self.y_len) // 1)  #

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
        return res if not np.all(res) else np.array([0])

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
        if self.structural_break_factor >= self._q95:
            if self.info:
                print(f"tau it's structural break {self.structural_break_factor} >= {self._q95}")
            return self.structural_break_factor

    def __str__(self):
        return f"{'=' * 50} \nTau: {self.tau} \nStructural Break Factor {self.structural_break_factor} \n{'=' * 50}"


class LMT:
    def __init__(self, resid, conditional_volatility):
        self.__name = "LTM method (Lee, Tokutsu, Maekawa, 2004)"
        self.__q99 = 1.628

        self.y = resid / conditional_volatility
        self.len_ = self.y.shape[0]

        self.y2 = np.power(self.y, 2)
        self.eta = (np.sum(np.power(self.y, 4)) / self.len_) - np.power(np.sum(self.y2) / self.len_, 2)

        self.tau = None
        self.structural_break_factor = None

    def compute_ltm(self, k):
        return np.abs(np.sum(self.y2[:k]) - (k / self.len_) * np.sum(self.y2)) / np.sqrt(self.len_ * self.eta)

    def compute_ltm_glob(self):
        result = np.zeros(self.len_)
        sum_ = np.sum(self.y2) / self.len_
        div_ = 1 / np.sqrt(self.len_ * self.eta)
        for k in range(1, self.len_ + 1):
            result[k - 1] = div_ * np.abs(np.sum(self.y2[:k]) - (k * sum_))
        return result

    def compute_tau(self):
        return np.argmax(np.abs(self.compute_ltm_glob())) + 1

    def evaluate(self):
        self.structural_break_factor = self.compute_ltm(self.compute_tau())
        if self.structural_break_factor >= self.__q99:
            print(f"tau it's structural break {self.structural_break_factor} >= {self.__q99}")
            return self.structural_break_factor

    def __str__(self):
        return f"{'=' * 50} \nTau: {self.tau} \nStructural Break Factor {self.structural_break_factor} \n{'=' * 50}"