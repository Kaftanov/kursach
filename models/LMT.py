import numpy as np


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
