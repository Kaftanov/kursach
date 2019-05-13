import numpy as np


class IT:
    def __init__(self, resid, conditional_volatility):
        self.name = "IT method (Incl´an, Tiao, 1994)"
        self.__q99 = 1.628

        self.std_residuals = resid / conditional_volatility  # should power 2 but I don't know
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
        self.structural_break_factor = np.sqrt(self.len_ / 2) * np.abs(self.compute_it(self.compute_tau()))
        if self.structural_break_factor >= self.__q99:
            print(f"tau it's structural break {self.structural_break_factor} >= {self.__q99}")
            return self.structural_break_factor

    def __str__(self):
        return f"{'=' * 50} \nTau: {self.tau} \nStructural Break Factor {self.structural_break_factor} \n{'=' * 50}"
