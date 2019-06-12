import numpy as np


class GARCH:
    def __init__(self, w=None, a=None, b=None):
        self.name = "GARCH model simulation"
        self.w = w if w else 0.1
        self.a = a if a else [0.3]
        self.b = b if b else [0.2]

    def generate_sample(self, length=1000, breaks=None, break_weight=1):
        p = len(self.a)
        q = len(self.b)

        drop_size = np.max([p, q]) * 100
        n = length + drop_size
        eps = np.random.normal(0, 1, n)
        y = np.zeros(n)
        sigma = np.zeros(n)

        for i in range(np.max([p, q])):
            sigma[i] = np.random.normal(0, 1)
            y[i] = sigma[i] * eps[i]

        for i in range(np.max([p, q]), n):
            temp_a = sum([self.a[j] * y[i - j]**2 for j in range(p)])
            temp_b = np.sum([self.b[j] * sigma[i - j]**2 for j in range(q)])
            sigma[i] = np.sqrt(self.w + temp_a + temp_b)
            y[i] = sigma[i] * eps[i]
        if not breaks:
            return y[drop_size:]
        else:
            for i in range(breaks):
                index = np.random.randint(low=3, high=y[drop_size:].shape[0])
                print(f"generated break index = {index}")
                y[drop_size + index:] = y[drop_size + index:] + np.random.choice([1, -1]) * break_weight
            return y[drop_size:]

    def get_sample(self, length=1000, breaks=None):
        if not breaks:
            self.generate_sample(length=length, breaks=breaks)
        else:
            result = []
            for i in range(breaks):
                self.a[0] *= np.random.choice([-2, 2])
                self.w += np.random.choice([0.4, -1, 1.7])
                result.append(self.generate_sample(length=length // breaks, breaks=None))
            return [item for sub_item in result for item in sub_item]
