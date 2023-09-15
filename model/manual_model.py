import numpy as np

from .base_model import BaseModel


class ManualModel(BaseModel):
    def __init__(
            self,
            lr: float,
            threshold: float,
            max_iters: int,
            theta: np.array
    ):
        self.n_iters_ = None
        self.lr = lr
        self.threshold = threshold
        self.max_iters = max_iters
        self.theta = theta

    def predict(self, x: np.array) -> np.array:
        return np.dot(np.insert(x, 0, [1], axis=1), self.theta)

    def compute_cost(self, x: np.array, y: np.array) -> float:
        residual = np.power(self.predict(x) - y, 2)
        return np.sum(residual) / (2 * x.shape[0])

    def fit(self, x: np.array, y: np.array) -> np.array:
        x_one = np.insert(x, 0, [1], axis=1)
        cost, i = [], 0
        while i < self.max_iters:
            self.theta -= ((self.lr/y.shape[0]) *
                           np.expand_dims(np.sum(x_one * (self.predict(x) - y), axis=0), axis=1))
            cost.append(self.compute_cost(x, y))

            if i > 0 and np.abs(cost[i - 1] - cost[i]) < self.threshold:
                break

            i += 1

        self.n_iters_ = i

        return np.array(cost)

    def get_num_itrs(self):
        return self.n_iters_

    def get_coefs(self):
        return self.theta
