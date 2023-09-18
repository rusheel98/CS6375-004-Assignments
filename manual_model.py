import numpy as np

from base_model import BaseModel


class ManualModel(BaseModel):
    def __init__(
            self,
            lr: float,
            threshold: float,
            max_iters: int,
            theta: np.array
    ):
        super().__init__(lr, threshold, max_iters)
        self.theta = theta

    def predict(self, x: np.array) -> np.array:
        return np.dot(np.insert(x, 0, [1], axis=1), self.theta)

    def fit(self, x: np.array, y: np.array, test_x: np.array = None, test_y: np.array = None) -> np.array:
        x_one = np.insert(x, 0, [1], axis=1)
        cost, test_cost, i = [], [], 0
        while i < self.max_iters:
            self.theta -= ((self.lr / y.shape[0]) *
                           np.expand_dims(np.sum(x_one * (self.predict(x) - y), axis=0), axis=1))
            cost.append(self.compute_cost(x, y))

            if test_x is not None and test_y is not None:
                test_cost.append(self.compute_cost(test_x, test_y))

            if i > 0 and np.abs(cost[i - 1] - cost[i]) < self.threshold:
                break

            i += 1

        self.n_iters_ = i

        return np.array(cost), np.array(test_cost)

    def get_num_itrs(self) -> int:
        return self.n_iters_

    def get_coefs(self) -> np.array:
        return np.squeeze(self.theta, axis=1)
