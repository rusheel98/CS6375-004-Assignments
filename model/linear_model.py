import numpy as np
from sklearn.linear_model import SGDRegressor

from .base_model import BaseModel


class LinearModel(BaseModel):
    def __init__(self, lr: float, threshold: float, max_iters: int):
        super().__init__(lr, threshold, max_iters)
        self.model = SGDRegressor(eta0=self.lr, penalty=None, max_iter=self.max_iters,
                                  tol=self.threshold, early_stopping=False, learning_rate='constant', random_state=0)

    def fit(self, x: np.array, y: np.array, test_x: np.array = None, test_y: np.array = None) -> np.array:
        y = y.ravel()
        cost, test_cost, i = [], [], 0
        while i < self.max_iters:
            self.model.partial_fit(x, y)
            cost.append(self.compute_cost(x, y))

            if test_x is not None and test_y is not None:
                test_cost.append(self.compute_cost(test_x, test_y))

            if i > 0 and np.abs(cost[i - 1] - cost[i]) < self.threshold:
                break

            i += 1

        self.n_iters_ = i

        return np.array(cost), np.array(test_cost)

    def predict(self, x):
        return self.model.predict(x)

    def get_num_itrs(self) -> int:
        return self.n_iters_

    def get_coefs(self) -> np.array:
        return np.concatenate((self.model.intercept_, self.model.coef_), axis=0)
