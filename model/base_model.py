import numpy as np
from sklearn.metrics import mean_squared_error


class BaseModel:
    def __init__(self, lr: float, threshold: float, max_iters: int):
        self.n_iters_ = None
        self.lr = lr
        self.max_iters = max_iters
        self.threshold = threshold

    def fit(self, x: np.array, y: np.array, test_x: np.array = None, test_y: np.array = None):
        raise NotImplementedError

    def predict(self, x: np.array):
        raise NotImplementedError

    def get_num_itrs(self):
        raise NotImplementedError

    def get_coefs(self):
        raise NotImplementedError

    def fit_predict(self, x: np.array, y: np.array, test_x: np.array = None, test_y: np.array = None):
        self.fit(x, y, test_x, test_y)
        return self.predict(x)

    def compute_cost(self, x: np.array, y: np.array) -> float:
        return mean_squared_error(y, self.predict(x))
