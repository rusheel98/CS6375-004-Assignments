from sklearn.linear_model import SGDRegressor

from .base_model import BaseModel


class LinearModel(BaseModel):
    def __init__(
            self,
            lr: float,
            threshold: float,
            max_iters: int
    ):
        self.lr = lr
        self.iterations = max_iters
        self.threshold = threshold
        self.model = SGDRegressor(eta0=self.lr, penalty=None, max_iter=self.iterations,
                                  tol=self.threshold, early_stopping=True, learning_rate='constant')

    def fit(self, x, y):
        y = y.ravel()
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def get_num_itrs(self):
        return self.model.n_iter_

    def get_coefs(self):
        return self.model.coef_
