from sklearn.linear_model import SGDRegressor

from .base_model import BaseModel


class LinearModel(BaseModel):
    def __init__(self, alpha, eta0, max_iter, threshold):
        self.alpha = alpha
        self.eta0 = eta0
        self.iterations = max_iter
        self.threshold = threshold
        self.model = SGDRegressor(alpha=self.alpha, eta0=self.eta0, max_iter=self.iterations, tol=self.threshold)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)
