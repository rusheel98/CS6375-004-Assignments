import numpy as np

from .base_model import BaseModel


class ManualModel(BaseModel):
    def __init__(
            self,
            alpha: float,
            threshold: float,
            iters: int,
            theta: np.array
    ):
        self.alpha = alpha
        self.threshold = threshold
        self.iters = iters
        self.theta = theta

    def predict(self, x: np.array) -> np.array:
        return np.dot(np.insert(x, 0, [1], axis=1), self.theta)

    def compute_cost(self, x: np.array, y: np.array) -> float:
        residual = np.power(self.predict(x) - y, 2)
        return np.sum(residual) / (2 * x.shape[0])

    def fit(self, x: np.array, y: np.array) -> np.array:
        x_one = np.insert(x, 0, [1], axis=1)
        cost, i = [], 0
        while i < self.iters:
            self.theta -= np.expand_dims(
                (self.alpha/x.shape[0]) * np.sum(
                    x_one * (self.predict(x) - y),
                    axis=0
                ),
                axis=1
            )
            cost.append(self.compute_cost(x, y))

            if i > 0 and np.abs(cost[i - 1] - cost[i]) < self.threshold:
                break

            i += 1

        print(f"Stopped after {i} iterations")

        return np.array(cost)
