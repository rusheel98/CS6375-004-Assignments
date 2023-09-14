class BaseModel:
    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x)
