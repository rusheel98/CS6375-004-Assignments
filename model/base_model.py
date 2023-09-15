class BaseModel:
    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def get_num_itrs(self):
        raise NotImplementedError

    def get_coefs(self):
        raise NotImplementedError

    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x)
