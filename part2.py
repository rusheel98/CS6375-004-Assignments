import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor



from dataset import Dataset
from preprocess import Preprocess




features = [
    "sex", "length", "diameter", "height", "whole_weight",
    "shucked_weight", "viscera_weight", "shell_weight", "rings"
]


class linear_model:
    def __init__(self, alpha, eta0, max_iter, threshold):
        self.alpha = alpha
        self.eta0 = eta0
        self.iterations = max_iter
        self.threshold = threshold
        self.model =  SGDRegressor(alpha=self.alpha, eta0=self.eta0, max_iter = self.iterations, tol = self.threshold)
    
    def fit(self, train_x, train_y):
        self.model.fit(train_x, train_y)
    
    def predict(self, data):
        return self.model.predict(data)



if __name__ == "__main__":
    dataset = Dataset("./data/abalone.data", features)
    preprocess = Preprocess(dataset)

    preprocess.remove_nulls()
    preprocess.remove_duplicates()
    preprocess.normalization(["sex"], ["shell_weight", "rings"])
    preprocess.categorical_to_numerical("sex")
    preprocess.drop_columns(["sex"])
    preprocess.reorder_columns(["shell_weight", "rings"])

    train_x, train_y, test_x, test_y = dataset.train_test_split(["rings"])

    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    
    l = linear_model(0.01, 0.001, 10000, 0.00001)
    
    l.fit(train_x,train_y)
    
    train_predict_y = l.predict(train_x)
    mse = (mean_squared_error(train_y, train_predict_y))
    
    test_predict_y = l.predict(test_x)
    mse = (mean_squared_error(test_y, test_predict_y))
    
    print("mse - part2",mse)