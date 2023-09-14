from sklearn.metrics import mean_squared_error

from dataset import Dataset
from preprocess import Preprocess
from model.linear_model import LinearModel

features = [
    "sex", "length", "diameter", "height", "whole_weight",
    "shucked_weight", "viscera_weight", "shell_weight", "rings"
]


if __name__ == "__main__":
    dataset = Dataset("./data/abalone.data", features)
    preprocess = Preprocess(dataset)

    preprocess.remove_nulls()
    preprocess.remove_duplicates()
    preprocess.normalization(["sex"], ["rings"])
    preprocess.categorical_to_numerical("sex")
    preprocess.drop_columns(["sex"])
    preprocess.reorder_columns(["rings"])

    train_x, train_y, test_x, test_y = dataset.train_test_split(["rings"])

    model = LinearModel(0.01, 0.001, 10000, 0.00001)

    train_pred = model.fit_predict(train_x, train_y)
    print("mse [train] - part2", mean_squared_error(train_y, train_pred))
    print("mse [test] - part2", mean_squared_error(test_y, model.predict(test_x)))
