import numpy as np

from dataset import Dataset
from preprocess import Preprocess
from manual_model import ManualModel


features = [
    "sex", "length", "diameter", "height", "whole_weight",
    "shucked_weight", "viscera_weight", "shell_weight", "rings"
]

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

    model = ManualModel(
        0.01, 0.00001, 10000,
        np.zeros([train_x.shape[1]+1, 1])
    )

    cost = model.fit(train_x, train_y)

    print(model.compute_cost(train_x, train_y), model.compute_cost(test_x, test_y))
    print(preprocess.denormalize_prediction(model.predict(test_x)))
