import numpy as np
from typing import List
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from dataset import Dataset
from model.linear_model import LinearModel
from model.manual_model import ManualModel

import os

from preprocess import Preprocess


class ExperimentData:
    def __init__(self, lr, threshold, max_itrs, train_mse, test_mse, train_adj_r_sqrd, test_adj_r_sqrd):
        decimal = 4
        self.lr = lr
        self.threshold = threshold
        self.max_itrs = max_itrs
        self.train_mse = np.round(train_mse, decimal)
        self.test_mse = np.round(test_mse, decimal)
        self.train_adj_r_sqrd = np.round(train_adj_r_sqrd, decimal)
        self.test_adj_r_sqrd = np.round(test_adj_r_sqrd, decimal)

    def __str__(self):
        return (f"lr: {self.lr}, threshold: {self.threshold}, max_itrs: {self.max_itrs}\n"
                f"train_mse: {self.train_mse}, test_mse: {self.test_mse}\n"
                f"train_adj_r_sqrd: {self.train_adj_r_sqrd}, test_adj_r_sqrd: {self.test_adj_r_sqrd}\n") + \
            "--" * 25 + "\n"

    def __lt__(self, other):
        return self.test_mse < other.test_mse


class Experiments:
    def __init__(
            self,
            data: Dataset,
            log_path: str,
            categorical: List[str],
            target: List[str],
            model_name: str
    ):
        self.data = data
        self.log_path = log_path
        self.model_name = model_name

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        preprocess = Preprocess(data)

        preprocess.remove_nulls()
        preprocess.remove_duplicates()
        preprocess.normalization(categorical, target)
        for column in categorical:
            preprocess.categorical_to_numerical(column)
        preprocess.drop_columns(categorical)
        preprocess.reorder_columns(target)

        self.train_x, self.train_y, self.test_x, self.test_y = self.data.train_test_split(target)

        print(self.train_x.shape, self.train_y.shape, self.test_x.shape, self.test_y.shape)

    def log_experiment_data(self, experiment_data: List[ExperimentData]):
        path = os.path.join(self.log_path, f"{self.model_name}.txt")
        with open(path, "w") as f:
            for experiment_datum in experiment_data:
                f.write(str(experiment_datum))

        print(f"Logged experiment data to {path}")

    def run(self):
        lrs = [1e-1, 1e-2, 1e-3, 1e-4]
        thresholds = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
        max_itrs = [10000, 50000, 100000, 500000, 1000000]

        experiments_data: List[ExperimentData] = []

        for lr in lrs:
            for threshold in thresholds:
                for max_itr in max_itrs:
                    model = None
                    if self.model_name == "LinearModel":
                        model = LinearModel(lr, threshold, max_itr)
                    elif self.model_name == "ManualModel":
                        theta = np.zeros([self.train_x.shape[1] + 1, 1]) + 0.5
                        model = ManualModel(lr, threshold, max_itr, theta=theta)

                    model.fit(self.train_x, self.train_y)
                    train_pred = model.predict(self.train_x)
                    test_pred = model.predict(self.test_x)
                    experiments_data.append(
                        ExperimentData(
                            lr,
                            threshold,
                            max_itr,
                            mean_squared_error(self.train_y, train_pred),
                            mean_squared_error(self.test_y, test_pred),
                            r2_score(self.train_y, train_pred),
                            r2_score(self.test_y, test_pred)
                        )
                    )
                    print(f"Stopped after {model.get_num_itrs()} iterations")
                    print(str(experiments_data[-1]))

                    if model.get_num_itrs() < max_itr:
                        break

        experiments_data.sort()

        self.log_experiment_data(experiments_data)


if __name__ == "__main__":
    features = [
        "sex", "length", "diameter", "height", "whole_weight",
        "shucked_weight", "viscera_weight", "shell_weight", "rings"
    ]

    manual_model_experiments = Experiments(
        Dataset("./data/abalone.data", features),
        "./logs/",
        ["sex"],
        ["rings"],
        "ManualModel"
    )
    manual_model_experiments.run()

    linear_model_experiments = Experiments(
        Dataset("./data/abalone.data", features),
        "./logs/",
        ["sex"],
        ["rings"],
        "LinearModel"
    )
    linear_model_experiments.run()
