import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from dataset import Dataset
from model.base_model import BaseModel
from model.linear_model import LinearModel
from model.manual_model import ManualModel

import os

from preprocess import Preprocess


class ExperimentData:
    def __init__(self, model: BaseModel, model_name: str, lr: float, threshold: float, max_itrs: int, num_itrs: int,
                 train_mse: float, test_mse: float, train_adj_r_sqrd: float,
                 test_adj_r_sqrd: float, coeffs: np.array, costs: np.array, test_costs: np.array):
        decimal = 4
        self.model = model
        self.model_name = model_name
        self.lr = lr
        self.threshold = threshold
        self.max_itrs = max_itrs
        self.num_itrs = num_itrs
        self.train_mse = np.round(train_mse, decimal)
        self.test_mse = np.round(test_mse, decimal)
        self.train_adj_r_sqrd = np.round(train_adj_r_sqrd, decimal)
        self.test_adj_r_sqrd = np.round(test_adj_r_sqrd, decimal)

        self.coeffs = coeffs
        self.costs = costs
        self.test_costs = test_costs

    def plot_train_vs_test(self):
        plt.plot(np.arange(self.num_itrs + 1), self.costs, 'r', label="train cost")
        plt.plot(np.arange(self.num_itrs + 1), self.test_costs, 'b', label="test cost")
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title(f'Error for train vs test ({self.model_name})')
        plt.legend()
        plt.show()

    def plot_iters_vs_cost(self, other=None):
        self_costs = self.costs
        if other:
            max_length = max(len(self.costs), len(other.costs))
            self_costs = np.pad(self.costs, (0, max_length - len(self.costs)), mode='edge')
            other_costs = np.pad(other.costs, (0, max_length - len(other.costs)), mode='edge')
            plt.plot(np.arange(len(other_costs)), other_costs, 'b', label=other.model_name)
        plt.plot(np.arange(len(self_costs)), self_costs, 'r', label=self.model_name)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        if other:
            plt.title(f'Error against Training Epoch for {self.model_name} vs {other.model_name}')
        else:
            plt.title(f'Error against Training Epoch for {self.model_name}')
        plt.legend()
        plt.show()

    def plot_top_features_against_coeffs(self, x, n: int = 5):
        feature_coefficients = pd.DataFrame({'Feature': x.columns[1:, ], 'Coefficient': self.coeffs[1:, ]})
        feature_coefficients = feature_coefficients.sort_values(by='Coefficient', ascending=False)
        top_features = feature_coefficients.head(n)

        plt.barh(top_features['Feature'], top_features['Coefficient'], color='b')
        plt.xlabel('Coefficient')
        plt.ylabel('Feature')
        plt.title(f'Top {n} Important Features for {self.model_name} based on model weights')
        plt.gca().invert_yaxis()
        plt.show()

    def plot_model_line_for_each_feature(self, x, y, cols):
        bias, coeffs = self.coeffs[0,], self.coeffs[1:, ]

        fig, axes = plt.subplots(2, (len(cols) + 1) // 2, figsize=(15, 6))
        for i in range(2):
            for j in range((len(cols) + 1) // 2):
                subplot_index = i * len(cols) // 2 + j
                feature_name = cols[subplot_index]
                top_feature_values = x[:, subplot_index]
                top_feature_values_range = np.linspace(np.min(top_feature_values), np.max(top_feature_values), 100)

                # assuming other feature values are 0
                y_pred_range = top_feature_values_range.reshape(-1, 1) * coeffs[subplot_index,].reshape(-1, 1) + bias

                ax = axes[i, j]
                ax.scatter(top_feature_values, y, alpha=0.5, label='Actual Data')
                ax.plot(top_feature_values_range, y_pred_range, color='red', linewidth=2,
                        label=f'Regression Line ({self.model_name})')
                ax.set_xlabel(f"{feature_name}")
                ax.legend()

        plt.tight_layout()
        fig.suptitle(f'Actual Data vs. Regression Line for features ({self.model_name})')
        plt.show()

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
            ignore: List[str],
            model_name: str,
            plot_corr: bool = False
    ):
        self.data = data
        self.log_path = log_path
        self.model_name = model_name

        self.target = target

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        preprocess = Preprocess(data)

        preprocess.drop_columns(ignore)
        preprocess.remove_nulls()
        preprocess.remove_duplicates()
        preprocess.normalization(categorical, target)
        for column in categorical:
            preprocess.categorical_to_numerical(column)
        preprocess.drop_columns(categorical)
        preprocess.reorder_columns(target)

        self.feature_cols = [i for i in self.data.data.columns if i not in self.target]

        self.train_x, self.train_y, self.test_x, self.test_y = self.data.train_test_split(target)

        if plot_corr:
            self.data.plot_top_features_against_corr()

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
                        theta = np.zeros([self.train_x.shape[1] + 1, 1])
                        model = ManualModel(lr, threshold, max_itr, theta=theta)

                    costs, test_costs = model.fit(self.train_x, self.train_y, self.test_x, self.test_y)
                    train_pred = model.predict(self.train_x)
                    test_pred = model.predict(self.test_x)
                    experiments_data.append(
                        ExperimentData(
                            model, self.model_name, lr,
                            threshold,
                            max_itr,
                            model.get_num_itrs(),
                            mean_squared_error(self.train_y, train_pred),
                            mean_squared_error(self.test_y, test_pred),
                            r2_score(self.train_y, train_pred),
                            r2_score(self.test_y, test_pred),
                            model.get_coefs(),
                            costs, test_costs
                        )
                    )
                    print(f"Stopped after {model.get_num_itrs()} iterations")
                    print(str(experiments_data[-1]))

                    if model.get_num_itrs() < max_itr:
                        break

        experiments_data.sort()

        self.log_experiment_data(experiments_data)
        experiments_data[0].plot_top_features_against_coeffs(self.data.data)
        experiments_data[0].plot_model_line_for_each_feature(self.train_x, self.train_y, self.feature_cols)

        return experiments_data[0]


if __name__ == "__main__":
    features = [
        "sex", "length", "diameter", "height", "whole_weight",
        "shucked_weight", "viscera_weight", "shell_weight", "rings"
    ]

    manual_model_experiments = Experiments(
        Dataset("./data/abalone.data", features),
        "./logs/",
        [],
        ["rings"],
        ["sex"],
        "ManualModel",
        True
    )
    best_manual_model = manual_model_experiments.run()

    linear_model_experiments = Experiments(
        Dataset("./data/abalone.data", features),
        "./logs/",
        [],
        ["rings"],
        ["sex"],
        "LinearModel"
    )
    best_linear_model = linear_model_experiments.run()

    best_manual_model.plot_iters_vs_cost(best_linear_model)
    best_manual_model.plot_train_vs_test()
    best_linear_model.plot_train_vs_test()
