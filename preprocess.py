from typing import List

import numpy as np
import pandas as pd
from dataset import Dataset


class Preprocess:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.target_mean = 0
        self.target_std = 0

    def remove_nulls(self):
        self.dataset.data.dropna(inplace=True)

    def remove_duplicates(self):
        self.dataset.data.drop_duplicates(inplace=True)

    def categorical_to_numerical(self, column_name: str):
        dummies = pd.get_dummies(self.dataset.data[column_name])
        self.dataset.data = pd.concat([self.dataset.data, dummies], axis='columns')

    def normalization(self, ignore_cols: List[str], target_col: List[str]):
        ignored_cols = self.dataset.data[ignore_cols]
        non_ignored_cols = self.dataset.data.drop(ignore_cols + target_col, axis='columns').apply(
            lambda iterator: ((iterator - iterator.mean()) / iterator.std())
        )

        target = self.dataset.data[target_col]
        self.target_mean = target.mean().to_numpy()
        self.target_std = target.std().to_numpy()

        print(self.target_mean, self.target_std)

        target = (target - self.target_mean) / self.target_std

        self.dataset.data = pd.concat([non_ignored_cols, ignored_cols, target], axis='columns')

    def denormalize_prediction(self, prediction: np.array) -> np.array:
        return (prediction * self.target_std) + self.target_mean

    def drop_columns(self, column_names: List[str]):
        self.dataset.data.drop(column_names, axis=1, inplace=True)

    def reorder_columns(self, column_names: List[str]):
        self.dataset.data = self.dataset.data[
            [col for col in self.dataset.data if col not in column_names]
            + column_names
        ]
