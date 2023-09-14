from typing import List

import pandas as pd


class Dataset:
    def __init__(self, dataset_name: str, col_names: List[str] = None):
        self.dataset_name = dataset_name
        self.col_names = col_names
        if not col_names:
            self.data = pd.read_csv(self.dataset_name)
        else:
            self.data = pd.read_csv(self.dataset_name, names=col_names)

    def __str__(self):
        return str(self.data.head(5))

    def train_test_split(self, y_col: List[str], split: float = 0.8, random_state: int = 200):
        train = self.data.sample(frac=split, random_state=random_state)
        test = self.data.drop(train.index)

        train_x = train.drop(y_col, axis=1).to_numpy()
        train_y = train[y_col].to_numpy()

        test_x = test.drop(y_col, axis=1).to_numpy()
        test_y = test[y_col].to_numpy()

        return train_x, train_y, test_x, test_y

    def correlation(self):
        return self.data.corr()
