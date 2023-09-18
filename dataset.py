import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt


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

    def plot_top_features_against_corr(self, n: int = 5):
        corr = np.squeeze(self.correlation()[["rings"]][: -1], axis=1)

        feature_coefficients = pd.DataFrame({'Feature': [i for i in self.data.columns if i != 'rings'], 'Correlation': corr})
        feature_coefficients = feature_coefficients.sort_values(by='Correlation', ascending=False)
        top_features = feature_coefficients.head(n)

        plt.barh(top_features['Feature'], top_features['Correlation'], color='b')
        plt.xlabel('Correlation with Target')
        plt.ylabel('Feature')
        plt.title('Important Features based on correlation')
        plt.gca().invert_yaxis()
        plt.show()
