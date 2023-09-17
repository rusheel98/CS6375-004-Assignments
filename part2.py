from dataset import Dataset
from experiments import Experiments

features = [
    "sex", "length", "diameter", "height", "whole_weight",
    "shucked_weight", "viscera_weight", "shell_weight", "rings"
]


if __name__ == "__main__":
    features = [
        "sex", "length", "diameter", "height", "whole_weight",
        "shucked_weight", "viscera_weight", "shell_weight", "rings"
    ]

    linear_model_experiments = Experiments(
        Dataset("./data/abalone.data", features),
        "./logs/",
        ["sex"],
        ["rings"],
        "LinearModel"
    )
    best_linear_model = linear_model_experiments.run()
