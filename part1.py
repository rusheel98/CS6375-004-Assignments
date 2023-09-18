import sys
from dataset import Dataset
from experiments import Experiments


features = [
    "sex", "length", "diameter", "height", "whole_weight",
    "shucked_weight", "viscera_weight", "shell_weight", "rings"
]

if __name__ == "__main__":
    data_path = "./data/abalone.data" if len(sys.argv) == 0 else sys.argv[0]

    manual_model_experiments = Experiments(
        Dataset(data_path, features),
        "./logs/",
        [],
        ["rings"],
        ["sex"],
        "ManualModel",
    )
    best_manual_model = manual_model_experiments.run()
