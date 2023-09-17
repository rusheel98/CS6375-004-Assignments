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

    manual_model_experiments = Experiments(
        Dataset("./data/abalone.data", features),
        "./logs/",
        ["sex"],
        ["rings"],
        "ManualModel"
    )
    best_manual_model = manual_model_experiments.run()
