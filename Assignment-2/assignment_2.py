import pandas as pd
from model import NeuralNetwork

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv(
    'https://raw.githubusercontent.com/chaitanya-basava/CS6375-004-Assignment-1-data/main/NHANES_age_prediction.csv'
)
data.drop(["RIAGENDR"], axis=1)
classes = ['Adult', 'Senior']
for i in range(len(classes)):
    data["age_group"].replace(classes[i], i, inplace=True)

columns = ["RIDAGEYR", "BMXBMI", "LBXGLU", "DIQ010", "LBXGLT", "LBXIN"]

param_grid = {
    'activation': ['sigmoid', 'tanh', 'relu'],
    'learning_rate': [0.001, 0.01, 0.1],
    'hidden_size': [8, 16, 32]
}

if __name__ == "__main__":
    results = pd.DataFrame(columns=['Parameters', 'Training Accuracy', 'Test Accuracy'])

    for activation in param_grid['activation']:
        for learning_rate in param_grid['learning_rate']:
            for hidden_size in param_grid['hidden_size']:
                params = {
                    'activation': activation,
                    'learning_rate': learning_rate,
                    'hidden_layer_size': hidden_size
                }
                print(params)

                nn = NeuralNetwork(data=data, input_size=len(columns), hidden_size=hidden_size, output_size=1,
                                   activation=activation, learning_rate=learning_rate)
                x, y = nn.preprocess_train(columns, 'age_group')
                nn.split_data(x, y, test_size=0.2, random_state=42)

                nn.train(epochs=10000)

                training_accuracy = nn.accuracy(nn.x_train, nn.y_train)
                test_accuracy = nn.accuracy(nn.x_test, nn.y_test)

                param_results = {
                    'Parameters': params,
                    'Training Accuracy': training_accuracy,
                    'Test Accuracy': test_accuracy
                }
                print(param_results)
                results = pd.concat([results, pd.DataFrame([param_results])], ignore_index=True)

    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv('results.csv')