import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def to_one_hot(y):
    n_col = np.amax(y) + 1
    binarized = np.zeros((len(y), n_col))
    for i in range(len(y)):
        binarized[i, y[i]] = 1.
    return binarized


# Convert one-hot encoding to array
def from_one_hot(y):
    arr = np.zeros((len(y), 1))

    for i in range(len(y)):
        l = layer2[i]
        for j in range(len(l)):
            if l[j] == 1:
                arr[i] = j + 1
    return arr


# sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Normalize array
def normalize(x, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
    l2[l2 == 0] = 1
    return x / np.expand_dims(l2, axis)

dataset = pd.read_csv("NHANES_age_prediction.csv")
dataset.drop(["RIAGENDR"], axis=1)

classes = ['Adult', 'Senior']
for i in range(len(classes)):
    dataset["age_group"].replace(classes[i], i, inplace=True)

columns = ["RIDAGEYR", "BMXBMI", "LBXGLU", "DIQ010", "LBXGLT", "LBXIN"]

x = pd.DataFrame(dataset, columns=columns)
x = normalize(x.values)

# Get Output, flatten and encode to one-hot
columns = ['age_group']
y = pd.DataFrame(dataset, columns=columns)
y = y.values
y = y.flatten()
print(y)
y = to_one_hot(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Weights
w0 = 2 * np.random.random((6, 4)) - 1  # for input   - 4 inputs, 3 outputs
w1 = 2 * np.random.random((4, 2)) - 1  # for layer 1 - 5 inputs, 3 outputs
# learning rate
n = 0.001

# Errors - for graph later
errors = []

# Train
for i in range(1000):
    # Feed forward
    layer0 = X_train
    layer1 = sigmoid(np.dot(layer0, w0))
    layer2 = sigmoid(np.dot(layer1, w1))

    # Back propagation using gradient descent
    layer2_error = y_train - layer2
    layer2_delta = layer2_error * sigmoid_deriv(layer2)

    layer1_error = layer2_delta.dot(w1.T)
    layer1_delta = layer1_error * sigmoid_deriv(layer1)

    w1 += layer1.T.dot(layer2_delta) * n
    w0 += layer0.T.dot(layer1_delta) * n

    error = np.mean(np.abs(layer2_error))
    print(error)
    errors.append(error)
    accuracy = (1 - error) * 100

print("Training Accuracy " + str(round(accuracy, 2)) + "%")

layer0 = X_test
layer1 = sigmoid(np.dot(layer0, w0))
layer2 = sigmoid(np.dot(layer1, w1))

layer2_error = y_test - layer2

error = np.mean(np.abs(layer2_error))
accuracy = (1 - error) * 100

print("Validation Accuracy " + str(round(accuracy, 2)) + "%")
