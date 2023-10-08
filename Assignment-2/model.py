import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (relu(x) > 0).astype(float)


class NeuralNetwork:
    def __init__(self, data, input_size, hidden_size, output_size, activation='sigmoid', learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.data = data
        self.in_col = []

        self.scaler = StandardScaler()

        self.weights_input_hidden = 2 * np.random.rand(input_size, hidden_size) - 1
        self.bias_hidden = 2 * np.random.rand(1, hidden_size) - 1
        self.weights_hidden_output = 2 * np.random.rand(hidden_size, output_size) - 1
        self.bias_output = 2 * np.random.rand(1, output_size) - 1

        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        else:
            raise NotImplementedError('only tanh, relu and sigmoid activation functions are allowed')

        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.hidden_layer_input = None
        self.hidden_layer_output = None
        self.output_layer_input = None
        self.output_layer_output = None

    def preprocess_train(self, in_col, output_col):
        self.data.drop_duplicates(inplace=True)

        self.in_col = in_col
        x = pd.DataFrame(self.data, columns=in_col)
        y = self.data[output_col]

        x = self.scaler.fit_transform(x)
        return x, np.array(y)

    def preprocess_test(self, x):
        x = pd.DataFrame(x, columns=self.in_col)
        return self.scaler.transform(x)

    def split_data(self, x, y, test_size, random_state=None):
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x, y, test_size=test_size, random_state=random_state)

    def forward(self, x):
        self.hidden_layer_output = self.activation(x.dot(self.weights_input_hidden) + self.bias_hidden)
        self.output_layer_output = sigmoid(
            self.hidden_layer_output.dot(self.weights_hidden_output) + self.bias_output)

        return self.output_layer_output

    def backward(self, x, y, output):
        error = y - output

        delta_output = error * sigmoid_derivative(output)
        error_hidden = delta_output.dot(self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.activation_derivative(self.hidden_layer_output)

        self.weights_hidden_output += self.hidden_layer_output.T.dot(delta_output) * self.learning_rate
        self.bias_output += np.sum(delta_output, axis=0, keepdims=True) * self.learning_rate

        self.weights_input_hidden += x.T.dot(delta_hidden) * self.learning_rate
        self.bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * self.learning_rate

    @staticmethod
    def loss(y, y_pred):
        return 0.5 * (y - y_pred) ** 2

    def train(self, epochs):
        if self.x_train is None:
            raise NotImplementedError("Kindly split the data to train and test!!!")

        loss, train_losses, test_losses = [], [], []

        more_count = 0

        for epoch in range(epochs):
            t = []
            for i in range(len(self.x_train)):
                input_data = self.x_train[i:i + 1]
                target = self.y_train[i:i + 1]
                output = self.forward(input_data)

                loss.append(self.loss(target, output))
                t.append(self.loss(target, output))
                self.backward(input_data, target, output)

            train_loss = np.mean(t) # np.mean(self.loss(self.y_train, self.transform(self.x_train)))
            test_loss = np.mean(self.loss(self.y_test, self.transform(self.x_test)))

            # print(self.y_test,
            #       np.squeeze(self.transform(self.x_test), axis=1),
            #       np.squeeze(self.get_preds(self.x_test), axis=1))

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            print(f"EPOCH {epoch} - TRAIN LOSS: {train_loss} - TEST LOSS: {test_loss}")

            if train_loss > train_losses[epoch - 1]:
                more_count += 1
                if more_count > 25:
                    break
            else:
                more_count = 0
            
            if epoch > 0 and np.abs(train_loss - train_losses[epoch - 1]) < 0.000001:
                break

    def transform(self, x):
        x = self.preprocess_test(x)
        return self.forward(x)

    def get_preds(self, x):
        return (self.transform(x) >= 0.5).astype(int)

    def accuracy(self, x, y):
        predictions = self.get_preds(x)
        return np.mean(predictions == y) * 100.
