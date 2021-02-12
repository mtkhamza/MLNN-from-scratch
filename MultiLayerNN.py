import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import init_methods


# Activation functions

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def sigmoid_derivation(x):
    return (x * (1 - x))


def relu(x):
    result = np.maximum(0, x)
    return result


def relu_derivation(x):
    result = x
    result[result <= 0] = 0
    result[result > 0] = 1
    return result


def tanh(z):
    return (np.tanh(z))


def tanh_derivation(z):
    return (1 - np.tanh(z) ** 2)


# my class Multi Layer Neural Network

class MultiLayerNN:

    def __init__(self, input_dims, hidden_dims, output_dims, init_method, activation, learning_rate, num_epochs):

        self.__input_dims = input_dims
        self.__hidden_dims = hidden_dims
        self.__output_dims = output_dims
        self.__init_method = init_method
        self.__learning_rate = learning_rate
        self.__num_epochs = num_epochs

        # Determine if the model has to deal a binary classification or not
        if self.__output_dims == 1:
            self.__binary_class = True
            self.__activation = activation
        else:
            self.__binary_class = False
            self.__activation = activation

        self.__weights = None
        self.__bias = None

    def __init_weights(self):
        if self.__init_method == 'xavier':
            initial_parameters = init_methods.initialize_parameters_xavier(self.__input_dims, self.__hidden_dims,
                                                                           self.__output_dims)
        elif self.__init_method == 'he':
            initial_parameters = init_methods.initialize_parameters_he(self.__input_dims, self.__hidden_dims,
                                                                       self.__output_dims)
        elif self.__init_method == 'random':
            initial_parameters = init_methods.initialize_parameters_random(self.__input_dims, self.__hidden_dims,
                                                                           self.__output_dims)
        elif self.__init_method == 'zeros':
            initial_parameters = init_methods.initialize_parameters_zeros(self.__input_dims, self.__hidden_dims,
                                                                          self.__output_dims)

        self.__hidden_weights, self.__hidden_bias, self.__output_weights, self.__output_bias = initial_parameters

    def __forward_propagation(self, X):
        function = {
            'sigmoid': sigmoid,
            'relu': relu,
            'tanh': tanh,
        }
        activation = function[self.__activation]
        hidden_layer_activation = np.dot(X, self.__hidden_weights)
        hidden_layer_activation += self.__hidden_bias
        hidden_layer_output = activation(hidden_layer_activation)
        activation = function['sigmoid']  # I used sigmoid by default
        output_layer_activation = np.dot(hidden_layer_output, self.__output_weights)
        output_layer_activation += self.__output_bias
        predicted_output = activation(output_layer_activation)
        return predicted_output, hidden_layer_output

    def __back_propagation(self, predicted_output, expected_output, hidden_layer_output):
        function = {
            'sigmoid': sigmoid_derivation,
            'relu': relu_derivation,
            'tanh': tanh_derivation,
        }
        activation_p = function[self.__activation]
        error = expected_output - predicted_output
        d_predicted_output = error * activation_p(predicted_output)
        activation_p = function['sigmoid']  # I used sigmoid by default
        error_hidden_layer = d_predicted_output.dot(self.__output_weights.T)
        d_hidden_layer = error_hidden_layer * activation_p(hidden_layer_output)
        return d_predicted_output, d_hidden_layer

    def __update_weights(self, d_predicted_output, d_hidden_layer, hidden_layer_output, X):
        self.__output_weights += self.__learning_rate * hidden_layer_output.T.dot(d_predicted_output)
        self.__output_bias += self.__learning_rate * np.sum(d_predicted_output, axis=0, keepdims=True)
        self.__hidden_weights += self.__learning_rate * X.T.dot(d_hidden_layer)
        self.__hidden_bias += self.__learning_rate * np.sum(d_hidden_layer, axis=0, keepdims=True)

    def fit(self, X, y):
        y = y.reshape(y.shape[0], 1)
        self.__init_weights()

        for i in range(self.__num_epochs):
            # Forward Propagation
            predicted_output, hidden_layer_output = self.__forward_propagation(X)
            # Back propagation
            d_predicted_output, d_hidden_layer = self.__back_propagation(predicted_output, y, hidden_layer_output)
            # Updating Weights and Biases
            self.__update_weights(d_predicted_output, d_hidden_layer, hidden_layer_output, X)
            print("iteration : {} | Loss => {:.4f} ".format(i, np.mean(predicted_output - y)))

    def predict(self, X, threshold=0.5):
        AL, _ = self.__forward_propagation(X)
        if not self.__binary_class:
            return np.atleast_2d(AL.argmax(axis=0))
        predictions = (AL >= threshold)
        predictions = predictions.reshape(1, predictions.shape[0])[0]
        return (predictions).astype(int)


if __name__ == '__main__':
    rng = np.random.RandomState(0)
    X = rng.randn(300, 2)
    # XOR problem
    y = np.array(np.logical_xor(X[:, 0] > 0, X[:, 1] > 0), dtype=int)

    epochs = 10000
    learning_rate = 0.001
    input_layer = X.shape[1]
    hidden_layer = 10
    output_layer = 1
    init_function = "random"
    activation_function = "tanh"

    model = MultiLayerNN(input_layer, hidden_layer, output_layer, init_function,
                         activation_function, learning_rate, epochs)

    model.fit(X, y)

    fig = plt.figure(figsize=(10, 8))
    fig = plot_decision_regions(X=X, y=y, clf=model, legend=2)
    plt.title("MLNN PLOT")
    plt.show()
