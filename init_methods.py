import numpy as np


def initialize_parameters_zeros(input_dims, hidden_dims, output_dims):
    hidden_weights = np.zeros((input_dims, hidden_dims))
    hidden_bias = np.zeros((1, hidden_dims))
    output_weights = np.zeros((hidden_dims, output_dims))
    output_bias = np.zeros((1, output_dims))

    return [hidden_weights, hidden_bias, output_weights, output_bias]


def initialize_parameters_random(input_dims, hidden_dims, output_dims):
    hidden_weights = np.random.uniform(size=(input_dims, hidden_dims))
    hidden_bias = np.random.uniform(size=(1, hidden_dims))
    output_weights = np.random.uniform(size=(hidden_dims, output_dims))
    output_bias = np.random.uniform(size=(1, output_dims))

    return [hidden_weights, hidden_bias, output_weights, output_bias]


def initialize_parameters_xavier(input_dims, hidden_dims, output_dims):
    hidden_weights = np.random.uniform(size=(input_dims, hidden_dims)) * np.sqrt(1 / input_dims)
    hidden_bias = np.zeros((1, hidden_dims))
    output_weights = np.random.uniform(size=(hidden_dims, output_dims)) * np.sqrt(1 / hidden_dims)
    output_bias = np.zeros((1, output_dims))

    return [hidden_weights, hidden_bias, output_weights, output_bias]


def initialize_parameters_he(input_dims, hidden_dims, output_dims):
    hidden_weights = np.random.uniform(size=(input_dims, hidden_dims)) * np.sqrt(2 / input_dims)
    hidden_bias = np.zeros((1, hidden_dims))
    output_weights = np.random.uniform(size=(hidden_dims, output_dims)) * np.sqrt(2 / hidden_dims)
    output_bias = np.zeros((1, output_dims))

    return [hidden_weights, hidden_bias, output_weights, output_bias]
