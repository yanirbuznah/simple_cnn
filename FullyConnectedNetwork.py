from typing import List

import config
from common import ActivationFunction

import numpy as np

if config.USE_GPU:
    import cupy as np

class NeuralLayer(object):
    def __init__(self, size: int, index: int,with_bias):
        self.index = index
        self.bias = with_bias
        self.size = int(size)
        if with_bias:
            self.size += 1
        #self.clear_feeded_values()

    def feed(self, values: np.array):
        self.feeded_values += values
        # make sure that the bias still shut -1
        if self.bias:
            self.feeded_values[-1] = -1

    def clear_feeded_values(self):
        self.feeded_values = np.zeros(self.size)
        # update the bias neuron to -1
        if self.bias:
            self.feeded_values[-1] = -1

    def __repr__(self):
        return self.feeded_values.__repr__()


class NeuralNetwork(object):
    def __init__(self, input_layer_size: int, hidden_layers_sizes: List[int], output_layer_size: int, activation_function:ActivationFunction, learning_rate=0.001, randrange=0.01):
        self.input_layer = NeuralLayer(input_layer_size, 0, with_bias=True)
        self.hidden_layers = [NeuralLayer(size, index + 1, with_bias=True) for index, size in enumerate(hidden_layers_sizes)]
        self.output_layer = NeuralLayer(output_layer_size, 1 + len(hidden_layers_sizes), with_bias=False)
        self.randrange = randrange

        self.weights = [np.random.uniform(-randrange, randrange, (y.size, x.size)) for x, y in zip(self.layers[1:], self.layers[:-1])]

        self.activation_function = activation_function
        self.lr = learning_rate

    @property
    def layers(self):
        return [self.input_layer] + self.hidden_layers + [self.output_layer]

    def _clear_feeded_values(self):
        for layer in self.layers:
            layer.clear_feeded_values()

    @staticmethod
    def softmax(x):
        return np.exp(x)/sum(np.exp(x))

    def train_sample(self, input_values: np.array, correct_output: np.array):
        errors = []
        self._clear_feeded_values()
        self.feed_forward(input_values)
        current_errors = self.calculate_errors(correct_output)
        if not errors:
            errors = current_errors
        else:
            errors = [sum(l) for l in zip(errors, current_errors)]

        self.update_weights(errors)

        return errors[0]


    def feed_forward(self, input_values: np.array):
        self._clear_feeded_values()

        # Append one zero for bias neuron
        input_values = np.append(input_values, [0])
        self.input_layer.feed(input_values)
        for layer in self.hidden_layers + [self.output_layer]:
            prev_layer_index = layer.index - 1
            if config.SOFTMAX and layer == self.output_layer:
                values = self.softmax(np.dot(self.layers[prev_layer_index].feeded_values, self.weights[prev_layer_index]))

            else:
                values = self.activation_function.f(np.dot(self.layers[prev_layer_index].feeded_values, self.weights[prev_layer_index]))
            layer.feed(values)

    def set_weights(self, weights):
        self.weights = weights

    def calculate_errors(self, correct_output: np.array):
        errors = []
        prev_layer_error = correct_output - self.output_layer.feeded_values
        errors.insert(0, prev_layer_error)
        for layer in self.layers[:-1][::-1]:
            prev_layer_error = errors[0]
            weighted_error = np.dot(prev_layer_error, self.weights[layer.index].T) * self.activation_function.d(layer.feeded_values)
            errors.insert(0, weighted_error)

        return errors

    def update_weights(self, errors: List[np.array]):
        for layer in self.layers[:-1][::-1]:
            self.weights[layer.index] = self.weights[layer.index] + self.lr * np.outer(self.activation_function.f(layer.feeded_values), errors[layer.index + 1])

    def __str__(self):
        return f"Net[layers={','.join([str(layer.size) for layer in self.layers])}_randrange={self.randrange}]"
