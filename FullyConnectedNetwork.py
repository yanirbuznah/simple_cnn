from typing import List

from config import *
from common import ActivationFunction, timeit

import numpy as np

if USE_GPU:
    import cupy as np

from NeuralLayer import NeuralLayer


class NeuralNetwork(object):
    def __init__(self, input_layer_size: int, hidden_layers_sizes: List[int], output_layer_size: int, activation_function:ActivationFunction,hidden_layer_dropout:List[float], learning_rate=0.001, randrange=0.01):
        self.input_layer = NeuralLayer(input_layer_size, 0, with_bias=True,dropout=0)
        self.hidden_layers = [NeuralLayer(size, index + 1, with_bias=True,dropout=hidden_layer_dropout[index]) for index, size in enumerate(hidden_layers_sizes)]
        self.output_layer = NeuralLayer(output_layer_size, 1 + len(hidden_layers_sizes), with_bias=False,dropout=0)
        self._initial_weights(randrange)

        self.activation_function = activation_function
        self.lr = learning_rate

    @property
    def layers(self):
        return [self.input_layer] + self.hidden_layers + [self.output_layer]

    def _initial_weights(self,randrange):
        if randrange > 0:
            self._weights = [np.random.uniform(-randrange, randrange, (y.size, x.size)) for x, y in zip(self.layers[1:], self.layers[:-1])]
            self.randrange = randrange
        else:
            self.randrange = "xavier"
            self._weights = []
            for i in range(len(self.layers)-1):
                n = self.layers[i].size
                std = np.sqrt(2.0/n)
                numbers = np.random.randn(self.layers[i].size,self.layers[i+1].size)
                scaled = numbers * std
                self._weights.append(scaled)

    def _clear_feeded_values(self):
        for layer in self.layers:
            layer.clear_feeded_values()

    @staticmethod
    def softmax(x):
        return np.exp(x)/sum(np.exp(x))

    @staticmethod
    def softmax_d(x):
        # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
        s = x.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    def train_sample(self, input_values: np.array, correct_output: np.array):
        if USE_GPU:
            correct_output = np.array(correct_output)

        errors = []
        self._clear_feeded_values()
        self.feed_forward(input_values)
        current_errors = self._calculate_errors(correct_output)
        if not errors:
            errors = current_errors
        else:
            errors = [sum(l) for l in zip(errors, current_errors)]

        self._update_weights(errors)

        ret = errors[0]

        if USE_GPU:
            ret = np.asnumpy(ret)

        return ret

    @timeit
    def feed_forward(self, input_values: np.array):
        if USE_GPU:
            input_values = np.array(input_values)

        self._clear_feeded_values()

        # Append one zero for bias neuron
        input_values = np.append(input_values, [0])
        self.input_layer.feed(input_values)
        for layer in self.hidden_layers + [self.output_layer]:
            prev_layer_index = layer.index - 1
            if SOFTMAX and layer == self.output_layer:
                f = self.softmax
            else:
                f = self.activation_function.f

            values = f(np.dot(self.layers[prev_layer_index].feeded_values, self._weights[prev_layer_index]))
            layer.feed(values)

    @property
    def weights(self):
        weights = self._weights
        if USE_GPU:
            weights = [np.asnumpy(w) for w in weights]

        return weights

    def set_weights(self, weights):
        if USE_GPU:
            weights = [np.array(w) for w in weights]

        self._weights = weights

    @timeit
    def _calculate_errors(self, correct_output: np.array):
        errors = []
        prev_layer_error = correct_output - self.output_layer.feeded_values
        errors.insert(0, prev_layer_error)
        for layer in self.layers[:-1][::-1]:
            prev_layer_error = errors[0]
            weighted_error = np.dot(prev_layer_error, self._weights[layer.index].T) * self.activation_function.d(layer.feeded_values)
            errors.insert(0, weighted_error)

        return errors

    @timeit
    def _update_weights(self, errors: List[np.array]):
        for layer in self.layers[:-1][::-1]:
            self._weights[layer.index] = self._weights[layer.index] + self.lr * np.outer(self.activation_function.f(layer.feeded_values), errors[layer.index + 1])

    def __str__(self):
        return f"Net[layers={','.join([str(layer.size) for layer in self.layers])}_randrange={self.randrange}]"
