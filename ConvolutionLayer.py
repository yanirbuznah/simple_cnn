from numba import prange
from scipy import ndimage

import config
from common import ActivationFunction, timeit

import numpy as np
from numba import njit


class ConvolutionLayer(object):
    def __init__(self, shape, index: int, with_bias, next_weights):
        self.index = index
        self.bias = with_bias
        self.input_shape = shape
        self.output_shape = (next_weights.shape[1], shape[1], shape[2])
        self.next_weights = next_weights
        if with_bias:
            self.size += 1
        self.clear_feeded_values()

    def feed(self, values: np.array):
        self.feeded_values += values
        # make sure that the bias still shut -1
        if self.bias:
            self.feeded_values[-1] = -1

        result = np.zeros(self.output_shape)
        for i in range(self.output_shape[0]):
            for j in range(self.input_shape[0]):
                result[i] += self._do_convolution(self.feeded_values[j], self.next_weights[j][i])

        return ActivationFunction.ReLU.f(result)

    def clear_feeded_values(self):
        self.feeded_values = np.zeros(self.input_shape)
        # update the bias neuron to -1
        if self.bias:
            self.feeded_values[-1] = -1


    def calculate_errors(self, prev_layer_error: np.array):
        result = np.zeros(self.input_shape)
        for i in range(self.output_shape[0]):
            for j in range(self.input_shape[0]):
                result[j] += ActivationFunction.ReLU.d(self.feeded_values[j]) * self._do_convolution(prev_layer_error[i], self.next_weights[j][i], forward=False)

        return result

    @staticmethod
    @njit(parallel=True)
    def _apply_weight_delta(padded_errors, prev_error, lr, weights, feeded_values):
        for i in prange(weights.shape[1]):
            x = padded_errors[i]
            for j in prange(weights.shape[0]):
                deltas = np.zeros((3, 3))
                w = weights[j][i]
                values = np.maximum(0, feeded_values[j]) # ReLU
                #values = self.feeded_values[j]
                deltas[0][0] += np.sum(values * x[:-2, :-2])  # bottom right
                deltas[0][1] += np.sum(values * x[:-2, 1:-1])  # bottom
                deltas[0][2] += np.sum(values * x[:-2, 2:])  # bottom left

                deltas[1][0] += np.sum(values * x[1:-1, :-2])  # right
                deltas[1][1] += np.sum(values * prev_error[i])  # center
                deltas[1][2] += np.sum(values * x[1:-1, 2:])  # left

                deltas[2][0] += np.sum(values * x[2:, :-2])  # top right
                deltas[2][1] += np.sum(values * x[2:, 1:-1])  # top
                deltas[2][2] += np.sum(values * x[2:, 2:])

                deltas *= lr
                w += deltas

    @staticmethod
    @njit
    def _generate_padded_errors(padded_errors, prev_error):
        count, orig_w, orig_h = padded_errors.shape
        for i in prange(count):
            padded_errors[i, 1:orig_w - 1, 1:orig_h - 1] = prev_error[i]

    @timeit
    def update_weights(self, prev_error, lr):
        padded_errors = np.zeros((prev_error.shape[0], prev_error.shape[1] + 2, prev_error.shape[2] + 2))
        self._generate_padded_errors(padded_errors, prev_error)
        self._apply_weight_delta(np.array(padded_errors), prev_error, lr, self.next_weights, self.feeded_values)

    def _rotate_180(self, mat: np.array):
        ret = np.copy(mat)
        ret[0][0], ret[2][2] = ret[2][2], ret[0][0]
        ret[0][1], ret[2][1] = ret[2][1], ret[0][1]
        ret[2][0], ret[0][2] = ret[0][2], ret[2][0]
        ret[1][0], ret[1][2] = ret[1][2], ret[1][0]

        return ret

    def _do_convolution(self, input_values, kernel, forward=True):
        kernel = kernel if forward else self._rotate_180(kernel)
        return ndimage.convolve(input_values, kernel, mode="constant", cval=0.0)

    def __repr__(self):
        return self.feeded_values.__repr__()
