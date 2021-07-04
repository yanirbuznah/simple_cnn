from scipy import ndimage

import config
from common import ActivationFunction

import numpy as np

if config.USE_GPU:
    import cupy as np
    from cupyx.scipy import ndimage

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

        return ActivationFunction.ReLU.f(self._do_convolution(self.feeded_values))

    def clear_feeded_values(self):
        self.feeded_values = np.zeros(self.input_shape)
        # update the bias neuron to -1
        if self.bias:
            self.feeded_values[-1] = -1

    def calculate_errors(self, prev_layer_error: np.array):
        return ActivationFunction.ReLU.d(self._do_convolution(prev_layer_error, forward=False))

    def update_weights(self, error, lr):
        for i in range(self.output_shape[0]):
            for j in range(self.input_shape[0]):
                w = self.next_weights[j][i]
                w[0][0] += lr * np.sum(ActivationFunction.ReLU.d(self.feeded_values[j]) * np.pad(error[i],((1,0),(1,0)), mode='constant')[:-1, :-1])  # bottom right
                w[0][1] += lr * np.sum(ActivationFunction.ReLU.d(self.feeded_values[j]) * np.pad(error[i],((1,0),(0,0)), mode='constant')[:-1, :]) # bottom
                w[0][2] += lr * np.sum(ActivationFunction.ReLU.d(self.feeded_values[j]) * np.pad(error[i],((1,0),(0,1)), mode='constant')[:-1, 1:])   # bottom left

                w[1][0] += lr * np.sum(ActivationFunction.ReLU.d(self.feeded_values[j]) * np.pad(error[i],((0,0),(1,0)), mode='constant')[:, :-1]) # right
                w[1][1] += lr * np.sum(ActivationFunction.ReLU.d(self.feeded_values[j]) * error[i]) # center
                w[1][2] += lr * np.sum(ActivationFunction.ReLU.d(self.feeded_values[j]) * np.pad(error[i],((0,0),(0,1)), mode='constant')[:, 1:]) # left

                w[2][0] += lr * np.sum(ActivationFunction.ReLU.d(self.feeded_values[j]) * np.pad(error[i],((0,1),(1,0)), mode='constant')[1:, :-1])   # top right
                w[2][1] += lr * np.sum(ActivationFunction.ReLU.d(self.feeded_values[j]) * np.pad(error[i],((0,1),(0,0)), mode='constant')[1:, :]) # top
                w[2][2] += lr * np.sum(ActivationFunction.ReLU.d(self.feeded_values[j]) * np.pad(error[i],((0,1),(0,1)), mode='constant')[1:, 1:])   # top left

    def _rotate_180(self, mat: np.array):
        ret = np.copy(mat)
        ret[0][0], ret[2][2] = ret[2][2], ret[0][0]
        ret[0][1], ret[2][1] = ret[2][1], ret[0][1]
        ret[2][0], ret[0][2] = ret[0][2], ret[2][0]
        ret[1][0], ret[1][2] = ret[1][2], ret[1][0]

        return ret

    def _do_convolution(self, input_values, forward=True):
        result = np.zeros(self.output_shape)
        for i in range(self.output_shape[0]):
            for j in range(self.input_shape[0]):
                kernel = self.next_weights[j][i] if forward else self._rotate_180(self.next_weights[j][i])
                result[i] += ndimage.convolve(input_values[j], kernel, mode="constant", cval=0.0)

        return result

    def __repr__(self):
        return self.feeded_values.__repr__()
