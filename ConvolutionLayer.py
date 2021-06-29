import numpy as np
from scipy import ndimage

from common import ActivationFunction


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
        return ActivationFunction.ReLU.d(self._do_convolution(prev_layer_error))

    def _do_convolution(self, input_values):
        result = np.zeros(self.output_shape)
        for i in range(self.output_shape[0]):
            for j in range(self.input_shape[0]):
                result[i] += ndimage.convolve(input_values[j], self.next_weights[j][i], mode="constant", cval=0.0)

        return result

    def __repr__(self):
        return self.feeded_values.__repr__()
