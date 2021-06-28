# This is a sample Python script.

import numpy as np
from scipy import ndimage

from common import ActivationFunction


class ConvolutionLayer(object):
    def __init__(self, shape, index: int, with_bias, next_weights):
        self.index = index
        self.bias = with_bias
        self.shape = shape
        self.next_weights = next_weights
        if with_bias:
            self.size += 1
        self.clear_feeded_values()

    def feed(self, values: np.array):
        self.feeded_values += values
        # make sure that the bias still shut -1
        if self.bias:
            self.feeded_values[-1] = -1

        return self.convu()

    def clear_feeded_values(self):
        self.feeded_values = np.zeros(self.shape)
        # update the bias neuron to -1
        if self.bias:
            self.feeded_values[-1] = -1

    def convu(self):
        result = np.zeros((self.next_weights.shape[1], self.shape[1], self.shape[2]))
        for i in range(self.next_weights.shape[1]):
            for j in range(self.shape[0]):
                result[i] += ndimage.convolve(self.feeded_values[j], self.next_weights[j][i], mode="constant", cval=0.0)

        return ActivationFunction.ReLU.f(result)

    def __repr__(self):
        return self.feeded_values.__repr__()
