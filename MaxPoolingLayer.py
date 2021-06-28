# This is a sample Python script.

import numpy as np
import skimage.measure


class MaxPoolingLayer(object):
    def __init__(self, shape, index: int, with_bias, prev_weights):
        self.index = index
        self.bias = with_bias
        self.shape = shape
        self.prev_weights = prev_weights
        if with_bias:
            self.size += 1
        self.clear_feeded_values()

    def clear_feeded_values(self):
        self.feeded_values = np.zeros(self.shape)
        # update the bias neuron to -1
        if self.bias:
            self.feeded_values[-1] = -1

    def feed(self, values: np.array):
        self.feeded_values += values
        # make sure that the bias still shut -1
        if self.bias:
            self.feeded_values[-1] = -1

        return self.max_pooling()

    def max_pooling(self):
        result = np.zeros((self.feeded_values.shape[0], self.shape[1]//2, self.shape[2]//2))
        for i in range(self.feeded_values.shape[0]):
            result[i] += skimage.measure.block_reduce(self.feeded_values[i], (2, 2), self.abs_max)

        return result

    def abs_max(self, a, axis=None):
        amax = a.max(axis)
        amin = a.min(axis)
        return np.where(-amin > amax, amin, amax)
