import skimage.measure as measure
from numba import njit, prange

import numpy as np

from common import timeit


class MaxPoolingLayer(object):
    def __init__(self, shape, index: int, with_bias, prev_weights):
        self.index = index
        self.bias = with_bias
        self.input_shape = shape
        self.output_shape = (shape[0], shape[1] // 2, shape[2] // 2)
        self.prev_weights = prev_weights
        if with_bias:
            self.size += 1
        self.clear_feeded_values()

    def clear_feeded_values(self):
        self.feeded_values = np.zeros(self.input_shape)
        self.outputed_values = np.zeros(self.output_shape)
        self.max_indexes = np.empty((self.input_shape[0], 2, self.output_shape[1], self.output_shape[2]), dtype=np.int16)
        # update the bias neuron to -1
        if self.bias:
            self.feeded_values[-1] = -1

    @timeit
    def feed(self, values: np.array):
        self.feeded_values += values
        # make sure that the bias still shut -1
        if self.bias:
            self.feeded_values[-1] = -1

        result = np.zeros(self.output_shape)
        self._do_max_pooling(result, self.feeded_values, self.max_indexes)
        self.outputed_values += result
        return self.outputed_values

    @staticmethod
    @njit(parallel=True)
    def _calculate_errors(result, prev_layer_error, max_indexes):
        C, H, W = prev_layer_error.shape
        for c in prange(C):
            for h in prange(H):
                for w in prange(W):
                    result[c][max_indexes[c, 0, h, w]][max_indexes[c, 1, h, w]] = prev_layer_error[c, h, w]

    @timeit
    def calculate_errors(self, prev_layer_error: np.array):
        result = np.zeros(self.input_shape)
        return result

    @staticmethod
    @njit
    def _do_max_pooling(result, feeded_values, max_indexes):
        C = feeded_values.shape[0]

        output_dim = feeded_values.shape[1] // 2

        for c in prange(C):
            for h in prange(output_dim):
                for w in prange(output_dim):
                    h_start = h * 2
                    h_end = h_start + 2
                    w_start = w * 2
                    w_end = w_start + 2

                    result[c, h, w] = np.max(feeded_values[c, h_start:h_end, w_start:w_end])

                    scalar_ind = np.argmax(feeded_values[c, h_start:h_end, w_start:w_end])
                    # ind is in (row_ind, col_ind) format

                    # unravel_index is not supported with numba, so we implement it inline
                    if scalar_ind == 0:
                        max_i, max_j = 0, 0
                    elif scalar_ind == 1:
                        max_i, max_j = 0, 1
                    elif scalar_ind == 2:
                        max_i, max_j = 1, 0
                    elif scalar_ind == 3:
                        max_i, max_j = 1, 1

                    # real index of maximum element in the local region
                    real_ind = (max_i + h_start, max_j + w_start)

                    # store this real index in two part
                    max_indexes[c, 0, h, w] = real_ind[0]
                    max_indexes[c, 1, h, w] = real_ind[1]
        return result

