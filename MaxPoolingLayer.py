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
        self._do_max_pooling(result, self.feeded_values)
        self.outputed_values += result
        return self.outputed_values

    @staticmethod
    def split_matrix(array, nrows, ncols):
        """Split a matrix into sub-matrices."""

        r, h = array.shape
        return (array.reshape(h // nrows, nrows, -1, ncols)
                .swapaxes(1, 2)
                .reshape(-1, nrows, ncols))

    @staticmethod
    @njit(parallel=True)
    def _calculate_errors(result, prev_layer_error, feeded_values):
        for feature_map_index, feature_map in enumerate(feeded_values):
            for i in prange(prev_layer_error[feature_map_index].shape[0]):
                for j in prange(prev_layer_error[feature_map_index].shape[1]):
                    pool = feature_map[i * 2:i * 2 + 2, j * 2:j * 2 + 2]

                    # unravel_index is not supported with numba, so we implement it inline
                    max_unraveled = pool.argmax()
                    if max_unraveled == 0:
                        max_i, max_j = 0, 0
                    elif max_unraveled == 1:
                        max_i, max_j = 0, 1
                    elif max_unraveled == 2:
                        max_i, max_j = 1, 0
                    elif max_unraveled == 3:
                        max_i, max_j = 1, 1

                    result[feature_map_index][i * 2 + max_i][j * 2 + max_j] += prev_layer_error[feature_map_index][i][j]

    @timeit
    def calculate_errors(self, prev_layer_error: np.array):
        result = np.zeros(self.input_shape)
        self._calculate_errors(result, prev_layer_error, self.feeded_values)
        return result

    def _do_max_pooling_a(self):
        result = np.zeros(self.output_shape)
        for i in range(self.feeded_values.shape[0]):
            result[i] += measure.block_reduce(self.feeded_values[i], (2, 2), np.max)

        return result

    @staticmethod
    @njit
    def _do_max_pooling(result, feeded_values):
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
                    #self.max_index[n, c, 0, h, w] = real_ind[0]
                    #self.max_index[n, c, 1, h, w] = real_ind[1]
        return result

