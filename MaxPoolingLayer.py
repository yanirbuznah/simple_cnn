import skimage.measure as measure

import blockwise_view

import numpy
import numpy as np

import config

if config.USE_GPU:
    import cupy as np
    import cucim.skimage.measure as measure


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

    def feed(self, values: np.array):
        self.feeded_values += values
        # make sure that the bias still shut -1
        if self.bias:
            self.feeded_values[-1] = -1

        self.outputed_values += self._do_max_pooling()
        return self.outputed_values

    @staticmethod
    def split_matrix(array, nrows, ncols):
        """Split a matrix into sub-matrices."""

        r, h = array.shape
        return (array.reshape(h // nrows, nrows, -1, ncols)
                .swapaxes(1, 2)
                .reshape(-1, nrows, ncols))

    def _calc_error_mask(self, feeded, outputed):
        mask = np.equal(feeded, outputed.repeat(2, axis=0).repeat(2, axis=1)).astype(int)

        # if multiple neurons in the same group were the same value, choose one arbitrarily
        split_mask = blockwise_view.blockwise_view(mask, (2, 2), aslist=False)
        for i in range(split_mask.shape[0]):
            for j in range(split_mask.shape[1]):
                mask_item = split_mask[i][j]
                s = mask_item.sum()
                if s == 1:
                    continue

                # Choose a random neuron index in the mask item that will take the error
                index = np.random.choice(np.nonzero(mask_item.reshape(4))[0], size=1)[0]  # Must give size for cupy
                index = np.unravel_index(index, (2, 2))
                mask_item.fill(0)
                mask_item[index] = 1

        return mask

    def calculate_errors(self, prev_layer_error: np.array):
        result = np.zeros(self.input_shape)
        for i in range(self.feeded_values.shape[0]):
            # Find which neurons caused the errors
            mask = self._calc_error_mask(self.feeded_values[i], self.outputed_values[i])
            errors = prev_layer_error[i].repeat(2, axis=0).repeat(2, axis=1) * mask
            result[i] += errors

        return result

    def _do_max_pooling(self):
        result = np.zeros(self.output_shape)
        for i in range(self.feeded_values.shape[0]):
            result[i] += measure.block_reduce(self.feeded_values[i], (2, 2), self._abs_max)

        return result

    def _abs_max(self, a, axis=None):
        amax = a.max(axis)
        amin = a.min(axis)
        return np.where(-amin > amax, amin, amax)
