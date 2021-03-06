from typing import List, Tuple

import numpy

import FullyConnectedNetwork
import config
from ConvolutionLayer import ConvolutionLayer
from MaxPoolingLayer import MaxPoolingLayer

import numpy as np


class CNN(object):
    def __init__(self, first_layers_shapes: Tuple, fully_connected_feature_map_dim, cnn_lr, fc_lr,
                 cnn_randrange=0.05, fully_connected_randrange=0.035):
        self.randrange = cnn_randrange

        self.init_layers(first_layers_shapes, fully_connected_feature_map_dim)
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]

        output_size = numpy.prod(self.output_layer.output_shape)
        self._fully_connected_net = FullyConnectedNetwork.NeuralNetwork(int(output_size), config.HIDDEN_LAYERS_SIZES, config.OUTPUT_LAYER_SIZE, config.ACTIVATION_FUNCTION, config.DROP_OUT,fc_lr, fully_connected_randrange)
        self._lr = cnn_lr

    def init_layers(self, layers_shapes, fully_connected_feature_map_dim):
        count, size, _ = layers_shapes[1]
        self.layers = []
        weights = np.random.uniform(self.randrange, -self.randrange, (layers_shapes[0][0], count, 3, 3))
        self.layers.append(ConvolutionLayer(layers_shapes[0], 0, False, next_weights=weights))
        self.layers.append(MaxPoolingLayer(layers_shapes[1], 1, False, prev_weights=weights))
        index = 2
        size //= 2
        while size > fully_connected_feature_map_dim:
            weights = np.random.uniform(self.randrange, -self.randrange, (count, count * 2, 3, 3))
            self.layers.append(ConvolutionLayer((count, size, size), index, False, next_weights=weights))
            count *= 2
            index += 1
            self.layers.append(MaxPoolingLayer((count, size, size), index, False, prev_weights=weights))
            index += 1
            size //= 2

    def _clear_feeded_values(self):
        for layer in self.layers:
            layer.clear_feeded_values()

    def train_sample(self, input_values: np.array, correct_output: np.array):
        errors = []
        self._clear_feeded_values()
        flattened_out = self._feed_forward(input_values)

        fully_connected_error = self._fully_connected_net.train_sample(flattened_out, correct_output)
        fully_connected_input_no_bias = fully_connected_error[:-1]
        fully_connected_error = fully_connected_input_no_bias.reshape(self.layers[-1].output_shape)

        errors += self._calculate_errors(fully_connected_error)
        self._update_weights(errors)

        prediction = np.argmax(self._fully_connected_net.output_layer.feeded_values)
        correct = np.argmax(correct_output)
        return correct == prediction

    def classify_sample(self, input_values: np.array):
        self._clear_feeded_values()
        flattened_out = self._feed_forward(input_values)
        self._fully_connected_net.feed_forward(flattened_out)
        prediction = np.argmax(self._fully_connected_net.output_layer.feeded_values)
        return prediction

    def validate_sample(self, input_values: np.array, correct_output: np.array):
        prediction = self.classify_sample(input_values)
        correct = np.argmax(correct_output)
        #print(prediction, correct, f"Certainty: {self.output_layer.feeded_values[prediction]}")
        return correct == prediction, self._fully_connected_net.output_layer.feeded_values[prediction]

    def _feed_forward(self,input_values):
        values = input_values
        for index in range(0,len(self.layers), 2):
            result = self.layers[index].feed(values)
            # #TODO: Check if true!
            # TODO: Probably not true. Last layer must be MaxPooling
            # if index != len(self.layers)-2:
            #     values = self.layers[index + 1].feed(result)
            values = self.layers[index + 1].feed(result)

        flattened = values.flatten()
        return flattened

    def _calculate_errors(self, prev_layer_error: np.array):
        errors = []
        for layer in self.layers[::-1]:
            errors.insert(0, layer.calculate_errors(prev_layer_error))
            prev_layer_error = errors[0]

        return errors

    def _update_weights(self, errors):
        for layer in self.layers[:-1][::-1]:
            if type(layer) == MaxPoolingLayer:
                continue
            layer.update_weights(errors[layer.index + 1], self._lr)

    @property
    def weights(self):
        cnn_weights = []
        for layer in self.layers:
            if type(layer) == ConvolutionLayer:
                cnn_weights.append(layer.next_weights)
            elif type(layer) == MaxPoolingLayer:
                cnn_weights.append(layer.prev_weights)
            else:
                raise

        return cnn_weights, self._fully_connected_net.weights

    def set_weights(self, cnn_weights, fc_weights):
        self._fully_connected_net.set_weights(fc_weights)
        for layer in self.layers:
            if type(layer) == ConvolutionLayer:
                layer.next_weights = cnn_weights[layer.index]
            elif type(layer) == MaxPoolingLayer:
                layer.prev_weights = cnn_weights[layer.index]
            else:
                raise

    @property
    def lr(self):
        return self._lr, self._fully_connected_net.lr

    def set_lr(self, cnn_lr, fc_lr):
        if cnn_lr > 0:
            self._lr = cnn_lr
        if fc_lr > 0:
            self._fully_connected_net.lr = fc_lr
