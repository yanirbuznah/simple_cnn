# This is a sample Python script.
from typing import List

import numpy as np

import FullyConnectedNetwork
import config
from ConvolutionLayer import ConvolutionLayer
from MaxPoolingLayer import MaxPoolingLayer


class CNN(object):
    def __init__(self, layers_shapes: List[int], learning_rate=0.001, randrange=0.01):
        self.randrange = randrange

        self.init_layers(layers_shapes)
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]

        output_size = self.output_layer.shape[0] * self.output_layer.shape[1] * self.output_layer.shape[2]
        self._fully_connected_net = FullyConnectedNetwork.NeuralNetwork(output_size, config.HIDDEN_LAYERS_SIZES, config.OUTPUT_LAYER_SIZE, config.ACTIVATION_FUNCTION)


    # self.weights = [np.random.uniform(-randrange, randrange, (y.size, x.size)) for x, y in zip(self.layers[1:], self.layers[:-1])]

        #        self.activation_function = activation_function
        self.lr = learning_rate

    def init_layers(self, layers_shapes):
        count, size, _ = layers_shapes[1]
        self.layers = []
        weights = np.random.uniform(self.randrange, -self.randrange, (layers_shapes[0][0], count, 3, 3))

        self.layers.append(ConvolutionLayer(layers_shapes[0], 0, False, next_weights=weights))
        self.layers.append(MaxPoolingLayer(layers_shapes[1], 1, False, prev_weights=weights))
        index = 2
        size //= 2
        while size > 4:
            weights = np.random.uniform(1, -1, (count, count * 2, 3, 3))
            self.layers.append(ConvolutionLayer((count, size, size), index, False, next_weights=weights))
            count *= 2
            index += 1
            self.layers.append(MaxPoolingLayer((count, size, size), index, False, prev_weights=weights))
            index += 1
            size //= 2


    def _feed_forward(self,input_values):
        values = input_values
        for index in range(0,len(self.layers), 2):
            result = self.layers[index].feed(values)
            #TODO: Check if true!
            if index != len(self.layers)-2:
                values = self.layers[index + 1].feed(result)

        flattened = result.flatten()
        self._fully_connected_net.feed_forward(flattened)

        x = 6

        # self.input_layer.feed(input_values)
        # for layer in self.layers