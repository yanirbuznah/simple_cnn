import random

import numpy as np

from common import ActivationFunction, AdaptiveLearningRateMode

# Neural Network Configuration
SEED = random.randint(0, 100000000)
HIDDEN_LAYERS_SIZES = [1000,1000]
OUTPUT_LAYER_SIZE = 10
FC_FEATURE_MAP_DIM = 8
ACTIVATION_FUNCTION = ActivationFunction.ReLU
FC_RANDRANGE = -1
CNN_RANDRANGE = 1

CNN_LEARNING_RATE = 0.0003
FC_LEARNING_RATE = 0.0003

# Training Configuration
EPOCH_COUNT = 50
INPUT_LAYER_NOISE_PROB = 0
SUBSET_SIZE = -1
MINI_BATCH_SIZE = 1

CNN_ADAPTIVE_LEARNING_RATE_MODE = AdaptiveLearningRateMode.FORMULA
CNN_ADAPTIVE_LEARNING_RATE_FORMULA = lambda epoch: 0.005 * np.exp(-0.01 * epoch)
CNN_ADAPTIVE_LEARNING_RATE_DICT = {
    20: 0.002,
    40: 0.001,
    60: 0.0005,
    80: 0.0004,
    100: 0.0003
}



FC_ADAPTIVE_LEARNING_RATE_MODE = AdaptiveLearningRateMode.FORMULA
FC_ADAPTIVE_LEARNING_RATE_FORMULA = lambda epoch: 0.005 * np.exp(-0.01 * epoch)
FC_ADAPTIVE_LEARNING_RATE_DICT = {
    20: 0.002,
    40: 0.001,
    60: 0.0005,
    80: 0.0004,
    100: 0.0003
}
SHOULD_TRAIN = True

SAVED_MODEL_PICKLE_MODE = True  # Put False to use csv files, True to use pickle
TRAINED_NET_DIR = None #"ac443d74-364d-412e-b493-a5b9d8e289f6"  # Put None if you don't want to load a result dir

TAKE_BEST_FROM_TRAIN = False
TAKE_BEST_FROM_VALIDATE = False
SHOULD_SHUFFLE = False

SOFTMAX = True
DROP_OUT = [0, 0]
USE_GPU = True