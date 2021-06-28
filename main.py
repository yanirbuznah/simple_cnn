# This is a sample Python script.
from typing import Tuple, List

import numpy as np
import pandas as pd
import sys
import cv2

# TODO: BEFORE STARTING CONVOLUTION: sample[i] = np.pad(channel, (1, 1), 'constant')


class ConvolutionLayer(object):
    def __init__(self,shape, index: int,with_bias):
        self.index = index
        self.bias = with_bias
        self.shape = shape
        if with_bias:
            self.size += 1
        self.clear_feeded_values()

    def feed(self, values: np.array):
        self.feeded_values += values
        # make sure that the bias still shut -1
        if self.bias:
            self.feeded_values[-1] = -1

    def clear_feeded_values(self):
        self.feeded_values = np.zeros(self.shape)
        # update the bias neuron to -1
        if self.bias:
            self.feeded_values[-1] = -1


    def convu(self,data, kernels):
        values = np.zeros(self.shape)
        for i, feature_map in enumerate(data):
            for j, weight in enumerate(kernels[i]):
                values[j] += perform_convolution(feature_map, weight)

        return values


    def __repr__(self):
        return self.feeded_values.__repr__()



def perform_convolution(feature, kernel):
    feature = np.pad(feature, (1, 1), 'constant')
    kernel_n = kernel.shape[0]
    image_n = feature.shape[0]
    result_n = image_n - kernel_n + 1
    result = np.zeros((result_n, result_n))
    for i in range(result_n):
        for j in range(result_n):
            result[i][j] = np.sum(feature[i:i + kernel_n, j:j + kernel_n] * kernel)
            # TODO: BIAS?
    return result






def csv_to_data(path, count=-1) -> Tuple[np.array, np.array]:
    df = pd.read_csv(path, header=None)
    output = df.loc[:, 0]
    data = df.drop(columns=0).to_numpy()
    results_indexes = output.to_numpy()
    results = result_classifications_to_np_layers(results_indexes)
    data = list(data.reshape(data.shape[0], 3, 32, 32))

    if count == -1:
        return data, results
    else:
        return data[:count], results[:count]


def result_classifications_to_np_layers(results_classifications: List[int]) -> np.array:
    results = np.zeros((len(results_classifications), 10))
    for i in range(len(results_classifications)):
        if not str(results_classifications[i]).isdigit():
            # This is probably a test set. Ignore expected results column
            results = []
            break

        results[i][results_classifications[i] - 1] = 1

    return results

def save_image(data, path):
    data = data.transpose(1, 2, 0) * 255
    data = data.astype(np.float32)
    image_cv = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_cv)




def suffle(train_data, train_correct, validate_data, validate_correct):
    data = np.concatenate((train_data,validate_data))
    correct = np.concatenate((train_correct,validate_correct))
    rand_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rand_state)
    np.random.shuffle(correct)
    train_data, validate_data = np.split(data,[8000])
    train_correct, validate_correct = np.split(correct,[8000])
    return train_data,train_correct,validate_data,validate_correct

def main():
    if len(sys.argv) < 3:
        print("Not enough arguments")
        return
    train_csv = sys.argv[1]
    validate_csv = sys.argv[2]
    test_csv = sys.argv[3] if len(sys.argv) >= 4 else None
    validate_data, validate_correct = csv_to_data(validate_csv)
    #train_data, train_correct = csv_to_data(train_csv)
    #suffle(train_data,train_correct,validate_data,validate_correct)
    # x = np.split(train_data[0],1024)
    kernels = [np.random.uniform(1,-1,(3,3)) for i in range(3*12) ]
    kernels = np.array(kernels).reshape((3,12,3,3))
    layer = ConvolutionLayer((12,32,32),1,False)

    for data in validate_data:
        layer.feed(layer.convu(data,kernels))
    x = 5

    convu(validate_data[0],kernels)
    for i in train_data:
        perform_convolution(i,)


    save_image(validate_data[0], "before.bmp")
    result = perform_convolution(validate_data[0], kernel)
    save_image(result, "after.bmp")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
