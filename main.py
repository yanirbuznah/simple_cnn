# This is a sample Python script.
import sys
from datetime import datetime
from typing import Tuple, List

import cv2
import numpy as np
import pandas as pd
from scipy import signal

from CNN import CNN


# TODO: BEFORE STARTING CONVOLUTION: sample[i] = np.pad(channel, (1, 1), 'constant')


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
    data = np.concatenate((train_data, validate_data))
    correct = np.concatenate((train_correct, validate_correct))
    rand_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rand_state)
    np.random.shuffle(correct)
    train_data, validate_data = np.split(data, [8000])
    train_correct, validate_correct = np.split(correct, [8000])
    return train_data, train_correct, validate_data, validate_correct


def processImage(image):
    image = cv2.imread(image)
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    return image


def conv2d(a, f):
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape=s, strides=a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)


def main():
    if len(sys.argv) < 3:
        print("Not enough arguments")
        return
    train_csv = sys.argv[1]
    validate_csv = sys.argv[2]
    test_csv = sys.argv[3] if len(sys.argv) >= 4 else None
    # validate_data, validate_correct = csv_to_data(validate_csv)
    train_data, train_correct = csv_to_data(train_csv)
    kernels = [np.random.uniform(1, -1, (3, 3)) for i in range(3 * 12)]
    kernels = np.array(kernels).reshape((3, 12, 3, 3))
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    shape = ((3, 32, 32), (16, 32, 32))
    net = CNN(shape)
    net._feed_forward(train_data[0])

    output = net.layers[0].feed(train_data[0])
    net.layers[1].feed(output)
    x = net.layers[1].max_pooling()

    x = datetime.now()
    for i in train_data:
        grad = np.array((signal.convolve2d(i[0], kernel, boundary='fill', mode='same'),
                         signal.convolve2d(i[0], kernel, boundary='fill', mode='same'),
                         signal.convolve2d(i[0], kernel, boundary='fill', mode='same')))
    x = datetime.now() - x
    print(f"x time = {x}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
