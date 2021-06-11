# This is a sample Python script.
from typing import Tuple, List

import numpy as np
import pandas as pd
import sys
import cv2

# TODO: BEFORE STARTING CONVOLUTION: sample[i] = np.pad(channel, (1, 1), 'constant')


def perform_convolution(image, kernel):
    channels = []
    for channel in image:
        #channel = np.pad(channel, (1, 1), 'constant')
        kernel_n = kernel.shape[0]
        image_n = channel.shape[0]
        result_n = image_n - kernel_n + 1
        result = np.zeros((result_n, result_n))
        for i in range(result_n):
            for j in range(result_n):
                result[i][j] = np.sum(channel[i:i + kernel_n, j:j + kernel_n] * kernel)
                # TODO: BIAS?
        channels.append(result)
    return np.array(channels)


def csv_to_data(path, count=-1) -> Tuple[np.array, np.array]:
    df = pd.read_csv(path, header=None)
    output = df.loc[:, 0]
    data = df.drop(columns=0).to_numpy()
    results_indexes = output.to_numpy()
    results = result_classifications_to_np_layers(results_indexes)
    data = list(data.reshape(1000, 3, 32, 32))

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

def main():
    if len(sys.argv) < 3:
        print("Not enough arguments")
        return
    train_csv = sys.argv[1]
    validate_csv = sys.argv[2]
    test_csv = sys.argv[3] if len(sys.argv) >= 4 else None
    validate_data, validate_correct = csv_to_data(validate_csv)

    kernel = np.array([
        [1, 2, 1],
        [2, -11, 2],
        [1, 2, 1]
    ]).transpose()


    save_image(validate_data[0], "before.bmp")
    result = perform_convolution(validate_data[0], kernel)
    save_image(result, "after.bmp")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
