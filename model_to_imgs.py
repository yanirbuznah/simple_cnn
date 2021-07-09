import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ConvolutionLayer
import config
from CNN import CNN


def plot_from_existing_model(model_path, csv_path, count):
    from main import csv_to_data, load_state

    shape = ((3, 32, 32), (16, 32, 32))
    net = CNN(shape, config.FC_FEATURE_MAP_DIM, config.CNN_LEARNING_RATE, config.FC_LEARNING_RATE, config.CNN_RANDRANGE,
              config.FC_RANDRANGE)

    load_state(Path(model_path), net)

    train_images, _ = csv_to_data(csv_path, count)
    plot_feature_maps(net, train_images)


def plot_feature_maps(net, samples):
    count = 0
    plt.figure(figsize=(30,30))

    grid_rows = len(samples)
    grid_cols = net.layers[1].feeded_values.shape[0] + 1 # Num of feature maps + 1 for the original image

    print(f"Plotting {grid_rows * grid_cols} ({grid_rows}x{grid_cols}) images...")

    for i in range(len(samples)):
        bmp_data = samples[i] * 255
        bmp_data = bmp_data.astype(np.uint8)

        image = bmp_data.reshape(3, 32, 32).transpose(1, 2, 0)

        plt.subplot(grid_rows, grid_cols, count + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(True)
        plt.imshow(image)
        count += 1

        net.classify_sample(samples[i])
        for l in [x for x in net.layers[1:] if type(x) == ConvolutionLayer.ConvolutionLayer]:
            for feature_map in l.feeded_values:
                plt.xticks([])
                plt.yticks([])
                plt.grid(True)
                plt.subplot(grid_rows, grid_cols,count+1)
                plt.imshow(feature_map, cmap="gray")

                count += 1


    plt.show()


if __name__ == '__main__':
    model_path = sys.argv[1]
    csv_path = sys.argv[2]
    plot_from_existing_model(model_path, csv_path, 10)
