import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ConvolutionLayer
import config
from CNN import CNN
from main import csv_to_data, load_state

model_path = sys.argv[1]
csv_path = sys.argv[2]

shape = ((3, 32, 32), (16, 32, 32))
net = CNN(shape, config.FC_FEATURE_MAP_DIM, config.CNN_LEARNING_RATE, config.FC_LEARNING_RATE, config.CNN_RANDRANGE,
          config.FC_RANDRANGE)

load_state(Path(model_path), net)

train_images, _ = csv_to_data(csv_path, 10)

count = 0
plt.figure(figsize=(30,30))

grid_rows = len(train_images)
grid_cols = net.layers[1].feeded_values.shape[0] + 1 # Num of feature maps + 1 for the original image

for i in range(len(train_images)):
    bmp_data = train_images[i] * 255
    bmp_data = bmp_data.astype(np.uint8)

    image = bmp_data.reshape(3, 32, 32).transpose(1, 2, 0)

    plt.subplot(grid_rows, grid_cols, count + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True)
    plt.imshow(image)
    count += 1

    net.classify_sample(train_images[i])
    for l in [x for x in net.layers[1:] if type(x) == ConvolutionLayer.ConvolutionLayer]:
        for feature_map in l.feeded_values:
            bmp_data = feature_map * 255
            plt.xticks([])
            plt.yticks([])
            plt.grid(True)
            plt.subplot(grid_rows, grid_cols,count+1)
            plt.imshow(feature_map, cmap="gray")

            count += 1


print(count)
plt.show()