import numpy as np
import matplotlib.pyplot as plt

import copy
import time


def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min=image_min, max=image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image


def plot_images(images, labels, classes, normalize=True):

    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(10, 10))

    for i in range(rows * cols):

        ax = fig.add_subplot(rows, cols, i + 1)

        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax.set_title(classes[labels[i]])
        ax.axis('off')

    plt.show()


def plot_train_images(data, classes, n_images=25):
    images, labels = zip(*[(image, label) for image, label in
                           [data[i] for i in range(n_images)]])

    # classes = test_data.classes  # not sure why the train data has no classes

    plot_images(images, labels, classes)
