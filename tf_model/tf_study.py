import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from data_utils import load_datasets
from visualize_utils import plot_images


def linear_function():
    tf.random.set_seed(1)
    X = tf.random.normal([3, 1])
    W = tf.random.normal([4, 3])
    b = tf.random.normal([4, 1])
    Y = W @ X + b
    Y2 = tf.matmul(W, X) + b

    print(tf.reduce_all(Y == Y2))

    return Y


def sigmoid(z):
    z = tf.cast(z, tf.float32)
    s = tf.keras.activations.sigmoid(z)
    return s


def normalize(image):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [-1, ])
    return image


def investigate_data():
    train_x, train_y, test_x, test_y = load_datasets()
    x_train = tf.data.Dataset.from_tensor_slices(train_x)
    y_train = tf.data.Dataset.from_tensor_slices(train_y)

    img_iter = iter(x_train)
    label_iter = iter(y_train)
    img_item = next(img_iter)
    img_item_numpy = img_item.numpy()
    print(type(img_item))
    print(type(img_item_numpy))

    # plot_images(x_train, y_train)


def one_hot(label, C, axis):
    res = tf.one_hot(label, C, axis=axis)
    return res


def run():
    train_x, train_y, test_x, test_y = load_datasets()
    x_train = tf.data.Dataset.from_tensor_slices(train_x)
    y_train = tf.data.Dataset.from_tensor_slices(train_y)

    # x_norm = x_train.map(normalize)
    # print(type(x_norm))
    # print(x_norm.element_spec)
    # first_item = next(iter(x_norm))
    # print(first_item)

    # Y = linear_function()
    # print(Y)

    # s = sigmoid(tf.math.log(3.0))
    # print(s)

    C = 5
    labels_1 = np.random.permutation(np.arange(0, 4))
    labels_2 = np.random.permutation(np.arange(0, 4))
    labels = np.array([labels_1, labels_2])

    print(labels)
    for label in labels:
        y_label = one_hot(label, C, -1)
        print(y_label)

    print('\n\n')

    for label in labels:
        y_label = one_hot(label, C, 0)
        print(y_label)


if __name__ == '__main__':
    # investigate_data()
    run()
