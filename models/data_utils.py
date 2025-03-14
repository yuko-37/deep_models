import h5py
import numpy as np


def load_data():
    train_dataset = h5py.File('../datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('../datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def load_reshaped_data():
    train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = load_data()
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    train_x = train_x_flatten / 255
    train_y = train_y_orig

    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    test_x = test_x_flatten / 255
    test_y = test_y_orig

    return train_x, train_y, test_x, test_y