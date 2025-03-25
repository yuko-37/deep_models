import matplotlib.pyplot as plt
import sklearn.datasets
import numpy as np
import random


def load_data_set():
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2, random_state=3)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    return train_X, train_Y


def investigate_data_set():
    X, Y = load_data_set()
    print(X.shape)
    print(Y.shape)

    x1_min, x1_max = X[0, :].min(), X[0, :].max()
    x2_min, x2_max = X[1, :].min(), X[1, :].max()

    h = 0.1
    x1_range = np.arange(x1_min, x1_max, h)
    x2_range = np.arange(x2_min, x2_max, h)
    print(f'x1_range shape = {x1_range.shape}')
    # print(x1_range)
    print(f'x2_range shape = {x2_range.shape}')
    # print(x2_range)
    print()

    x1, x2 = np.meshgrid(x1_range, x2_range)

    print(x1.shape)
    print(x2.shape)

    x1_ravel = x1.ravel()
    x2_ravel = x2.ravel()

    print(x1_ravel.shape)
    print(x2_ravel.shape)

    z_scatter = [random.randint(0, 1) for _ in range(len(x1_ravel))]
    plt.scatter(x1, x2, c=z_scatter, cmap=plt.cm.Spectral)
    plt.show()

    # z_ndarray = np.random.randint(0, 2, size=len(x1_ravel))
    # z_ndarray = z_ndarray.reshape(x1.shape)
    # plt.contourf(x1, x2, z_ndarray, cmap=plt.cm.Spectral)
    # plt.show()


if __name__ == '__main__':
    investigate_data_set()
