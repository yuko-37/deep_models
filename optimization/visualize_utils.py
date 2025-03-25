import matplotlib.pyplot as plt
import numpy as np


def plot_cost(costs, title):
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title(title)
    plt.show()


def plot_decision_boundary(predict_lambda, X, Y):
    x1_min, x1_max = X[0, :].min(), X[0, :].max()
    x2_min, x2_max = X[1, :].min(), X[1, :].max()
    h = 0.05
    x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    X_grid = np.c_[x1.ravel(), x2.ravel()].T

    # print(X_grid.shape)
    assert(X_grid.shape[0] == 2)

    z = predict_lambda(X_grid)
    z = z.reshape(x1.shape)
    plt.contourf(x1, x2, z, cmap = plt.cm.Spectral)

    plt.scatter(X[0, :], X[1, :], c=Y, cmap=plt.cm.Spectral)

    plt.show()




