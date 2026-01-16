import matplotlib.pyplot as plt
import numpy as np

def visualize_dataset(train_X, train_Y, ax):
    # print(train_X.shape)
    # print(train_Y.shape)
    ax.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral)


def plot_costs(costs, ax):
    ax.plot(costs, label="Cost function")
    ax.set_ylabel('Cost')
    ax.set_xlabel('Iterations per thousand')


def plot_decision_boundary(model, X, y, ax):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    ax.set_xlim((-0.75, 0.40))
    ax.set_ylim((-0.75, 0.65))
    ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    ax.set_ylabel('x2')
    ax.set_xlabel('x1')
    ax.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)


