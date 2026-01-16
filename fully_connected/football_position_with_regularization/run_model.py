import scipy

from numpy_FC_regularization_model import model, forward_propagation
from visualize_utils import *


def predict(X, Y, parameters, label):
    AL, _ = forward_propagation(X, parameters)
    p = (AL > 0.5).astype(int)
    print(f'Accuracy {label} = {np.mean(p[0,:] == Y[0, :])}')


def predict_dec(parameters, X):
    a3, cache = forward_propagation(X, parameters)
    predictions = (a3 > 0.5)
    return predictions


def load_2D_dataset():
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    return train_X, train_Y, test_X, test_Y


if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = load_2D_dataset()
    # train No regularization
    parameters_ng, costs_ng = model(train_X, train_Y, lambd=0, keep_probs=1)
    predict(train_X, train_Y, parameters_ng, '(No Regularization) train data')
    predict(test_X, test_Y, parameters_ng, '(No Regularization) test set')
    print()

    # train L2 regularization
    parameters_L2, costs_L2 = model(train_X, train_Y, lambd=0.7, keep_probs=1)
    predict(train_X, train_Y, parameters_ng, '(L2 Regularization) train data')
    predict(test_X, test_Y, parameters_ng, '(L2 Regularization) test set')
    print()

    # train Dropout regularization
    parameters_dropout, costs_dropout = model(train_X, train_Y, lambd=0, keep_probs=0.85)
    predict(train_X, train_Y, parameters_ng, '(Dropout Regularization) train data')
    predict(test_X, test_Y, parameters_ng, '(Dropout Regularization) test set')
    print()

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    plot_decision_boundary(lambda x: predict_dec(parameters_ng, x.T), train_X, train_Y, axes[0][0])
    axes[0][0].set_title("Decision Boundary No Regularization")

    plot_costs(costs_ng, axes[0][1])
    axes[0][1].set_title("Cost Function")

    plot_decision_boundary(lambda x: predict_dec(parameters_L2, x.T), train_X, train_Y, axes[1][0])
    axes[1][0].set_title("Decision Boundary L2 Regularization")

    plot_costs(costs_L2, axes[1][1])
    axes[1][1].set_title("Cost Function")

    plot_decision_boundary(lambda x: predict_dec(parameters_dropout, x.T), train_X, train_Y, axes[2][0])
    axes[2][0].set_title("Decision Boundary Dropout Regularization")

    plot_costs(costs_dropout, axes[2][1])
    axes[2][1].set_title("Cost Function")

    plt.tight_layout()
    plt.show()