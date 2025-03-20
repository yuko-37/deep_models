import numpy as np
from data_utils import load_2D_dataset
from visualize_utils import scatter, plot_costs
from model import model, forward_propagation


def predict(X, Y, parameters, label):
    AL, _ = forward_propagation(X, parameters)
    p = (AL > 0.5).astype(int)
    print(f'Accuracy {label} = {np.mean(p[0,:] == Y[0, :])}')

def run():
    train_X, train_Y, test_X, test_Y = load_2D_dataset()
    print(train_X.shape)
    print(train_Y.shape)
    # scatter(train_X, train_Y)

    # parameters, costs = model(train_X, train_Y, lambd=0, keep_probs=0.86)
    # parameters, costs = model(train_X, train_Y, lambd=0.1, keep_probs=1)
    parameters, costs = model(train_X, train_Y, lambd=0, keep_probs=1)
    predict(train_X, train_Y, parameters, 'train data')
    predict(test_X, test_Y, parameters, 'test set')
    plot_costs(costs)


if __name__ == '__main__':
    run()


