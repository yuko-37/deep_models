import os
import sklearn.datasets
from numpy_moon_data_binary_classifier import model, forward_propagation, schedule_lr
from visualize_utils import *
from save_load_parameters import *


def load_moon_dataset():
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2, random_state=3)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    return train_X, train_Y


def predict(X, Y, parameters, dataset_name="", print_accuracy=True):
    m = X.shape[1]
    probs, caches = forward_propagation(X, parameters)
    predictions = (probs > 0.5).astype(int)

    if print_accuracy:
        int_predictions = predictions.astype(int)
        accuracy = np.sum((int_predictions == Y) / m)
        print(f"Accuracy {dataset_name} = {accuracy:.4}")
    return predictions


def run_model(optimizer):
    train_X, train_Y = load_moon_dataset()
    layers_dims = [train_X.shape[0], 5, 2, 1]
    # layers_dims = [train_X.shape[0], 1]
    learning_rate = 0.1

    if not os.path.isfile("parameters/parameters.pkl"):
        print('Run learning...')
        parameters, costs = model(train_X, train_Y, layers_dims, optimizer, learning_rate=learning_rate,
                                  decay=schedule_lr)
        save_to(parameters, layers_dims, "parameters/parameters.pkl")
        plot_cost(costs, f"Model with {optimizer} optimization")
    else:
        print('Restore parameters from file.')
        parameters = load_from("parameters/parameters.pkl")

    predict(train_X, train_Y, parameters)
    plot_decision_boundary(lambda x: predict(x, train_Y, parameters, print_accuracy=False), train_X, train_Y)


if __name__ == '__main__':
    # ['gd', 'momentum', 'adam']
    run_model('adam')

