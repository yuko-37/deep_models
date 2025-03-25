from model import model, forward_propagation, schedule_lr
from data_utils import load_data_set
from visualize_utils import *
from save_load import *


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
    train_X, train_Y = load_data_set()
    layers_dims = [train_X.shape[0], 5, 2, 1]
    # layers_dims = [train_X.shape[0], 1]
    learning_rate = 0.1

    if parameters_dont_exist():
        print('Run learning...')
        parameters, costs = model(train_X, train_Y, layers_dims, optimizer, learning_rate=learning_rate,
                                  decay=schedule_lr)
        # save_to_file(parameters, layers_dims)
        plot_cost(costs, f"Model with {optimizer} optimization")
    else:
        print('Restore parameters from file.')
        parameters = load_from_file()

    predict(train_X, train_Y, parameters)
    plot_decision_boundary(lambda x: predict(x, train_Y, parameters, print_accuracy=False), train_X, train_Y)


if __name__ == '__main__':
    run_model('adam')
