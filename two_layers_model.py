import data_utils as data
from model_utils import *


def two_layers_model(X, Y, hidden_units, num_iterations=10, learning_rate=0.05):
    np.random.seed(1)
    n_x = X.shape[0]
    n_h = hidden_units
    n_y = Y.shape[0]

    grads = {}
    costs = []

    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(num_iterations):
        A2, cache = forward(X, parameters)
        grads = backward(X, Y, cache)
        parameters = update_parameters(parameters, grads, learning_rate)

        cost = cost_compute(A2, Y)

        if (i % 10 == 0 or i == num_iterations-1):
            print(f'{i} iteration: cost value = {cost}')
            costs.append(cost)

    return parameters, costs

extract = 50
train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = data.load_data()
X = train_x_orig.reshape(train_x_orig.shape[0], -1).T[:, :extract]
Y = train_y_orig[:, :extract]
print(X.shape)
print(Y.shape)
assert X.shape == (12288, extract)
assert Y.shape == (1, extract)

learning_rate = 0.05
hidden_units = 4
num_iterations = 100
parameters, costs = two_layers_model(X, Y, hidden_units, num_iterations, learning_rate)
data.plot_costs(costs, learning_rate)
