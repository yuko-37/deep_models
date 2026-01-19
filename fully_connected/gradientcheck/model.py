import numpy as np


def model(X, Y, parameters, learning_rate=0.3, num_iterations=1_000):
    costs = []

    for i in range(num_iterations):
        A3, cache = forward_propagation(X, parameters)
        grads = backward_propagation(X, Y, cache)
        parameters = update_parameters(parameters, grads, learning_rate)

        if (i % 100 == 0) or (i == num_iterations - 1):
            gradient_check(X, Y, parameters)
            cost = cost_compute(A3, Y)
            print(f"Cost after {i} iteration {cost}.")
            costs.append(cost)

    return parameters, costs


def gradient_check(X, Y, parameters, grads):
    epsilon = 1e-7
    params_vector = dictionary_to_vector(parameters, 'w')
    grads_vector = dictionary_to_vector(grads, 'dw')
    num_params = params_vector.shape[0]
    dJapprox = np.zeros(grads_vector.shape)

    for i in range(num_params):
        theta_plus = np.copy(params_vector)
        theta_plus[i] += epsilon
        AL, _ = forward_propagation(X, vector_to_dictionary(theta_plus))
        J_plus = cost_compute(AL, Y)
        theta_minus = theta_plus
        theta_minus[i] -= 2*epsilon
        AL, _ = forward_propagation(X, vector_to_dictionary(theta_minus))
        J_minus = cost_compute(AL, Y)
        dJapprox[i] = (J_plus - J_minus) / 2.0 / epsilon

    dJ = grads_vector
    numerator = np.linalg.norm(dJ - dJapprox)
    denominator = np.linalg.norm(dJapprox) + np.linalg.norm(dJ)
    norm = numerator / denominator

    if norm < 2*epsilon:
        print(f'Gradient is correct. Norm = {norm}')
    else:
        print(f'Warning: there may be error computing gradient, norm = {norm}')

    return norm


def dictionary_to_vector(dict, label):
    if label == 'w':
        keys = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
    elif label == 'dw':
        keys = ['dW1', 'db1', 'dW2', 'db2', 'dW3', 'db3']
    else:
        raise Exception(f'Unknown label {label}.')
    flatten_arrays = []
    for key in keys:
        arr = dict[key]
        flatten_arrays.append(arr.ravel())

    vector = np.concatenate(flatten_arrays).reshape(-1,1)
    return vector


def vector_to_dictionary(vector):
    # W1 = np.random.randn(5,4)
    # b1 = np.random.randn(5,1)
    # W2 = np.random.randn(3,5)
    # b2 = np.random.randn(3,1)
    # W3 = np.random.randn(1,3)
    # b3 = np.random.randn(1,1)
    parameters = {}
    parameters['W1'] = vector[: 20].reshape(5, 4)
    parameters['b1'] = vector[20: 25].reshape(5, 1)
    parameters["W2"] = vector[25: 40].reshape((3, 5))
    parameters["b2"] = vector[40: 43].reshape((3, 1))
    parameters["W3"] = vector[43: 46].reshape((1, 3))
    parameters["b3"] = vector[46: 47].reshape((1, 1))

    return parameters


def initialize_parameters(ldims):
    np.random.seed(3)
    parameters = {}
    L = len(ldims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(ldims[l], ldims[l-1]) / np.sqrt(ldims[l-1]) # np.sqrt(ldims[l-1] / 2)
        parameters['b' + str(l)] = np.zeros((ldims[l], 1))
    return parameters


def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    return A3, (A3, Z3, A2, Z2, A1, Z1, W1, W2, W3)


def backward_propagation(X, Y, cache):
    grads = {}
    m = X.shape[1]
    A3, Z3, A2, Z2, A1, Z1, W1, W2, W3 = cache

    dZ3 = A3 - Y
    dW3 = np.dot(dZ3, A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m
    dA2 = np.dot(W3.T, dZ3)

    dZ2 = relu_backward(dA2, Z2)
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dA1 = np.dot(W2.T, dZ2)

    dZ1 = relu_backward(dA1, Z1)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads['dW1'] = dW1
    grads['db1'] = db1
    grads['dW2'] = dW2
    grads['db2'] = db2
    grads['dW3'] = dW3
    grads['db3'] = db3

    return grads


def update_parameters(parameters, grads, learning_rate):
    n = len(parameters) // 2
    for i in range(1, n + 1):
        parameters['W' + str(i)] = parameters['W' + str(i)] - learning_rate * grads['dW' + str(i)]
        parameters['b' + str(i)] = parameters['b' + str(i)] - learning_rate * grads['db' + str(i)]

    return parameters


def cost_compute(AL, Y):
    m = Y.shape[1]
    # cost = -1 * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL)), axis=1, keepdims=True) / m
    # cost = np.squeeze(cost)

    cost = np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1-AL), 1-Y)
    cost = np.nansum(cost) / m
    return cost


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    mask = (Z <= 0)
    dZ[mask] = 0
    return dZ


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def relu(Z):
    return np.maximum(0, Z)