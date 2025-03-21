import numpy as np


def model(X, Y, learning_rate=0.3, num_iterations=30_000, printCost=True, lambd=0, keep_probs=1):
    costs = []
    layers_dims = (X.shape[0], 20, 3, 1)
    parameters = initialize_parameters(layers_dims)

    assert(lambd == 0 or keep_probs == 1)

    for i in range(num_iterations):
        if keep_probs == 1:
            A3, cache = forward_propagation(X, parameters)
        else:
            A3, cache = forward_propagation_with_dropout(X, parameters, keep_probs)

        if lambd == 0:
            if keep_probs == 1:
                grads = backward_propagation(X, Y, cache)
            else:
                grads = backward_propagation_with_dropout(X, Y, cache, keep_probs)
        else:
            grads = backward_propagation_with_lambd(X, Y, lambd, cache)

        parameters = update_parameters(parameters, grads, learning_rate)

        if (i % 1000 == 0) or (i == num_iterations - 1):
            if lambd == 0:
                cost = cost_compute(A3, Y)
            else:
                cost = cost_compute_with_lambd(A3, Y, parameters, lambd)
            print(f"Cost after {i} iteration {cost}.")
            costs.append(cost)

    return parameters, costs


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


def forward_propagation_with_dropout(X, parameters, keep_probs):
    np.random.seed(1)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    D1 = (D1 < keep_probs).astype(int)
    A1 = np.multiply(A1, D1)
    A1 /= keep_probs

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = (D2 < keep_probs).astype(int)
    A2 = np.multiply(A2, D2)
    A2 /= keep_probs

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    return A3, (A3, Z3, A2, Z2, A1, Z1, W1, W2, W3, D1, D2)

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


def backward_propagation_with_lambd(X, Y, lambd, cache):
    m = X.shape[1]
    A3, Z3, A2, Z2, A1, Z1, W1, W2, W3 = cache
    grads = backward_propagation(X, Y, cache)

    grads['dW1'] = grads['dW1'] + lambd * W1 / m
    grads['dW2'] = grads['dW2'] + lambd * W2 / m
    grads['dW3'] = grads['dW3'] + lambd * W3 / m

    return grads


def backward_propagation_with_dropout(X, Y, cache, keep_probs):
    grads = {}
    m = X.shape[1]
    A3, Z3, A2, Z2, A1, Z1, W1, W2, W3, D1, D2 = cache

    dZ3 = A3 - Y
    dW3 = np.dot(dZ3, A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m
    dA2 = np.dot(W3.T, dZ3)
    dA2 = np.multiply(dA2, D2)
    dA2 /= keep_probs

    # dZ2 = relu_backward(dA2, Z2)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dA1 = np.dot(W2.T, dZ2)
    dA1 = np.multiply(dA1, D1)
    dA1 /= keep_probs

    # dZ1 = relu_backward(dA1, Z1)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
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


def cost_compute_with_lambd(AL, Y, parameters,  lambd):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cost = cost_compute(AL, Y)
    regularization = (np.square(np.linalg.norm(W1)) + np.square(np.linalg.norm(W2)) + np.square(np.linalg.norm(W3))) * lambd / m / 2
    cost = cost + regularization

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