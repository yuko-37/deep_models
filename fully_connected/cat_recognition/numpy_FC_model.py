import numpy as np


def L_layers_model(X, Y, layers_dims, num_iterations, learning_rate):
    parameters = initialize_parameters(layers_dims)
    costs = []

    # cache[i] = (Zi, Ai-1, Wi, bi)
    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if (i % 100 == 0) or (i == num_iterations-1):
            cost = cost_compute(AL, Y)
            print(f"Cost value after {i} iterations = {cost}")
            costs.append(cost)

    return parameters, costs


def initialize_parameters(layers_dims):
    parameters = {}
    L = len(layers_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) / np.sqrt(layers_dims[l-1]) # * 0.01
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layers_dims[l], 1))

    return parameters


def cost_compute(AL, Y):
    m = Y.shape[1]
    cost = -1 * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL)), axis=1, keepdims=True) / m
    cost = np.squeeze(cost)
    return cost


def backward(dA, cache, activation):
    Z, A_prev, W, b = cache
    m = A_prev.shape[1]

    if activation == "relu":
        dZ = relu_backward(dA, Z)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, Z)

    else:
        raise Exception(f"Unknown activation {activation}")

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    assert(dA_prev.shape == A_prev.shape)

    return dA_prev, dW, db


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(1, L+1):
        parameters['W'+str(l)] = parameters['W'+str(l)] - learning_rate * grads['dW'+str(l)]
        parameters['b'+str(l)] = parameters['b'+str(l)] - learning_rate * grads['db'+str(l)]

    return parameters


def L_model_forward(X, parameters):
    A = X
    L = len(parameters) // 2 # pairs W1,b1, ..., WL,bL
    caches = [] # cache[i] - (Zi, Ai-1, Wi, bi)
    caches.append(()) # cache[0] пустой

    for i in range(1, L):
        A_prev = A
        W = parameters['W' + str(i)]
        b = parameters['b' + str(i)]
        A, cache = forward(A_prev, W, b, 'relu')
        caches.append(cache)

    A_prev = A
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    AL, cache = forward(A_prev, W, b, 'sigmoid')
    caches.append(cache)

    return AL, caches


def forward(A_prev, W, b, activation):
    Z = W.dot(A_prev) + b
    assert (Z.shape == (W.shape[0], A_prev.shape[1]))

    if activation == "sigmoid":
        A, Z = sigmoid(Z)

    elif activation == "relu":
        A, Z = relu(Z)

    else:
        raise Exception(f"Unknown activation [{activation}]")

    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (Z, A_prev, W, b)

    return A, cache


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z


def relu(Z):
    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)

    return A, Z


def sigmoid_backward(dA, Z):
    S = 1/(1+np.exp(-Z))
    dZ = dA * S * (1-S)
    return dZ


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    mask = (Z <= 0)
    dZ[mask] = 0
    return dZ


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) - 1

    assert(AL.shape == Y.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L]
    dA_prev, dW, db = backward(dAL, current_cache, 'sigmoid')
    grads['dA'+str(L-1)] = dA_prev
    grads['dW'+str(L)] = dW
    grads['db'+str(L)] = db

    for l in reversed(range(1, L)):
        current_cache = caches[l]
        dA_prev = grads['dA' + str(l)]
        dA_calculated, dW, db = backward(dA_prev, current_cache, 'relu')
        grads['dA' + str(l-1)] = dA_calculated
        grads['dW' + str(l)] = dW
        grads['db' + str(l)] = db

    return grads
