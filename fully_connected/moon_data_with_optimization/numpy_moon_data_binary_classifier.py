import numpy as np
import math


def model(X, Y, layers_dims, optimizer, num_iterations=5000, mini_batch_size=64, learning_rate=0.0007,
          beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, print_cost=True, decay=None, decay_rate=1):
    m = X.shape[1]
    seed = 10
    parameters = initialize_parameters(layers_dims)
    costs = []
    t = 0
    v, s = initialize_v_s(parameters)
    learning_rate0 = learning_rate

    for i in range(num_iterations):
        cost = 0
        seed += 1
        mini_batches = split_by_mini_batches(X, Y, mini_batch_size, seed)

        for mini_X, mini_Y in mini_batches:
            AL, caches = forward_propagation(mini_X, parameters)
            cost += cost_compute(AL, mini_Y)
            grads = backward_propagation(AL, mini_Y, caches)
            t += 1
            parameters, v, s = update_parameters(parameters, grads, optimizer, learning_rate,
                                           v, s, t, beta, beta1, beta2, epsilon)

        if decay:
            learning_rate = decay(learning_rate0, decay_rate, i)

        cost = cost / m

        if (i % 1000 == 0 or i == num_iterations-1) and print_cost:
            print(f"Cost after {i} iteration: {cost}")
            if decay:
                print(f"Learning rate after epoch {i}: {learning_rate}")
        if (i % 100 == 0) or i == num_iterations-1:
            costs.append(cost)

    return parameters, costs


def schedule_lr(learning_rate0, decay_rate, epoch_num):
    learning_rate = 1 / (1 + decay_rate * math.floor(epoch_num / 1000)) * learning_rate0
    return learning_rate


def update_parameters(parameters, grads, optimizer, learning_rate, v, s, t, beta, beta1, beta2, epsilon):
    assert(optimizer in ['gd', 'momentum', 'adam'])

    if optimizer == 'gd':
        parameters = update_parameters_gd(parameters, grads, learning_rate)
    elif optimizer == 'momentum':
        parameters, v = update_parameters_momentum(parameters, grads, v, learning_rate, beta)
    elif optimizer == 'adam':
        parameters, v, s = update_parameters_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)

    return parameters, v, s


def update_parameters_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}
    for l in range(1, L+1):
        v['dW' + str(l)] = beta1 * v['dW' + str(l)] + (1 - beta1) * grads['dW' + str(l)]
        v['db' + str(l)] = beta1 * v['db' + str(l)] + (1 - beta1) * grads['db' + str(l)]
        v_corrected['dW' + str(l)] = v['dW' + str(l)] / (1 - beta1 ** t)
        v_corrected['db' + str(l)] = v['db' + str(l)] / (1 - beta1 ** t)

        s['dW' + str(l)] = beta2 * s['dW' + str(l)] + (1 - beta2) * np.square(grads['dW' + str(l)])
        s['db' + str(l)] = beta2 * s['db' + str(l)] + (1 - beta2) * np.square(grads['db' + str(l)])
        s_corrected['dW' + str(l)] = s['dW' + str(l)] / (1 - beta2 ** t)
        s_corrected['db' + str(l)] = s['db' + str(l)] / (1 - beta2 ** t)

        parameters['W' + str(l)] -= (learning_rate * v_corrected['dW' + str(l)] /
                                      (np.sqrt(s_corrected['dW' + str(l)]) + epsilon))
        parameters['b' + str(l)] -= (learning_rate * v_corrected['db' + str(l)] /
                                      (np.sqrt(s_corrected['db' + str(l)]) + epsilon))

    return parameters, v, s


def update_parameters_momentum(parameters, grads, v, learning_rate, beta):
    L = len(parameters) // 2
    for l in range(1, L+1):
        v['dW' + str(l)] = beta * v['dW' + str(l)] + (1-beta) * grads['dW' + str(l)]
        v['db' + str(l)] = beta * v['db' + str(l)] + (1-beta) * grads['db' + str(l)]
        parameters['W'+str(l)] -= learning_rate * v['dW'+str(l)]
        parameters['b'+str(l)] -= learning_rate * v['db'+str(l)]
    return parameters, v


def update_parameters_gd(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L+1):
        parameters['W'+str(l)] -= learning_rate * grads['dW'+str(l)]
        parameters['b'+str(l)] -= learning_rate * grads['db'+str(l)]
    return parameters


def split_by_mini_batches(X, Y, size, seed):
    np.random.seed(seed)
    mini_batches = []
    m = X.shape[1]
    permutation = np.random.permutation(m)
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    assert(shuffled_X.shape == X.shape)
    assert(shuffled_Y.shape == Y.shape)

    num_batches = math.floor(m / size)

    for i in range(num_batches):
        mini_batch_X = shuffled_X[:, i*size: (i+1)*size]
        mini_batch_Y = shuffled_Y[:, i*size: (i+1)*size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def forward_propagation(X, parameters):
    A = X
    L = len(parameters) // 2 # because pairs W1,b1, ..., WL,bL
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


def initialize_v(parameters):
    L = len(parameters) // 2
    v = {}
    for l in range(1, L+1):
        v['dW' + str(l)] = np.zeros(parameters['W' + str(l)].shape)
        v['db' + str(l)] = np.zeros(parameters['b' + str(l)].shape)
    return v


def initialize_v_s(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}
    for l in range(1, L+1):
        v['dW' + str(l)] = np.zeros(parameters['W' + str(l)].shape)
        v['db' + str(l)] = np.zeros(parameters['b' + str(l)].shape)
        s['dW' + str(l)] = np.zeros(parameters['W' + str(l)].shape)
        s['db' + str(l)] = np.zeros(parameters['b' + str(l)].shape)
    return v, s


def initialize_parameters(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layers_dims[l], 1))

    return parameters


def cost_compute(AL, Y):
    logprobs = -1 * (np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL)))
    cost = np.sum(logprobs)
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


def sigmoid_backward(dA, Z):
    S = 1/(1+np.exp(-Z))
    dZ = dA * S * (1-S)
    return dZ


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    mask = (Z <= 0)
    dZ[mask] = 0
    return dZ


def backward_propagation(AL, Y, caches):
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
