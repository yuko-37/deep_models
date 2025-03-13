import numpy as np
import copy


def initialize_parameters(n_x, n_h, n_y):
    parameters = {}
    parameters['W1'] = np.random.randn(n_h, n_x) * 0.01
    parameters['b1'] = np.zeros((n_h, 1))
    parameters['W2'] = np.random.randn(n_y, n_h) * 0.01
    parameters['b2'] = np.zeros((n_y, 1))
    return parameters


def relu(X):
    return np.maximum(0, X)

def d_relu(X):
    return  np.array(X >= 0, dtype=int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {}
    cache['Z1'] = Z1
    cache['A1'] = A1
    cache['Z2'] = Z2
    cache['A2'] = A2
    cache['W2'] = W2
    
    return A2, cache


def cost_compute(AL, Y):
    m = Y.shape[1]
    cost = -1 * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y),np.log(1-AL)), axis=1, keepdims=True) / m
    cost = np.squeeze(cost)
    return cost


def backward(X, Y, cache):
    m = Y.shape[1]

    A1 = cache['A1']
    A2 = cache['A2']
    W2 = cache['W2']
    Z1 = cache['Z1']

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = np.multiply(np.dot(W2.T, dZ2), d_relu(Z1))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {}
    grads['dW1'] = dW1
    grads['db1'] = db1
    grads['dW2'] = dW2
    grads['db2'] = db2

    return grads


def update_parameters(params, grads, learning_rate):
    parameters = copy.deepcopy(params)

    parameters['W1'] = parameters['W1'] - learning_rate * grads['dW1']
    parameters['b1'] = parameters['b1'] - learning_rate * grads['db1']
    parameters['W2'] = parameters['W2'] - learning_rate * grads['dW2']
    parameters['b2'] = parameters['b2'] - learning_rate * grads['db2']

    return parameters

# TODO 2-layers model
# LINEAR->RELU->LINEAR->SIGMOID
# initialize_parameters(n_x, n_h, n_y)
# forward()
#   relu() & sigmoid()
# cost_compute()
# backward()
# update_parameters()
