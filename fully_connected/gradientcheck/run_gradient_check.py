import numpy as np
from model import forward_propagation, backward_propagation, gradient_check


def run_gradient_check():
    X, Y, parameters = generate_test_case()
    print(X.shape)
    print(Y.shape)
    AL, cache = forward_propagation(X, parameters)
    grads = backward_propagation(X, Y, cache)
    gradient_check(X, Y, parameters, grads)


def generate_test_case():
    np.random.seed(1)
    x = np.random.randn(4,3)
    y = np.array([1, 1, 0]).reshape(1, -1)
    W1 = np.random.randn(5,4)
    b1 = np.random.randn(5,1)
    W2 = np.random.randn(3,5)
    b2 = np.random.randn(3,1)
    W3 = np.random.randn(1,3)
    b3 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    return x, y, parameters


if __name__ == '__main__':
    run_gradient_check()

