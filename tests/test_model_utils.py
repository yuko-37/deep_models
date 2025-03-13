import numpy as np
from testing_utils import do_check
from model_utils import *

def test_initialize_parameters_1():
    np.random.seed(1)
    n_x, n_h, n_y = 3, 2, 1

    actual = initialize_parameters(n_x, n_h, n_y)
    
    expected_W1 = np.array([[ 0.01624345, -0.00611756, -0.00528172],
                     [-0.01072969,  0.00865408, -0.02301539]])
    expected_b1 = np.array([[0.],
                            [0.]])
    expected_W2 = np.array([[ 0.01744812, -0.00761207]])
    expected_b2 = np.array([[0.]])

    expected = {
        'W1': expected_W1,
        'b1': expected_b1,
        'W2': expected_W2,
        'b2': expected_b2
    }

    do_check('datatype_check', actual, expected)
    do_check('output_check', actual, expected)


def test_initialize_parameters_test_2():
    np.random.seed(1)
    n_x, n_h, n_y = 4, 3, 2

    actual = initialize_parameters(n_x, n_h, n_y)
    
    expected_W1 = np.array([[ 0.01624345, -0.00611756, -0.00528172, -0.01072969],
                            [ 0.00865408, -0.02301539,  0.01744812, -0.00761207],
                            [ 0.00319039, -0.0024937,   0.01462108, -0.02060141]])
    expected_b1 = np.array([[0.],
                            [0.],
                            [0.]])
    expected_W2 = np.array([[-0.00322417, -0.00384054,  0.01133769],
                            [-0.01099891, -0.00172428, -0.00877858]])
    expected_b2 = np.array([[0.],
                            [0.]])
    expected = {"W1": expected_W1,
                  "b1": expected_b1,
                  "W2": expected_W2,
                  "b2": expected_b2}
    
    do_check('datatype_check', actual, expected)
    do_check('output_check', actual, expected)


def test_forward():
    np.random.seed(1)
    A_prev = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    expected_cache = (A_prev, W, b)
    expected_Z = np.array([[ 3.26295337, -1.23429987]])
    expected_output = (expected_Z, expected_cache)


