from tests.utils import do_checks
from models.deep_model import *


def test_initialize_parameters():
    np.random.seed(3)
    layer_dims = [5, 4, 3]
    expected_W1 = np.array([[0.799899,  0.195213,  0.043155, -0.833379, -0.124052],
                            [-0.158653, -0.037003, -0.280403, -0.019596, -0.213418],
                            [-0.587578,  0.395615,  0.394137,  0.764544,  0.022376],
                            [-0.180977, -0.243892, -0.691606,  0.439328, -0.492412]])
    expected_b1 = np.array([[0.],
                            [0.],
                            [0.],
                            [0.]])
    expected_W2 = np.array([[-0.592523, -0.102825,  0.743074,  0.118358],
                            [-0.511893, -0.356497,  0.312622, -0.080257],
                            [-0.384418, -0.115015,  0.372528,  0.988055]])
    expected_b2 = np.array([[0.],
                            [0.],
                            [0.]])
    expected = {"W1": expected_W1,
                       "b1": expected_b1,
                       "W2": expected_W2,
                       "b2": expected_b2}
    actual = initialize_parameters(layer_dims)

    do_checks(actual, expected)


def test_L_model_forward():
    np.random.seed(6)
    X = np.random.randn(5, 4)
    W1 = np.random.randn(4, 5)
    b1 = np.random.randn(4, 1)
    W2 = np.random.randn(3, 4)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    expected_cache = [(),
                      (np.array([[-5.23825714, 3.18040136, 0.4074501, -1.88612721],
                                 [-2.77358234, -0.56177316, 3.18141623, -0.99209432],
                                 [4.18500916, -1.78006909, -0.14502619, 2.72141638],
                                 [5.05850802, -1.25674082, -3.54566654, 3.82321852]]),
                       np.array([[-0.31178367, 0.72900392, 0.21782079, -0.8990918],
                                 [-2.48678065, 0.91325152, 1.12706373, -1.51409323],
                                 [1.63929108, -0.4298936, 2.63128056, 0.60182225],
                                 [-0.33588161, 1.23773784, 0.11112817, 0.12915125],
                                 [0.07612761, -0.15512816, 0.63422534, 0.810655]]),
                       np.array([[0.35480861, 1.81259031, -1.3564758, -0.46363197, 0.82465384],
                                 [-1.17643148, 1.56448966, 0.71270509, -0.1810066, 0.53419953],
                                 [-0.58661296, -1.48185327, 0.85724762, 0.94309899, 0.11444143],
                                 [-0.02195668, -2.12714455, -0.83440747, -0.46550831, 0.23371059]]),
                       np.array([[1.38503523],
                                 [-0.51962709],
                                 [-0.78015214],
                                 [0.95560959]])),

                      (np.array([[2.2644603, 1.09971298, -2.90298027, 1.54036335],
                                 [6.33722569, -2.38116246, -4.11228806, 4.48582383],
                                 [10.37508342, -0.66591468, 1.63635185, 8.17870169]]),
                       np.array([[0., 3.18040136, 0.4074501, 0.],
                                 [0., 0., 3.18141623, 0.],
                                 [4.18500916, 0., 0., 2.72141638],
                                 [5.05850802, 0., 0., 3.82321852]]),
                       np.array([[-0.12673638, -1.36861282, 1.21848065, -0.85750144],
                                 [-0.56147088, -1.0335199, 0.35877096, 1.07368134],
                                 [-0.37550472, 0.39636757, -0.47144628, 2.33660781]]),
                       np.array([[1.50278553],
                                 [-0.59545972],
                                 [0.52834106]])
                       ),

                      (np.array([[-3.19864676, 0.87117055, -1.40297864, -3.00319435]]),
                       np.array([[2.2644603, 1.09971298, 0., 1.54036335],
                                 [6.33722569, 0., 0., 4.48582383],
                                 [10.37508342, 0., 1.63635185, 8.17870169]]),
                       np.array([[0.9398248, 0.42628539, -0.75815703]]),
                       np.array([[-0.16236698]]))
                      ]
    expected_AL = np.array([[0.03921668, 0.70498921, 0.19734387, 0.04728177]])
    expected_output = (expected_AL, expected_cache)
    actual_output = L_model_forward(X, parameters)

    do_checks(actual_output, expected_output)


def test_forward():
    np.random.seed(2)
    A_prev = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    Z = np.array([[ 3.43896131, -2.08938436]])

    expected_cache = (Z, A_prev, W, b)
    expected_A_relu = np.array([[3.43896131, 0.]])
    expected_A_sigmoid = np.array([[0.96890023, 0.11013289]])
    expected_output_relu = (expected_A_relu, expected_cache)
    expected_output_sigmoid = (expected_A_sigmoid, expected_cache)

    actual_output_relu = forward(A_prev, W, b, 'relu')
    actual_output_sigmoid = forward(A_prev, W, b, 'sigmoid')



    do_checks(actual_output_relu, expected_output_relu)
    do_checks(actual_output_sigmoid, expected_output_sigmoid)


def test_cost_compute():
    Y = np.array([[1, 1, 0]])
    AL = np.array([[.8,.9,0.4]])

    expected = np.array(0.27977656)
    actual = cost_compute(AL, Y)

    do_checks(actual, expected)


def test_backward():
    np.random.seed(2)
    dA = np.random.randn(1, 2)
    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    Z = np.random.randn(1, 2)
    cache = (Z, A, W, b)

    expected_dA_prev_sigmoid = np.array([[0.11017994, 0.01105339],
                                         [0.09466817, 0.00949723],
                                         [-0.05743092, -0.00576154]])
    expected_dW_sigmoid = np.array([[0.10266786, 0.09778551, -0.01968084]])
    expected_db_sigmoid = np.array([[-0.05729622]])
    expected_output_sigmoid = (expected_dA_prev_sigmoid,
                               expected_dW_sigmoid,
                               expected_db_sigmoid)

    expected_dA_prev_relu = np.array([[0.44090989, 0.],
                                      [0.37883606, 0.],
                                      [-0.2298228, 0.]])
    expected_dW_relu = np.array([[0.44513824, 0.37371418, -0.10478989]])
    expected_db_relu = np.array([[-0.20837892]])
    expected_output_relu = (expected_dA_prev_relu,
                            expected_dW_relu,
                            expected_db_relu)

    actual_output_sigmoid = backward(dA, cache, 'sigmoid')
    actual_output_relu = backward(dA, cache, 'relu')

    do_checks(actual_output_sigmoid, expected_output_sigmoid)
    do_checks(actual_output_relu, expected_output_relu)


def test_L_model_backward():
    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    cache_layer_0 = ()
    A0 = np.random.randn(4, 2)
    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    Z1 = np.random.randn(3, 2)
    cache_layer_1 = (Z1, A0, W1, b1)

    A1 = np.random.randn(3, 2)
    W2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)
    Z2 = np.random.randn(1, 2)
    cache_layer_2 = (Z2, A1, W2, b2)

    caches = [cache_layer_0, cache_layer_1, cache_layer_2]

    expected_dA1 = np.array([[0.12913162, -0.44014127],
                             [-0.14175655, 0.48317296],
                             [0.01663708, -0.05670698]])
    expected_dW2 = np.array([[-0.39202432, -0.13325855, -0.04601089]])
    expected_db2 = np.array([[0.15187861]])
    expected_dA0 = np.array([[0., 0.52257901],
                             [0., -0.3269206],
                             [0., -0.32070404],
                             [0., -0.74079187]])
    expected_dW1 = np.array([[0.41010002, 0.07807203, 0.13798444, 0.10502167],
                             [0., 0., 0., 0.],
                             [0.05283652, 0.01005865, 0.01777766, 0.0135308]])
    expected_db1 = np.array([[-0.22007063],
                             [0.],
                             [-0.02835349]])
    expected_output = {'dA1': expected_dA1,
                       'dW2': expected_dW2,
                       'db2': expected_db2,
                       'dA0': expected_dA0,
                       'dW1': expected_dW1,
                       'db1': expected_db1
                       }

    actual_output = L_model_backward(AL, Y, caches)
    do_checks(actual_output, expected_output)


def test_update_parameters():
    np.random.seed(2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    np.random.seed(3)
    dW1 = np.random.randn(3,4)
    db1 = np.random.randn(3,1)
    dW2 = np.random.randn(1,3)
    db2 = np.random.randn(1,1)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    learning_rate = 0.1
    expected_W1 = np.array([[-0.59562069, -0.09991781, -2.14584584,  1.82662008],
        [-1.76569676, -0.80627147,  0.51115557, -1.18258802],
        [-1.0535704 , -0.86128581,  0.68284052,  2.20374577]])
    expected_b1 = np.array([[-0.04659241],
            [-1.28888275],
            [ 0.53405496]])
    expected_W2 = np.array([[-0.55569196,  0.0354055 ,  1.32964895]])
    expected_b2 = np.array([[-0.84610769]])
    expected_output = {"W1": expected_W1,
                       'b1': expected_b1,
                       'W2': expected_W2,
                       'b2': expected_b2
                       }
    actual_output = update_parameters(parameters, grads, learning_rate)

    do_checks(actual_output, expected_output)