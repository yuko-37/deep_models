import numpy as np
from model_utils import *


def output_check(actual, expected):
    if hasattr(actual, 'shape'):
        np.testing.assert_array_almost_equal(actual, expected)
    else:
        assert actual == expected


def datatype_check(actual, expected):
    assert type(actual) == type(expected)


def shape_check(actual, expected):
    if hasattr(actual, 'shape'):
        assert actual.shape == expected.shape
    else:
        assert False


function_map = {
    'output_check': output_check,
    'datatype_check': datatype_check,
    'shape_check': shape_check
}


def do_check(check_name, actual, expected):
    if isinstance(actual, dict):
        for key in actual:
            do_check(check_name, actual[key], expected[key])
    elif isinstance(actual, tuple) or isinstance(actual, list):
        for i in range(len(actual)):
            do_check(check_name, actual[i], expected[i])
    else:
        function_map[check_name](actual, expected)


def multiple_test(test_cases, target):
    for test_case in test_cases:
        check_name = test_case['name']
        actual = target(*test_case['input'])
        expected = test_case['expected']
        do_check(check_name, actual, expected)