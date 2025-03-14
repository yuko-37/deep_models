from models.deep_model import *


def output_check(actual, expected):
    if hasattr(actual, 'shape'):
        np.testing.assert_array_almost_equal(actual, expected)
    else:
        assert actual == expected


def datatype_check(one, two):
    assert type(one) == type(two)


def shape_check(actual, expected):
    if hasattr(actual, 'shape'):
        assert actual.shape == expected.shape


def do_checks(actual, expected):
    if isinstance(actual, dict):
        for key in actual:
            do_checks(actual[key], expected[key])
    elif isinstance(actual, tuple) or isinstance(actual, list):
        for i in range(len(actual)):
            do_checks(actual[i], expected[i])
    else:
        datatype_check(actual, expected)
        if hasattr(actual, 'shape'):
            shape_check(actual, expected)
        output_check(actual, expected)
