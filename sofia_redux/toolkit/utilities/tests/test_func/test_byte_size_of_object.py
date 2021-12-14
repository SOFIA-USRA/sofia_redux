# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.utilities.func import byte_size_of_object

import numpy as np


class dummy_class(object):
    def __init__(self, input_data):
        self.internal = 100
        self.input_data = input_data


def test_byte_size_of_object():
    test_float = np.random.random(100).astype(float)
    test_bool = test_float > 0.5
    float_size = byte_size_of_object(test_float)
    bool_size = byte_size_of_object(test_bool)
    assert float_size > bool_size

    float_obj = dummy_class(test_float)
    bool_obj = dummy_class(test_bool)
    float_obj_size = byte_size_of_object(float_obj)
    bool_obj_size = byte_size_of_object(bool_obj)
    assert float_obj_size > float_size
    assert (float_obj_size - bool_obj_size) == (float_size - bool_size)
