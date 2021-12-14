# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.utilities.multiprocessing import (
    pickle_object, unpickle_file, pickle_list, unpickle_list)

import numpy as np
import os
import shutil


def test_pickle_object(tmpdir):
    test_file = str(tmpdir.mkdir('pickling_tests').join('pickle_object.p'))
    data = np.random.random(100)
    assert pickle_object(data, None) is data
    out_file = pickle_object(data, test_file)
    assert out_file == test_file
    assert np.allclose(unpickle_file(out_file)[0], data)


def test_unpickle_file(tmpdir):
    test_file = str(tmpdir.mkdir('pickling_tests').join('unpickle_file.p'))
    data = np.random.random(50)
    out_file = pickle_object(data, test_file)

    result, filename = unpickle_file(None)
    assert filename is None and result is None

    bad_file = '_this_is_not_a_valid_file'
    result, filename = unpickle_file(bad_file)
    assert result == bad_file and filename is None

    result, filename = unpickle_file(out_file)
    assert np.allclose(result, data) and filename == out_file


def test_pickle_list():
    objects = [None, 1, 2, 'a', np.arange(10)]
    original_objects = objects.copy()
    ids = [id(obj) for obj in objects]
    pickle_directory = pickle_list(objects)
    assert os.path.isdir(pickle_directory)
    files = [os.path.join(pickle_directory, f) for f in
             os.listdir(pickle_directory)]

    for f in files:
        assert f in objects
        assert int(os.path.basename(f)[:-2]) in ids

    unpickle_list(objects)
    for i, obj in enumerate(objects):
        if not isinstance(obj, np.ndarray):
            assert obj == original_objects[i]
        else:
            assert np.allclose(obj, original_objects[i])

    shutil.rmtree(pickle_directory)

    pickle_directory = pickle_list(objects, class_type=np.ndarray)
    for i, obj in enumerate(objects):
        if isinstance(original_objects[i], np.ndarray):
            assert os.path.isfile(obj)
        else:
            assert obj == original_objects[i]
    unpickle_list(objects)
    for i, obj in enumerate(objects):
        if not isinstance(obj, np.ndarray):
            assert obj == original_objects[i]
        else:
            assert np.allclose(obj, original_objects[i])
    shutil.rmtree(pickle_directory)

    pickle_directory = pickle_list(objects, naming_attribute='dtype')
    assert 'int' in os.path.basename(objects[-1])
    shutil.rmtree(pickle_directory)


def test_unpickle_list():
    unpickle_list(None)

    objects = [1, 2, 3, 4]
    original_objects = objects.copy()
    pickle_directory = pickle_list(objects)

    pickle_files = objects.copy()
    unpickle_list(objects, delete=False)
    for i, obj in enumerate(objects):
        assert obj == original_objects[i]
        assert os.path.isfile(pickle_files[i])

    new_objects = pickle_files.copy()
    unpickle_list(new_objects, delete=True)
    for i, obj in enumerate(new_objects):
        assert obj == original_objects[i]
        assert not os.path.isfile(pickle_files[i])

    shutil.rmtree(pickle_directory)
