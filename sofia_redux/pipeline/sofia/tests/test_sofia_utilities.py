# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.pipeline.sofia import sofia_utilities as u


def test_parse_apertures(capsys):
    # test helper function for parsing apertures from
    # parameters or headers

    # nominal input
    input_position = '1,2,3;4,5,6'
    expected = [[1., 2., 3.], [4., 5., 6.]]
    assert np.allclose(u.parse_apertures(input_position, 2), expected)

    # one file: error
    with pytest.raises(ValueError):
        u.parse_apertures(input_position, 1)
    assert 'Could not read input_position' in capsys.readouterr().err

    # two files, one input: applied to all
    input_position = '1,2,3'
    expected = [[1., 2., 3.], [1., 2., 3.]]
    assert np.allclose(u.parse_apertures(input_position, 2), expected)

    # bad value in aperture: error
    input_position = '1,2,3;4,5a,6'
    with pytest.raises(ValueError):
        u.parse_apertures(input_position, 2)
    assert 'Could not read input_position' in capsys.readouterr().err


def test_parse_bg(capsys):
    # test helper function for parsing background regions

    # nominal input
    input_position = '1-2,3-4;5-6'
    expected = [[[1., 2.], [3., 4.]], [[5., 6.]]]
    result = u.parse_bg(input_position, 2)
    assert len(result) == 2
    for r, e in zip(result, expected):
        assert np.allclose(r, e)

    # one file: error
    with pytest.raises(ValueError):
        u.parse_bg(input_position, 1)
    assert 'Could not read background region' in capsys.readouterr().err

    # two files, one input: applied to all
    input_position = '1-2,3-4'
    expected = [[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]]
    result = u.parse_bg(input_position, 2)
    assert len(result) == 2
    for r, e in zip(result, expected):
        assert np.allclose(r, e)

    # bad value in region: error
    input_position = '1-2,3-4;5-6a'
    with pytest.raises(ValueError):
        u.parse_bg(input_position, 2)
    assert 'Could not read background region' in capsys.readouterr().err
