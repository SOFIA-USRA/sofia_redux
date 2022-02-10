# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.flags.motion_flags import MotionFlags
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D


def test_motion_flags():
    flags = MotionFlags
    all_letters = 'xyzijkXYZMntscp'
    for letter in all_letters:
        assert flags.flag_to_letter(flags.letter_to_flag(letter)) == letter


def test_convert_flag():
    flags = MotionFlags
    f = flags.flags
    assert flags.convert_flag(f.NORM) == f.NORM
    assert flags.convert_flag(2) == f.Y
    assert flags.convert_flag('norm') == f.NORM
    assert flags.convert_flag('|x|') == f.X_MAGNITUDE
    assert flags.convert_flag('x^2') == f.X2
    assert flags.convert_flag('mag') == f.MAGNITUDE
    assert flags.convert_flag('nor') == f.NORM
    with pytest.raises(ValueError) as err:
        _ = flags.convert_flag('foo')
    assert 'Unknown flag' in str(err.value)
    with pytest.raises(ValueError) as err:
        _ = flags.convert_flag(1.0)
    assert "Invalid flag type" in str(err.value)


def test_init():
    c = Coordinate2D(np.arange(10).reshape((2, 5)))
    flags = MotionFlags('x')
    f = flags.flags
    assert flags.direction == f.X
    assert np.allclose(flags(c), np.arange(5))
    flags = MotionFlags('CHOPPER')
    assert np.all(np.isnan(flags(c)))


def test_get_value():
    c = Coordinate2D(np.arange(10).reshape((2, 5)) - 3)
    f = MotionFlags('y')
    assert np.allclose(f.get_value(c), [2, 3, 4, 5, 6])
    f = MotionFlags('x')
    assert np.allclose(f.get_value(c), [-3, -2, -1, 0, 1])
    f = MotionFlags('|x|')
    assert np.allclose(f.get_value(c), [3, 2, 1, 0, 1])
    f = MotionFlags('y^2')
    assert np.allclose(f.get_value(c), [4, 9, 16, 25, 36])
    f = MotionFlags('mag')
    assert np.allclose(f.get_value(c), [3.6055, 3.6055, 4.1231, 5, 6.0827],
                       atol=1e-3)
    f = MotionFlags('norm')
    assert np.allclose(f.get_value(c), [3.6055, 3.6055, 4.1231, 5, 6.0827],
                       atol=1e-3)


def test_call():
    c = Coordinate2D(np.arange(10).reshape((2, 5)))
    f = MotionFlags('x')
    assert np.allclose(f(c), [0, 1, 2, 3, 4])


def test_str():
    f = MotionFlags('norm')
    assert str(f) == 'MotionFlags: MotionFlagTypes.NORM'
