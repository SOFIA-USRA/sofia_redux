# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import stretch_correction

import numpy as np


def test_single_values():

    c = stretch_correction(1, 1, 0)
    assert c == 0

    c = stretch_correction(0.5, 1, 0)
    assert c < 0

    c = stretch_correction(2, 1, 0)
    assert c > 0


def test_vectors():
    rchi2 = np.linspace(0, 2, 11)
    c = stretch_correction(rchi2, 1.0, 0.0)
    assert c[5] == 0
    assert np.all(c[:5] < 0)
    assert np.all(c[6:] > 0)


def test_density():
    density = np.linspace(0, 3, 100)
    c = stretch_correction(1.0, density, 0.0)
    assert np.allclose(c, 0)

    # test correction factor decreases with increasing density for rchi2 < 1
    c = stretch_correction(0, density, 0)
    grad = c[1:] - c[:-1]
    assert np.all(grad < 0)

    # test correction factor increases with increasing density for rchi2 > 1
    c = stretch_correction(2, density, 0)
    grad = c[1:] - c[:-1]
    assert np.all(grad > 0)


def test_offset():
    offset = np.linspace(0, 3, 100)
    c = stretch_correction(1.0, 1.0, offset)
    assert np.allclose(c, 0)

    # test correction factor increases with increasing offset for rchi2 < 1
    c = stretch_correction(0.0, 1.0, offset)
    grad = c[1:] - c[:-1]
    assert np.all(grad > 0)

    # test correction factor decreases with increasing offset for rchi2 > 1
    c = stretch_correction(2.0, 1.0, offset)
    grad = c[1:] - c[:-1]
    assert np.all(grad < 0)


def test_extreme_offset_values():
    # High offset values break numpy
    c = stretch_correction(2.0, 1.0, 100)
    assert c == 0
