# Licensed under a 3-clause BSD style license - see LICENSE.rst

from argparse import Namespace

import numpy as np
import pytest

from sofia_redux.toolkit.utilities.base import Model


@pytest.fixture
def model():
    x = np.arange(100).astype(float)
    y = x * 2
    stats = Namespace
    stats.dof = 1
    stats.fit = y.copy()
    stats.residuals = y * 0
    stats.chi2 = 2
    stats.rchi2 = 3
    stats.q = 4
    stats.rms = 5
    m = Model(y, x)
    m.stats = stats
    return m


def test_properties(model):
    m = model
    assert m.state == 'initial fit'

    # Test error property
    assert isinstance(m.error, np.ndarray) and np.allclose(m.error, 1)
    m._error = m._interpolated_error
    m._interpolated_error = None
    assert isinstance(m.error, np.ndarray) and np.allclose(m.error, 1)


def test_repr(model):
    assert model.__repr__() == 'Model (1 features, None parameters)'


def test_str(model):
    assert 'Name: Model' in model.__str__()


def test_parameters_string(model):
    assert model._parameters_string() == ''


def test_stats_string(model):
    m = model
    s = m._stats_string()
    assert 'RMS deviation of fit : 5' in s
    assert 'Iteration termination' not in s
    m.robust = 1
    s = m._stats_string()
    assert 'Iteration termination' in s
    m.stats = None
    assert m._stats_string() == ''


def test_print(model):
    model.print_params()
    model.print_stats()


def test_call(model):
    m = model
    with pytest.raises(ValueError) as err:
        m(1, 1)
    assert "require 1 feature" in str(err.value).lower()

    with pytest.raises(ValueError) as err:
        m(1)
    assert "create this method" in str(err.value).lower()


def test_create_coordinates(model):
    m = model
    with pytest.raises(ValueError) as err:
        m._create_coordinates(np.arange(10))
    assert "require at least 2 arguments" in str(err.value).lower()

    # 3 args means coordinates are already generated
    args = 1, 2, 3
    assert m._create_coordinates(*args) == (1, 2, 3)

    c = m._create_coordinates(*m._samples)
    assert len(c) == 3
    assert np.allclose(c[0], np.arange(c[1].shape[0]))


def test_parse_args(model):
    m = model
    y, x = m._samples
    e = np.ones_like(y)
    mask = e.astype(bool)

    with pytest.raises(ValueError) as err:
        m._parse_args(e[:-1], mask, y, x)
    assert "error size does not match" in str(err.value).lower()

    with pytest.raises(ValueError) as err:
        m._parse_args(e, mask[:-1], y, x)
    assert "mask size does not match" in str(err.value).lower()

    m._parse_args(complex(2), mask, y, x)
    assert m._error == 1

    e[0] = 0.0
    mask[-1] = False
    y[1] = np.nan
    m._parse_args(e, mask, y, x)
    assert m._error[0] == 0
    assert np.allclose(m._error[1:], 1)
    assert not m._usermask[-1]
    assert not m._usermask[0:2].any()
    assert m._usermask[2:-1].all()

    m._ignorenans = False
    m._parse_args(e, mask, y, x)
    assert m._usermask[:-1].all()
    assert not m._usermask[-1]
