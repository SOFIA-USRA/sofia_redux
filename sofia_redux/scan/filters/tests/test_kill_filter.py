# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.filters.kill_filter import KillFilter
from sofia_redux.scan.utilities.range import Range


@pytest.fixture
def initialized_filter(populated_integration):
    return KillFilter(integration=populated_integration)


@pytest.fixture
def configured_filter(initialized_filter):
    f = initialized_filter
    f.set_integration(f.integration.copy())
    f.configuration.parse_key_value(f.get_config_name(), 'True')
    f.configuration.parse_key_value(f'{f.get_config_name()}.level', '2.0')
    f.configuration.parse_key_value(
        f'{f.get_config_name()}.bands',
        "0.0244140625:0.048828125,0.0732421875:0.078125")
    # frequency channels 5-10, 15-16
    return f


@pytest.fixture
def nonzero_filter(configured_filter):
    f = configured_filter
    f.reject[5:10] = True
    return f


def test_init(populated_integration):
    f = KillFilter()
    assert f.point_response is None
    assert f.rejected is None
    f = KillFilter(integration=populated_integration)
    assert isinstance(f.point_response, np.ndarray)
    assert np.allclose(f.point_response, 1)
    assert isinstance(f.rejected, np.ndarray)
    assert np.allclose(f.rejected, 0)
    assert np.allclose(f.reject, False)


def test_get_reject_mask(configured_filter):
    f = configured_filter
    assert f.get_reject_mask() is f.reject and f.reject is not None


def test_set_integration(populated_integration):
    f = KillFilter()
    assert f.reject is None
    f.set_integration(populated_integration)
    assert isinstance(f.reject, np.ndarray)
    assert f.reject.size == f.nf + 1
    assert np.allclose(f.reject, False)


def test_kill(configured_filter):
    f = configured_filter
    assert np.allclose(f.reject, False)
    assert not f.dft
    df = f.df
    i0 = 5
    i1 = 10
    from_f = i0 * df * units.Unit('Hz')
    to_f = i1 * df * units.Unit('Hz')
    frequency_range = Range(from_f, to_f)
    f.kill(frequency_range)
    assert np.allclose(f.reject[:i0], False)
    assert np.allclose(f.reject[i0:i1 + 1], True)
    assert np.allclose(f.reject[i1 + 1:], False)
    assert f.dft

    f.reject.fill(False)
    frequency_range = Range(to_f, from_f)
    f.kill(frequency_range)
    assert not f.reject.any()


def test_update_config(configured_filter):
    f = configured_filter
    f.update_config()
    assert np.allclose(f.reject[:5], 0)
    assert np.allclose(f.reject[5:11], 1)
    assert np.allclose(f.reject[11:15], 0)
    assert np.allclose(f.reject[15:17], 1)
    assert np.allclose(f.reject[17:], 0)

    f.reject.fill(False)
    f.integration.configuration.blacklist('filter.kill.bands')
    f.update_config()
    assert np.allclose(f.reject, False)


def test_auto_dft(nonzero_filter):
    f = nonzero_filter
    assert not f.dft
    f.auto_dft()
    assert f.dft
    f.reject.fill(True)
    f.auto_dft()
    assert not f.dft


def test_response_at(nonzero_filter):
    f = nonzero_filter
    f.reject[1] = True
    assert f.response_at(0) == 0
    assert f.response_at(f.reject.size) == 0
    f.reject[1] = False
    assert f.response_at(f.reject.size) == 1

    fch = np.asarray([0, 1, f.reject.size, 5, 6])
    response = f.response_at(fch)
    assert np.allclose(response, [1, 1, 1, 0, 0])


def test_count_parms(nonzero_filter):
    f = nonzero_filter
    assert f.count_parms() == 5
    f.reject = None
    assert f.count_parms() == 0


def test_get_id():
    f = KillFilter()
    assert f.get_id() == 'K'


def test_get_config_name():
    f = KillFilter()
    assert f.get_config_name() == 'filter.kill'
