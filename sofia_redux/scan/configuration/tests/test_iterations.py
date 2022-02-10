# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.configuration.iterations import IterationOptions


@pytest.fixture
def iteration_options(fits_configuration):
    """
    Return a test IterationOptions object.

    Parameters
    ----------
    fits_configuration : Configuration

    Returns
    -------
    IterationOptions
    """
    i = fits_configuration.iterations
    i.allow_error = True
    return i


def test_init():
    i = IterationOptions(allow_error=False, verbose=False)
    assert not i.allow_error and not i.verbose and i.max_iteration is None
    i = IterationOptions(allow_error=True, verbose=True, max_iteration=3)
    assert i.allow_error and i.verbose and i.max_iteration == 3


def test_copy(iteration_options):
    i = iteration_options
    i2 = i.copy()
    assert i.options == i2.options and i.options is not i2.options
    assert i.max_iteration == i2.max_iteration
    assert i.rounds_locked == i2.rounds_locked
    assert i.current_iteration == i2.current_iteration


def test_clear(iteration_options):
    i = iteration_options
    i.clear()
    assert i.max_iteration is None
    assert i.current_iteration is None
    assert i.size == 0


def test_get_item(iteration_options):
    i = iteration_options
    assert i[1] == {}
    assert i[2] == {'o1s1.v1': '10'}


def test_set_item(iteration_options):
    i = iteration_options
    current = i.options.copy()
    i.verbose = True
    with log.log_to_list() as log_list:
        i[3] = None
    assert 'Could not parse iteration' in log_list[0].msg
    assert i.options == current

    i[3] = {'o1s1.v1': '20'}
    assert i[3] == {'o1s1.v1': '20'}


def test_str(iteration_options):
    i = iteration_options
    expected = "Iteration configurations:"
    expected += '\nMaximum iterations: 5'
    expected += '\nIteration switches: 2, 0.6'
    assert str(i) == expected


def test_repr(iteration_options):
    i = iteration_options
    assert 'IterationOptions object' in repr(i)
    assert str(i) in repr(i)


def test_max_iteration(iteration_options):
    i = iteration_options
    assert i.max_iteration == 5
    i.max_iteration = 4
    assert i.max_iteration == 4
    i.lock_rounds()
    i.max_iteration = 3
    assert i.max_iteration == 4
    i.unlock_rounds()
    i.max_iteration = None
    assert i.max_iteration is None


def test_rounds_locked(iteration_options):
    i = iteration_options
    assert not i.rounds_locked
    i.lock_rounds()
    assert i.rounds_locked
    i.unlock_rounds()
    assert not i.rounds_locked


def test_lock_rounds(iteration_options):
    i = iteration_options
    i.lock_rounds(1)
    assert i.rounds_locked
    assert i.max_iteration == 1
    i.lock_rounds(2)
    assert i.rounds_locked
    assert i.max_iteration == 1
    i.unlock_rounds()
    i.max_iteration = 5
    i.lock_rounds()
    assert i.max_iteration == 5
    assert i.rounds_locked


def test_unlock_rounds(iteration_options):
    i = iteration_options
    i.lock_rounds()
    assert i.rounds_locked
    i.unlock_rounds()
    assert not i.rounds_locked


def test_parse_iteration():
    i = IterationOptions()
    i.allow_error = True
    i.max_iteration = 10
    assert i.parse_iteration('30%') == 0.3
    assert i.parse_iteration('0.25') == 0.25
    assert i.parse_iteration('last') == -1
    assert i.parse_iteration('final') == -1
    assert i.parse_iteration('first') == 1
    assert i.parse_iteration('2') == 2
    with log.log_to_list() as log_list:
        assert i.parse_iteration('a') is None
    assert "Could not parse iteration string" in log_list[0].msg

    with log.log_to_list() as log_list:
        assert i.parse_iteration(None) is None
    assert 'iteration must be' in log_list[0].msg

    assert i.parse_iteration(0.1) == 0.1
    with log.log_to_list() as log_list:
        assert i.parse_iteration(2.5) is None
    assert "Fractional iterations must be" in log_list[0].msg


def test_get():
    i = IterationOptions()
    i.max_iteration = 10

    i[5] = {'add': 'foo'}
    i[0.5] = {'add': 'bar'}
    i[2] = {'value': 'test_value'}
    i[3] = {'sub_options': {'value1': '1'}}
    i[0.3] = {'sub_options': {'value2': '2'}}
    i[7] = {'forget': 'bar'}

    assert i.get(1) == {}
    assert i.get(1, default={'key': 'value'}) == {'key': 'value'}

    assert i.get(2) == {'value': 'test_value'}
    assert i.get(3) == {'sub_options': {'value1': '1', 'value2': '2'}}

    assert i.get(5) == {'add': ['foo', 'bar']}
    assert i.get(7) == {'forget': 'bar'}


def test_set():
    i = IterationOptions()
    i.allow_error = True
    i.max_iteration = 10
    i.set('a', {'add': 'foo'})
    assert i.size == 0
    i.set(1, {'add': 'foo'})
    assert i[1] == {'add': 'foo'}
    i.set(-1, {'add': 'bar'})
    assert i[10] == {'add': 'bar'}
    i.set(1, {'add': 'baz'})
    assert i[1] == {'add': ['foo', 'baz']}


def test_relative_iteration():
    i = IterationOptions()
    i.allow_error = True
    i.max_iteration = None
    assert i.relative_iteration(0.1) is None
    assert i.relative_iteration(-1) is None
    assert i.relative_iteration(2) == 2
    i.max_iteration = 10
    assert i.relative_iteration(0.1) == 1
    assert i.relative_iteration(-1) == 10
    assert i.relative_iteration(2) == 2
    assert i.relative_iteration('a') is None


def test_update():
    i = IterationOptions()
    i.allow_error = True
    i.update({})
    assert i.size == 0
    with log.log_to_list() as log_list:
        i.update({'iteration': None})
    assert 'could not be parsed' in log_list[0].msg
    options = {1: {'add': 'foo'},
               0.5: {'key': 'value'}}
    options = {'iteration': options}
    i.update(options)
    assert i.options['1'] == {'add': 'foo'}
    assert i.options['0.5'] == {'key': 'value'}


def test_set_iteration():
    i = IterationOptions()
    i.allow_error = True
    i.max_iteration = 10
    i[1] = {'add': 'foo'}
    c = Configuration()
    i.set_iteration(c, 1)
    assert c.get_bool('foo')
    c.clear()
    assert c.size == 1
    i.set_iteration(c, 'a')
    assert c.size == 1  # no change
