# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import regex

from sofia_redux.scan.utilities.bracketed_values import BracketedValues


def test_initialize():
    # Testing default values
    bracket = BracketedValues()
    assert np.isnan(bracket.start)
    assert np.isnan(bracket.end)
    assert bracket.unit is None

    bracket = BracketedValues(start=-1.0, end=1.0)
    assert bracket.start == -1
    assert bracket.end == 1
    assert bracket.unit is None

    bracket = BracketedValues(start=-1.0, end=1.0, unit='second')
    assert bracket.start == -1 * units.Unit('second')
    assert bracket.end == 1 * units.Unit('second')
    assert bracket.unit == units.Unit('second')

    # Check unit conversion
    bracket = BracketedValues(start=-1.0, end=1.0 * units.Unit('minute'),
                              unit='second')
    assert bracket.start.value == -1
    assert bracket.end.value == 60
    assert bracket.unit == units.Unit('second')

    # Check start value units propagate
    bracket = BracketedValues(start=-1.0 * units.Unit('second'), end=1.0)
    assert bracket.unit == units.Unit('second')
    assert bracket.start.value == -1.0
    assert bracket.start.unit == units.Unit('second')
    assert bracket.end.unit == units.Unit('second')
    assert bracket.end.value == 1.0

    # Check start value units propagate
    bracket = BracketedValues(start=-1.0, end=1.0 * units.Unit('second'))
    assert bracket.unit == units.Unit('second')
    assert bracket.start.value == -1.0
    assert bracket.start.unit == units.Unit('second')
    assert bracket.end.unit == units.Unit('second')
    assert bracket.end.value == 1.0


def test_str():
    bracket = BracketedValues(start=5, end=25, unit='meter')
    result = str(bracket)
    assert result == '(5.0 --> 25.0 m)'

    bracket = BracketedValues(5, 25)
    assert str(bracket) == '(5 --> 25)'


def test_repr():
    bracket = BracketedValues(start=1, end=2)
    assert regex.search(r'.*BracketedValues object at.*\(1 --> 2\)',
                        repr(bracket)) is not None


def test_midpoint():
    bracket = BracketedValues(start=1, end=2, unit='second')
    midpoint = bracket.midpoint
    assert midpoint == 1.5 * units.Unit('second')

    bracket = BracketedValues(start=1, unit='second')
    assert bracket.midpoint == 1 * units.Unit('second')

    bracket = BracketedValues(end=2, unit='second')
    assert bracket.midpoint == 2 * units.Unit('second')

    bracket = BracketedValues()
    assert np.isnan(bracket.midpoint)


def test_copy():
    bracket1 = BracketedValues(start=1, end=2, unit='second')
    bracket2 = bracket1.copy()
    assert bracket1.start == bracket2.start
    assert bracket1.end == bracket2.end
    assert bracket1.unit == bracket2.unit
    assert bracket1 is not bracket2


def test_merge():
    bracket = BracketedValues(start=1, end=2)
    bracket.merge(BracketedValues(start=2, end=3))
    assert bracket.start == 1
    assert bracket.end == 3
    bracket.merge(BracketedValues(start=0, end=2))
    assert bracket.start == 0
    assert bracket.end == 3
    bracket.merge(BracketedValues(start=-1, end=4))
    assert bracket.start == -1
    assert bracket.end == 4

    bracket = BracketedValues(start=1, end=2, unit='minute')
    bracket.merge(BracketedValues(start=0, end=300, unit='second'))
    assert bracket.start.value == 0
    assert bracket.start.unit == units.Unit('minute')
    assert bracket.end.value == 5
    assert bracket.end.unit == units.Unit('minute')
