# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.utilities.func import date2seconds


def test_bad_time():
    assert date2seconds("invalid date string") is None


def test_standard():
    assert isinstance(date2seconds("2000-01-01T00:00:00.0"), float)
    assert isinstance(date2seconds("2000-01-01T00:00:00"), float)


def test_format():
    assert isinstance(date2seconds("2000_01_01", dformat='%Y_%m_%d'), float)
