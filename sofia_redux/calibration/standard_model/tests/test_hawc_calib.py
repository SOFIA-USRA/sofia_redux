# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.calibration.standard_model import hawc_calib


def test_setup_temperatures():
    temps = hawc_calib.setup_temperatures()
    assert len(temps) == 5
    assert temps['instrument'] == 10
    assert temps['telescope'] == 240


def test_main():
    assert 1
