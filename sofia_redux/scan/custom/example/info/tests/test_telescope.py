# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.custom.example.info.telescope import ExampleTelescopeInfo


def test_init():
    info = ExampleTelescopeInfo()
    assert info.telescope == 'Example Telescope'


def test_get_telescope_name():
    assert ExampleTelescopeInfo.get_telescope_name() == 'EGTEL'
