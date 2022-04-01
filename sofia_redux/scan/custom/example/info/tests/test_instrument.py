# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units

from sofia_redux.scan.custom.example.info.instrument import \
    ExampleInstrumentInfo


def test_init():
    info = ExampleInstrumentInfo()
    assert info.name == 'example'
    assert info.mount.name == 'CASSEGRAIN'
    assert info.resolution == 10 * units.Unit('arcsec')
    assert info.sampling_interval == 0.1 * units.Unit('second')
    assert info.integration_time == info.sampling_interval
