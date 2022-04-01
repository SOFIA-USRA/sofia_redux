# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.configuration.fits import FitsOptions
from sofia_redux.scan.info.base import InfoBase
from sofia_redux.scan.utilities.bracketed_values import BracketedValues


class TestInfoBase(object):
    def test_init(self):
        info = InfoBase()
        assert info.configuration is None
        assert info.scan is None
        assert not info.scan_applied
        assert not info.configuration_applied

        assert info.referenced_attributes == {'configuration', 'scan'}
        assert info.log_id == 'base'
        assert info.log_prefix == 'base.'

    def test_copy(self):
        info = InfoBase()

        # set placeholder objects to test referenced attribute
        info.scan = np.arange(10)
        info.scan_applied = True
        info.test = np.arange(10)

        new = info.copy()
        assert new.scan is info.scan
        assert new.scan_applied is True
        assert new.test is not info.test
        assert np.all(new.test == info.test)

    def test_options(self):
        info = InfoBase()
        assert info.options is None

        info.configuration = Configuration()
        assert info.options is None

        fits = FitsOptions()
        info.configuration.fits = fits
        assert info.options is None

        # returns fits options, only if enabled
        info.configuration.fits.enabled = True
        assert info.options is fits

    def test_set_configuration(self, mocker):
        info = InfoBase()
        m1 = mocker.patch.object(info, 'validate')

        with pytest.raises(ValueError) as err:
            info.set_configuration({})
        assert "configuration must be a <" in str(err)

        config = Configuration()
        info.set_configuration(config)
        assert info.configuration is config
        assert info.configuration_applied
        assert m1.call_count == 0

        info.scan_applied = True
        info.set_configuration(config)
        assert info.configuration is config
        assert info.configuration_applied
        assert m1.call_count == 1

    def test_set_scan(self, mocker):
        info = InfoBase()
        m1 = mocker.patch.object(info, 'validate')

        scan = [1, 2, 3]
        info.set_scan(scan)
        assert info.scan is scan
        assert info.scan_applied
        assert m1.call_count == 0

        info.configuration_applied = True
        info.set_scan(scan)
        assert info.scan is scan
        assert info.scan_applied
        assert m1.call_count == 1

    def test_str(self):
        info = InfoBase()
        assert str(info) == ('configuration: None\n'
                             'scan_applied: False\n'
                             'scan: None\n'
                             'configuration_applied: False\n')

    def test_merge(self):
        info = InfoBase()
        info.test = BracketedValues(start=1, end=3)

        last = InfoBase()
        last.test = BracketedValues(start=2, end=4)

        info.merge(last)
        assert info.test.start == 1
        assert info.test.end == 4

    def test_valid_header_value(self):
        assert not InfoBase.valid_header_value(None)

        assert InfoBase.valid_header_value(True)
        assert InfoBase.valid_header_value(False)

        assert InfoBase.valid_header_value(9999)
        assert not InfoBase.valid_header_value(-9999)

        assert InfoBase.valid_header_value(9999.0)
        assert not InfoBase.valid_header_value(-9999.0)
        assert not InfoBase.valid_header_value(np.nan)
        assert not InfoBase.valid_header_value(np.inf)

        assert InfoBase.valid_header_value('unknown')
        assert InfoBase.valid_header_value('')
        assert not InfoBase.valid_header_value('UNKNOWN')

        assert InfoBase.valid_header_value(9999.0 * units.arcsec)
        assert not InfoBase.valid_header_value(-9999.0 * units.arcsec)
        assert not InfoBase.valid_header_value(np.nan * units.arcsec)
        assert not InfoBase.valid_header_value(np.inf * units.arcsec)

        assert not InfoBase.valid_header_value([1, 2, 3])

    def test_get_table_entry(self):
        info = InfoBase()
        assert info.get_table_entry('scan_applied') is False
        assert info.get_table_entry('scan') is None
        assert info.get_table_entry('test') is None
