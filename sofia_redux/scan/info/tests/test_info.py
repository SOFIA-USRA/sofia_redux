# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import units
from astropy.io import fits
from configobj import ConfigObj
import numpy as np
import pytest

from sofia_redux.scan.custom.example.channels.channels import ExampleChannels
from sofia_redux.scan.custom.example.channels.channel_data.channel_data \
    import ExampleChannelData
from sofia_redux.scan.custom.example.channels.channel_group.channel_group \
    import ExampleChannelGroup
from sofia_redux.scan.custom.example.scan.scan import ExampleScan
from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.example.info.info import ExampleInfo
from sofia_redux.scan.info.info import Info
from sofia_redux.scan.reduction.reduction import Reduction
from sofia_redux.scan.source_models.astro_intensity_map \
    import AstroIntensityMap
from sofia_redux.scan.source_models.beams.instant_focus import InstantFocus
from sofia_redux.scan.source_models.sky_dip import SkyDip
from sofia_redux.scan.source_models.spectral_cube import SpectralCube
from sofia_redux.scan.custom.fifi_ls.simulation.simulation import \
    FifiLsSimulation


class TestInfo(object):
    def test_init(self):
        info = Info()
        assert info.name is None
        assert info.scan is None
        assert info.parent is None
        assert isinstance(info.configuration, Configuration)

        assert info.referenced_attributes == {'configuration', 'scan',
                                              'parent'}

    def test_set_parent(self):
        info = Info()
        info.set_parent('test')
        assert info.parent == 'test'

    def test_copy(self):
        info = Info()
        info.name = 'test'

        # set some placeholder objects to test referenced attributes
        info.parent = np.arange(10)
        info.scan = np.arange(10)
        info.parallelism = np.arange(10)

        new = info.copy()
        assert new is not info
        assert new.name == info.name
        assert new.parent is info.parent
        assert new.scan is info.scan
        assert new.parallelism is not info.parallelism
        assert np.allclose(new.parallelism, info.parallelism)

        # config and sub-info configs are referenced
        assert new.configuration is info.configuration
        infos = ['instrument', 'astrometry', 'observation',
                 'origin', 'telescope']
        for subinfo in infos:
            assert (getattr(new, subinfo).configuration
                    is info.configuration)

        # test dereference
        new.unlink_scan()
        assert new.scan is not info.scan
        assert np.allclose(new.scan, info.scan)

        new.unlink_configuration()
        assert new.configuration is not info.configuration
        for subinfo in infos:
            assert (getattr(new, subinfo).configuration
                    is not info.configuration)

    def test_config_path(self):
        info = Info()
        info.name = None
        assert info.config_path == info.configuration.config_path
        assert os.path.isdir(info.config_path)

        info.name = 'example'
        assert info.config_path == os.path.join(info.configuration.config_path,
                                                'example')
        assert os.path.isdir(info.config_path)

    def test_instance_from_intrument_name(self):
        info = Info.instance_from_instrument_name('example')
        assert isinstance(info, ExampleInfo)

    def test_properties(self):
        info = Info()

        # getters only
        assert info.size_unit == units.arcsec
        assert np.isnan(info.frequency)
        assert info.telescope_name == 'UNKNOWN'
        assert info.jansky_per_beam == 1 * units.Jy / units.beam
        assert info.data_unit == units.count
        assert np.isnan(info.kelvin)
        assert np.isnan(info.point_size)
        assert np.isnan(info.source_size)

        # getters and setters
        assert np.isnan(info.integration_time)
        info.integration_time = 10 * units.s
        assert info.integration_time == 10 * units.s
        with pytest.raises(ValueError) as err:
            info.integration_time = 20
        assert 'time must be <' in str(err)

        assert np.isnan(info.resolution)
        info.resolution = 10 * units.arcsec
        assert info.resolution == 10 * units.arcsec
        info.resolution = 20
        assert info.resolution == 20 * units.arcsec

        assert np.isnan(info.sampling_interval)
        info.sampling_interval = 1 * units.s
        assert info.sampling_interval == 1 * units.s
        with pytest.raises(ValueError) as err:
            info.sampling_interval = 2
        assert 'interval must be <' in str(err)

        assert info.gain == 1
        info.gain = 10
        assert info.gain == 10

    def test_get_class(self):
        info = Info()
        info.instrument.name = 'example'
        assert info.get_channel_class() is ExampleChannels
        assert info.get_channel_data_class() is ExampleChannelData
        assert info.get_channel_group_class() is ExampleChannelGroup
        assert info.get_scan_class() is ExampleScan
        assert isinstance(info.get_channels_instance(), ExampleChannels)

    def test_get_source_model_instance(self):
        info = Info()
        config = info.configuration

        # no source.type in config
        del config['source.type']
        assert info.get_source_model_instance([]) is None

        # set known types
        config.set_option('source.type', 'skydip')
        assert isinstance(info.get_source_model_instance([]), SkyDip)

        config.set_option('source.type', 'map')
        assert isinstance(info.get_source_model_instance([]),
                          AstroIntensityMap)

        fifi_sim = FifiLsSimulation()
        fifi_info = fifi_sim.info
        fifi_info.configuration.set_option('source.type', 'cube')
        assert isinstance(fifi_info.get_source_model_instance([]),
                          SpectralCube)

        # set null or unknown types
        config.set_option('source.type', 'null')
        assert info.get_source_model_instance([]) is None
        config.set_option('source.type', 'bad')
        assert info.get_source_model_instance([]) is None

    def test_validate_configuration_registration(self, mocker):
        info = Info()

        # default register does nothing
        info.validate_configuration_registration()

        # for subclass purposes, register should be called for
        # each config file
        m1 = mocker.patch.object(info, 'register_config_file')
        info.configuration.config_files = [1, 2, 3]
        info.validate_configuration_registration()
        assert m1.call_count == 3

    def test_set_options(self):
        info = Info()
        config = info.configuration.copy()

        info.set_date_options(12345)
        assert info.configuration.dates.current_date == 12345

        info.set_mjd_options(54321)
        assert info.configuration.dates.current_date == 54321

        # no change for unconfigured serial branch
        info.set_serial_options(12345)
        assert info.configuration.options == config.options

        # no change for unconfigured object
        info.set_object_options('Mars')
        assert info.configuration.options == config.options

        # adds fits branch from header
        info.parse_header(fits.Header({'test': 1}))
        assert info.configuration.options != config.options
        assert info.configuration.options['fits'] == ConfigObj({'TEST': '1'})

    def test_read_configuration(self):
        info = Info()
        config = info.configuration.copy()
        assert len(config.options) < 10

        # reads default.cfg by default
        info.read_configuration()
        assert info.configuration.options != config.options
        assert len(info.configuration.config_files) == 1
        assert 'default.cfg' in info.configuration.config_files[0]
        assert len(info.configuration.options) > 10

    def test_apply_configuration(self):
        info = Info()
        info.configuration = 'bad'
        with pytest.raises(ValueError) as err:
            info.apply_configuration()
        assert 'Configuration must be a <' in str(err)

        info.name = 'test'
        info.configuration = Configuration()
        info.configuration.set_option('test', True)
        info.apply_configuration()
        assert info.configuration.instrument_name == 'test'

        infos = ['instrument', 'astrometry', 'observation',
                 'origin', 'telescope']
        for subinfo in infos:
            assert getattr(info, subinfo).configuration.has_option('test')

    @pytest.mark.parametrize('func, args',
                             [('validate', []),
                              ('validate_scan', ['test']),
                              ('parse_image_header', ['test']),
                              ('edit_image_header', ['test']),
                              ('edit_scan_header', ['test'])])
    def test_subinfo_functions(self, mocker, func, args):
        info = Info()

        mocks = []
        for subinfo in info.available_info.values():
            mocks.append(mocker.patch.object(subinfo, func))

        # calls function for all subinfos, does nothing else
        info_func = getattr(info, func)
        info_func(*args)
        for mock in mocks:
            assert mock.call_count == 1

    def test_validate_scans(self, mocker, capsys, populated_integration):
        info = Info()

        # no op if no scans
        info.validate_scans(None)
        info.validate_scans([None])

        # does not actually validate scans, just checks to see
        # if they are valid
        scan = ExampleScan(ExampleChannels())
        info.validate_scans([scan, None])
        assert not scan.is_valid()

        # if a valid scan and size > 0,
        # then will check for jackknife configuration
        mocker.patch.object(scan, 'is_valid', return_value=True)
        scan.integrations = [populated_integration.copy(),
                             populated_integration.copy()]
        assert scan.size == 2
        scan2 = scan.copy()
        scan2.integrations = [populated_integration.copy(),
                              populated_integration.copy()]
        assert scan2.size == 2
        scan3 = scan.copy()
        scan3.integrations = []
        assert scan3.size == 0

        info.configuration.set_option('jackknife.alternate', True)
        scans = [scan, scan2, scan3]
        info.validate_scans(scans)
        assert 'JACKKNIFE: Alternating scans' in capsys.readouterr().out

        # every other scan has gain inverted
        assert np.allclose(scans[0].integrations[0].gain, -1)
        assert np.allclose(scans[0].integrations[1].gain, -1)
        assert np.allclose(scans[1].integrations[0].gain, 1)
        assert np.allclose(scans[1].integrations[1].gain, 1)

        # zero size skipped
        assert len(scans[2].integrations) == 0

    def test_get_focus_string(self):
        msg = Info.get_focus_string(None)
        assert msg == ' No instant focus.'

        focus = InstantFocus()
        focus.x = 1 * units.cm
        msg = Info.get_focus_string(focus)
        assert msg == '\n  Focus.dX --> 10.0 mm'

        focus.y = 2 * units.cm
        msg = Info.get_focus_string(focus)
        assert msg == '\n  Focus.dX --> 10.0 mm\n' \
                      '  Focus.dY --> 20.0 mm'

        focus.z = 3 * units.cm
        msg = Info.get_focus_string(focus)
        assert msg == '\n  Focus.dX --> 10.0 mm\n' \
                      '  Focus.dY --> 20.0 mm\n' \
                      '  Focus.dZ --> 30.0 mm'

    def test_get_set_name(self):
        info = Info()
        assert info.get_name() == ''
        info.set_name('test')
        assert info.get_name() == 'test'

    def test_set_outpath(self, mocker):
        # config function called
        info = Info()
        m1 = mocker.patch.object(info.configuration, 'set_outpath')
        info.set_outpath()
        assert m1.call_count == 1

    def test_perform_reduction(self, mocker):
        info = Info()
        reduction = Reduction('example')
        m1 = mocker.patch.object(reduction, 'read_scans')
        m2 = mocker.patch.object(reduction, 'validate')
        m3 = mocker.patch.object(reduction, 'reduce')

        info.perform_reduction(reduction, ['test.fits'])
        assert m1.call_count == 1
        assert m2.call_count == 1
        assert m3.call_count == 1
