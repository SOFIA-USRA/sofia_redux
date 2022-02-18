# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.epoch.epoch import B1950
from sofia_redux.scan.coordinate_systems.offset_2d import Offset2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates \
    import SphericalCoordinates
from sofia_redux.scan.custom.example.channels.channels import ExampleChannels
from sofia_redux.scan.custom.example.info.info import ExampleInfo
from sofia_redux.scan.custom.example.scan.scan import ExampleScan
from sofia_redux.scan.flags.mounts import Mount
from sofia_redux.scan.scan.scan import Scan
from sofia_redux.scan.source_models.beams.asymmetry_2d import Asymmetry2D
from sofia_redux.scan.source_models.beams.gaussian_source \
    import GaussianSource
from sofia_redux.toolkit.utilities.fits import set_log_level


# make a test class to implement abstract method but not
# override everything else
class DemoScanClass(Scan):
    def read(self, filename, read_fully=True):
        pass


class TestScan(object):

    def test_init(self):
        # abstract class, can't be instantiated directly
        with pytest.raises(TypeError):
            Scan(None)

        channels = ExampleChannels()
        scan = ExampleScan(channels)

        # channels is set as copy
        assert scan.channels is not channels
        assert isinstance(scan.channels, ExampleChannels)
        assert len(scan) == 0
        assert scan.serial == -1

    @pytest.mark.parametrize('prop', ['info', 'astrometry', 'frame_flagspace',
                                      'configuration', 'channel_flagspace',
                                      'mjd', 'lst', 'source_name',
                                      'equatorial', 'horizontal', 'site',
                                      'apparent'])
    def test_blank_properties_none(self, prop):
        scan = ExampleScan(None)
        assert getattr(scan, prop) is None

        # some have no-op setters, some don't
        try:
            setattr(scan, prop, 'test')
        except AttributeError:
            pass
        assert getattr(scan, prop) is None

    @pytest.mark.parametrize('prop', ['is_nonsidereal', 'is_tracking'])
    def test_blank_is_properties(self, prop):
        scan = ExampleScan(None)
        assert getattr(scan, prop) is False

        # some have no-op setters, some don't
        try:
            setattr(scan, prop, True)
        except AttributeError:
            pass
        assert getattr(scan, prop) is False

    @pytest.mark.parametrize('func', ['have_equatorial', 'have_horizontal',
                                      'have_site', 'have_apparent',
                                      'is_valid', 'have_valid_integrations'])
    def test_blank_bool_functions(self, func):
        scan = ExampleScan(None)
        assert getattr(scan, func)() is False

    @pytest.mark.parametrize('func', ['get_first_integration',
                                      'get_last_integration'])
    def test_blank_none_functions(self, func):
        scan = ExampleScan(None)
        assert getattr(scan, func)() is None

    @pytest.mark.parametrize('func', ['get_observing_time',
                                      'get_frame_count',
                                      'get_source_generation'])
    def test_blank_zero_functions(self, func):
        scan = ExampleScan(None)
        assert getattr(scan, func)() == 0

    def test_class_from_instrument_name(self):
        assert Scan.class_from_instrument_name('example') is ExampleScan
        with pytest.raises(ModuleNotFoundError):
            Scan.class_from_instrument_name('test')

    def test_has_option(self, initialized_scan):
        scan = ExampleScan(None)
        assert scan.has_option('test') is False

        scan = initialized_scan
        assert not scan.has_option('test')
        scan.configuration.set_option('test', True)
        assert scan.has_option('test')

    def test_set_integration(self, initialized_scan):
        initialized_scan.integrations = [0, 2, 3]
        # getter/setter
        assert initialized_scan[1] == 2
        initialized_scan[1] = 1
        assert initialized_scan.integrations == [0, 1, 3]

    def test_initialized_properties(self, initialized_scan):
        flags = initialized_scan.channel_flagspace
        assert flags is initialized_scan.channels.flagspace

        assert initialized_scan.mjd is not None
        initialized_scan.mjd = 12345
        assert initialized_scan.mjd == 12345
        assert initialized_scan.astrometry.mjd == 12345

        initialized_scan.source_name = 'test'
        assert initialized_scan.source_name == 'test'
        assert initialized_scan.info.observation.source_name == 'test'

        initialized_scan.equatorial = 'test'
        assert initialized_scan.equatorial == 'test'
        assert initialized_scan.astrometry.equatorial == 'test'

        initialized_scan.horizontal = 'test'
        assert initialized_scan.horizontal == 'test'
        assert initialized_scan.astrometry.horizontal == 'test'

        initialized_scan.site = 'test'
        assert initialized_scan.site == 'test'
        assert initialized_scan.astrometry.site == 'test'

        initialized_scan.site = 'test'
        assert initialized_scan.site == 'test'
        assert initialized_scan.astrometry.site == 'test'

        initialized_scan.apparent = 'test'
        assert initialized_scan.apparent == 'test'
        assert initialized_scan.astrometry.apparent == 'test'

        initialized_scan.serial = 1234
        assert initialized_scan.serial == 1234

    def test_validate(self, mocker, capsys, initialized_scan, populated_scan):
        initialized_scan.validate()
        assert not initialized_scan.have_valid_integrations()

        populated_scan.validate()
        assert populated_scan.have_valid_integrations()
        assert 'Processing integration 1' in capsys.readouterr().out

        # mock invalid integration
        integ = populated_scan.integrations[0]
        integ.is_valid = False
        mocker.patch.object(integ, 'get_frame_count', return_value=0)
        mocker.patch.object(integ, 'validate')
        populated_scan.validate()
        assert not populated_scan.have_valid_integrations()
        assert 'No valid integrations' in capsys.readouterr().err

        # remove the invalid scan
        assert len(populated_scan) == 1
        populated_scan.validate_integrations()
        assert len(populated_scan) == 0

    def test_validate_lst(self, populated_scan):
        populated_scan.lst = np.nan
        assert np.isnan(populated_scan.lst)
        populated_scan.validate()
        assert not np.isnan(populated_scan.lst)

        populated_scan.lst = np.nan
        f1 = populated_scan.integrations[0].get_first_frame()
        f2 = populated_scan.integrations[-1].get_last_frame()
        assert f1.lst != f2.lst
        populated_scan.validate()
        assert populated_scan.lst == 0.5 * (f1.lst + f2.lst)

    def test_validate_merge_segment(self, populated_scan, mocker, capsys):
        # merge - no op for single integration
        populated_scan.configuration.set_option('subscan.merge', True)
        with set_log_level('DEBUG'):
            populated_scan.validate()
        assert populated_scan.have_valid_integrations()
        assert len(populated_scan) == 1
        assert 'Total exposure time: 110.0 s' in capsys.readouterr().out

        # segment - more integrations
        populated_scan.configuration.set_option('segment', 50)
        with set_log_level('DEBUG'):
            populated_scan.validate()
        assert populated_scan.have_valid_integrations()
        assert len(populated_scan) == 3
        capt = capsys.readouterr()
        assert 'Processing integration 1' in capt.out
        assert 'Processing integration 2' in capt.out
        assert 'Processing integration 3' in capt.out

        # revalidate with merge to stick subscans back together
        del populated_scan.configuration['segment']
        populated_scan.configuration.set_option('subscan.merge', True)
        with set_log_level('DEBUG'):
            populated_scan.validate()
        assert len(populated_scan) == 1
        assert 'Merging 3 integrations' in capsys.readouterr().out
        assert 'Integration 1: 110.0 s'

    def test_validate_equatorial(self, populated_scan):
        # check that equatorial can be calculated from horizontal
        populated_scan.calculate_apparent()
        populated_scan.calculate_horizontal()
        populated_scan.astrometry.equatorial = None

        populated_scan.validate()
        assert populated_scan.horizontal is not None
        assert populated_scan.equatorial is not None

    def test_validate_horizontal(self, populated_scan):
        # check that horizontal can be calculated from equatorial
        populated_scan.astrometry.horizontal = None
        populated_scan.validate()

        assert populated_scan.horizontal is not None
        assert populated_scan.equatorial is not None

    def test_validate_options(self, capsys, populated_scan):
        # jackknife modifies name
        populated_scan.configuration.set_option('jackknife', True)
        assert populated_scan.source_name == 'Simulation'
        populated_scan.validate()
        assert populated_scan.source_name == 'Simulation-JK'
        del populated_scan.configuration['jackknife']

        # pointing sets correction
        populated_scan.configuration.set_option('pointing', [1, 2])
        populated_scan.validate()
        assert 'Adjusting pointing by ' \
               'x=1.0 arcsec y=2.0 arcsec' in capsys.readouterr().out

    def test_short_date(self, populated_scan):
        assert populated_scan.get_short_date_string() == '2021-12-06'

        populated_scan.mjd = 12345
        assert populated_scan.get_short_date_string() == '1892-09-04'

    @pytest.mark.parametrize('options', [{'value': [0, 0]}, {},
                                         [0, 0], ['0', '0'], 'bad value'])
    def test_get_zero_pointing_correction(self, initialized_scan, options):
        corr = initialized_scan.get_pointing_correction(options)
        assert corr.x == 0
        assert corr.y == 0

    @pytest.mark.parametrize('options', [{'value': [1, 2], 'offset': [3, 4]},
                                         {'offset': [4, 6]},
                                         {'value': [4, 6]}, [4, 6]])
    def test_get_nonzero_pointing(self, initialized_scan, options):
        corr = initialized_scan.get_pointing_correction(options)
        assert corr.x == 4.0 * units.arcsec
        assert corr.y == 6.0 * units.arcsec

    def test_pointing_at(self, populated_scan):
        assert populated_scan.pointing_correction.x is None
        assert populated_scan.pointing_correction.y is None

        # no op if not a coordinate
        options = [4, 6]
        populated_scan.pointing_at(options)
        assert populated_scan.pointing_correction.x is None
        assert populated_scan.pointing_correction.y is None

        # make a coordinate
        options = populated_scan.get_pointing_correction(options)

        # replaced if None
        populated_scan.pointing_at(options)
        assert populated_scan.pointing_correction.x == 4.0 * units.arcsec
        assert populated_scan.pointing_correction.y == 6.0 * units.arcsec

        # added if not None
        populated_scan.pointing_at(options)
        assert populated_scan.pointing_correction.x == 8.0 * units.arcsec
        assert populated_scan.pointing_correction.y == 12.0 * units.arcsec

        # no op if no integrations
        populated_scan.integrations = None
        populated_scan.pointing_at(options)
        assert populated_scan.pointing_correction.x == 8.0 * units.arcsec
        assert populated_scan.pointing_correction.y == 12.0 * units.arcsec

    def test_merge_integrations(self, populated_scan, capsys):
        integ1 = populated_scan.integrations[0].copy()
        integ2 = integ1.copy()
        # add a gap between integrations
        integ2.frames.mjd += 0.0015
        populated_scan.integrations = [integ1, integ2]
        assert len(populated_scan) == 2

        # merge: no op without config
        populated_scan.merge_integrations()
        assert len(populated_scan) == 2

        # after config: merges integrations to 1, with gap padding
        populated_scan.configuration.set_option('subscan.merge', True)
        with set_log_level('DEBUG'):
            populated_scan.merge_integrations()
        assert len(populated_scan) == 1
        capt = capsys.readouterr()
        assert 'Merging 2 integrations' in capt.out
        assert 'Padding with 196 frames' in capt.out

        # set a larger gap than tolerated: not merged
        populated_scan.configuration.set_option('subscan.merge.maxgap', 5000)
        integ1 = populated_scan.integrations[0].copy()
        integ2 = integ1.copy()
        integ2.frames.mjd += 0.1
        populated_scan.integrations = [integ1, integ2]
        with set_log_level('DEBUG'):
            populated_scan.merge_integrations()
        assert len(populated_scan) == 2
        capt = capsys.readouterr()
        assert 'Large gap before integration 1' in capt.out

    def test_get_summary_hdu(self, populated_scan):
        # 1 integration, no details
        populated_scan.configuration.set_option(
            'write.scandata.details', False)
        result = populated_scan.get_summary_hdu()
        assert isinstance(result, fits.BinTableHDU)
        assert len(result.data) == 1
        assert len(result.data.columns) == 7

        # additional integration added as another row
        integ1 = populated_scan.integrations[0]
        integ2 = integ1.copy()
        populated_scan.integrations = [integ1, integ2]
        result = populated_scan.get_summary_hdu()
        assert isinstance(result, fits.BinTableHDU)
        assert len(result.data) == 2
        assert len(result.data.columns) == 7

        # get extra details
        populated_scan.configuration.set_option(
            'write.scandata.details', True)
        result = populated_scan.get_summary_hdu()
        assert isinstance(result, fits.BinTableHDU)
        assert len(result.data) == 242
        assert len(result.data.columns) == 14

    def test_edit_scan_header(self, populated_scan):
        header = fits.Header()
        populated_scan.edit_scan_header(header)

        # check for expected simulation values
        assert header['EXTNAME'] == 'Scan-Simulation.1'
        assert header['INSTRUME'] == 'example'
        assert header['SCANID'] == 'Simulation.1'
        assert header['DATE-OBS'] == '2021-12-06T18:48:25.876'
        assert header['OBJECT'] == 'Simulation'
        assert header['RADESYS'] == 'FK5'
        assert header['RA'] == '17.76'
        assert header['DEC'] == '-29.0'
        assert header['EQUINOX'] == 'J2000.000'
        assert np.allclose(header['MJD'], 59554.783, atol=1e-3)
        assert np.allclose(header['LST'], 15.721, atol=1e-3)
        assert header['SITELON'] == -122.1
        assert header['SITELAT'] == 37.4
        assert header['WEIGHT'] == 1.0
        assert header['TRACKIN'] is False

        # add some more non-default info
        populated_scan.serial_number = 1
        populated_scan.info.origin.descriptor = 'descriptor'
        populated_scan.info.origin.observer = 'observer'
        populated_scan.info.observation.project = 'project'
        populated_scan.info.origin.creator = 'creator'
        populated_scan.edit_scan_header(header)
        assert header['SCANNO'] == 1
        assert header['SCANSPEC'] == 'descriptor'
        assert header['OBSERVER'] == 'observer'
        assert header['PROJECT'] == 'project'
        assert header['CREATOR'] == 'creator'

        # validate to add horizontal coords
        populated_scan.validate()
        assert 'AZ' not in header
        assert 'EL' not in header
        populated_scan.validate()
        populated_scan.edit_scan_header(header)
        assert np.allclose(header['AZ'], 151.864, atol=1e-3)
        assert np.allclose(header['EL'], 17.525, atol=1e-3)

    def test_edit_pointing_header(self, mocker, populated_scan):
        header = fits.Header()

        # no pointing: no op
        populated_scan.edit_pointing_header_info(header)
        assert 'PNT_DX' not in header
        assert 'PNT_DY' not in header

        # mock the pointing info
        populated_scan.pointing = GaussianSource()
        peak = SphericalCoordinates()
        peak.x = 1 * units.arcsec
        peak.y = 2 * units.arcsec

        m1 = mocker.patch.object(populated_scan,
                                 'get_native_pointing_increment',
                                 return_value=peak)
        m2 = mocker.patch.object(populated_scan.pointing, 'edit_header')

        populated_scan.edit_pointing_header_info(header)
        assert header['PNT_DX'] == 1 / 3600.
        assert header['PNT_DY'] == 2 / 3600.

        m1.assert_called_once()
        m2.assert_called_once()

    def test_get_frame_count(self, populated_scan):
        assert populated_scan.get_frame_count() == 1100
        assert populated_scan.get_frame_count(keep_flag=0) == 1100
        assert populated_scan.get_frame_count(keep_flag=1) == 0
        assert populated_scan.get_frame_count(discard_flag=0) == 0
        assert populated_scan.get_frame_count(discard_flag=1) == 1100
        assert populated_scan.get_frame_count(match_flag=0) == 1100
        assert populated_scan.get_frame_count(match_flag=1) == 0

    def test_get_table_entry(self, populated_scan, initialized_scan):
        # record expected values here instead of
        # parametrizing for speed
        expected = [('test', None),
                    ('?test', None),
                    ('?write.source', 'True'),
                    ('model.test', None),
                    ('pnt.test', None),
                    ('src.test', None),
                    ('object', 'Simulation'),
                    ('id', 'Simulation.1'),
                    ('serial', -1),
                    ('MJD', 59554.78363282),
                    ('UT', .78363282),
                    ('UTh', .78363282 * 24),
                    ('PA', -24.9382),
                    ('PAd', -24.9382),
                    ('AZ', 2.6555),
                    ('EL', 0.3082484),
                    ('AZd', 152.14965),
                    ('ELd', 17.66133),
                    ('RA', 17.761),
                    ('DEC', -29.0061),
                    ('RAd', 17.761 * 15),
                    ('RAh', 17.761),
                    ('DECd', -29.0061),
                    ('epoch', 'J2000.0'),
                    ('epochY', 2000),
                    ('LST', 4.1158),
                    ('LSTh', 15.7212),
                    ('date', '2021-12-06T18:48:25.876'),
                    ('obstime', 110),
                    ('obsmins', 110 / 60),
                    ('obshours', 110 / 60 / 60),
                    ('weight', 1),
                    ('frames', 1100),
                    ('project', None),
                    ('observer', None),
                    ('creator', None),
                    ('integrations', 1),
                    ('generation', 0),
                    ('Tamb', None)]

        for name, value in expected:
            result = populated_scan.get_table_entry(name)
            try:
                assert np.allclose(result, value)
            except TypeError:
                assert result == value

        # most values are None for scan with no integrations
        expected = ['test', 'system']
        for name in expected:
            result = initialized_scan.get_table_entry(name)
            assert result is None

    def test_get_table_entry_no_horizontal(self, populated_scan):
        # missing horizontal
        populated_scan.site = None
        expected = [('AZ', None),
                    ('EL', None),
                    ('AZd', None),
                    ('ELd', None)]
        for name, value in expected:
            result = populated_scan.get_table_entry(name)
            try:
                assert np.allclose(result, value)
            except TypeError:
                assert result == value

    def test_get_table_entry_config(self, populated_scan):
        # get a config value that exists but isn't set
        populated_scan.configuration.set_option('pointing', [])
        expected = [('?pointing', True)]
        for name, value in expected:
            result = populated_scan.get_table_entry(name)
            try:
                assert np.allclose(result, value)
            except TypeError:
                assert result == value

    def test_get_table_entry_weather(self, initialized_hawc_scan):
        # no scan: values mostly default to NaN
        expected = [('Tamb', np.nan),
                    ('humidity', np.nan),
                    ('pressure', np.nan),
                    ('windspeed', np.nan),
                    ('windpeak', np.nan),
                    ('winddir', 0)]
        for name, value in expected:
            result = initialized_hawc_scan.get_table_entry(name)
            try:
                assert np.allclose(result, value, equal_nan=True)
            except TypeError:
                assert result == value

    def test_get_table_entry_source_model(self, pointing_scan):
        expected = [('model.test', None),
                    ('model.system', 'EQ'),
                    ('pnt.test', None),
                    ('pnt.X', -0.147642 * units.arcsec),
                    ('src.test', None),
                    ('src.peak', 0.118942 * units.ct),
                    ('peak', None),
                    ('system', 'EQ')
                    ]
        for name, value in expected:
            result = pointing_scan.get_table_entry(name)
            try:
                assert np.allclose(result, value, equal_nan=True)
            except TypeError:
                assert result == value

    def test_report_pointing(self, capsys, pointing_scan):
        pointing_scan.report_pointing()
        assert 'Pointing Results' in capsys.readouterr().out

        pointing_scan.pointing = None
        pointing_scan.configuration.set_option('pointing', 'suggest')
        pointing_scan.report_pointing()
        assert 'Cannot suggest pointing' in capsys.readouterr().err

    def test_report_focus(self, capsys, pointing_scan):
        pointing_scan.report_focus()
        assert 'Focus Results' in capsys.readouterr().out

        pointing_scan.pointing = None
        pointing_scan.configuration.set_option('pointing', 'suggest')
        pointing_scan.report_focus()
        assert 'Cannot suggest focus' in capsys.readouterr().err

    def test_split(self, initialized_scan, populated_scan):
        # returns [self] for unpopulated scan or single integration
        result = initialized_scan.split()
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is initialized_scan

        result = populated_scan.split()
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is populated_scan

        # splits on integration list otherwise
        integ1 = populated_scan.integrations[0]
        integ2 = integ1.copy()
        populated_scan.integrations = [integ1, integ2]
        result = populated_scan.split()
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].integrations == [integ1]
        assert result[1].integrations == [integ2]

    def test_update_gains(self, populated_scan):
        # no op if not configured
        populated_scan.configuration.set_option('gains', False)
        populated_scan.update_gains('test')
        assert populated_scan.integrations[0].comments == []

        populated_scan.configuration.set_option('gains', True)
        populated_scan.configuration.set_option('correlated.test.nogains',
                                                True)
        populated_scan.update_gains('test')
        assert populated_scan.integrations[0].comments == []

        # configure to update, but for non-existent modality
        del populated_scan.configuration['correlated.test.nogains']
        populated_scan.update_gains('test')
        assert populated_scan.integrations[0].comments == []

        # update for an existing modality: fails if not previously created
        with pytest.raises(AttributeError):
            populated_scan.update_gains('bias')

        # perform correlated.bias task, then update_gains
        expected = ['O', ' ', 'C', '121', ' ']
        populated_scan.validate()
        populated_scan.perform('correlated.obs-channels')
        assert populated_scan.integrations[0].comments == expected
        test_integ = populated_scan.integrations[0].copy()

        populated_scan.update_gains('obs-channels')
        assert populated_scan.integrations[0].comments == expected + ['121']

        # set a trigger to False: gains not updated
        populated_scan.integrations[0] = test_integ
        modality = test_integ.channels.modalities.get('obs-channels')
        modality.trigger = 'False'
        populated_scan.update_gains('obs-channels')
        assert populated_scan.integrations[0].comments == expected

    def test_decorrelate(self, mocker, populated_scan):
        m1 = mocker.patch.object(populated_scan, 'update_gains')
        populated_scan.validate()
        populated_scan.decorrelate('obs-channels')

        # update_gains not called by default
        m1.assert_not_called()

        # configure span: calls update_gains
        populated_scan.configuration.set_option(
            'correlated.obs-channels.span', True)
        populated_scan.decorrelate('obs-channels')
        m1.assert_called_once()

        # decorrelate an unconfigured non existent modality - Nothing happens
        populated_scan.integrations[0].comments = None
        populated_scan.decorrelate('test')
        m1.assert_called_once()
        assert populated_scan.integrations[0].comments is None

        # decorrelate an configured non-existent modality - appends comment
        populated_scan.configuration.parse_key_value(
            'correlated.test', 'True')
        populated_scan.decorrelate('test')
        m1.assert_called_once()
        assert populated_scan.integrations[0].comments == [' ']

    def test_get_id(self):
        channels = ExampleChannels()
        scan = DemoScanClass(channels)
        scan.serial_number = None
        assert scan.get_id() == '1'
        scan.serial_number = 4
        assert scan.get_id() == '4'

    def test_get_pointing_data(self, mocker, pointing_scan):
        ptg = pointing_scan.pointing

        # error if pointing is None
        pointing_scan.pointing = None
        with pytest.raises(ValueError) as err:
            pointing_scan.get_pointing_data()
        assert 'No pointing data' in str(err)

        # expected values
        expected = {'dX': -0.14764207 * units.arcsec,
                    'dY': -0.11323433 * units.arcsec,
                    'X': -0.14764207 * units.arcsec,
                    'Y': -0.11323433 * units.arcsec,
                    'asymX': 0.3364783 * units.Unit('percent'),
                    'asymY': 0.10495604 * units.Unit('percent'),
                    'dasymX': 0.0085857 * units.Unit('percent'),
                    'dasymY': 0.00857513 * units.Unit('percent'),
                    'elong': 11.97763979 * units.Unit('percent'),
                    'delong': 29.43201024 * units.Unit('percent'),
                    'angle': 88.36674344 * units.deg,
                    'dangle': 88.36674344 * units.deg,
                    'elongX': -8.22818287 * units.Unit('percent'),
                    'delongX': 2446.21584057 * units.Unit('percent')}

        pointing_scan.pointing = ptg
        result = pointing_scan.get_pointing_data()
        assert set(result.keys()) == set(expected.keys())
        for name, value in expected.items():
            assert np.allclose(result[name], value)

        # mock a nasmyth mount
        mount = pointing_scan.info.instrument.mount
        pointing_scan.info.instrument.mount = Mount.LEFT_NASMYTH
        expected.update({'dNasX': -0.10669246 * units.arcsec,
                         'dNasY': -0.15243658 * units.arcsec,
                         'NasX': -0.10669246 * units.arcsec,
                         'NasY': -0.15243658 * units.arcsec})
        result = pointing_scan.get_pointing_data()
        assert set(result.keys()) == set(expected.keys())
        for name, value in expected.items():
            assert np.allclose(result[name], value)

        # mock spherical offset - recorded as lat/lon instead of x/y
        pointing_scan.info.instrument.mount = mount
        relative = SphericalCoordinates()
        relative.x = 1 * units.arcsec
        relative.y = 2 * units.arcsec
        mocker.patch.object(pointing_scan, 'get_native_pointing_increment',
                            return_value=relative)
        result = pointing_scan.get_pointing_data()
        assert 'X' not in result
        assert 'Y' not in result
        assert result['LAT'] == 2 * units.arcsec
        assert result['LON'] == 1 * units.arcsec

    def test_get_source_asymmetry(self, mocker, pointing_scan):
        src = pointing_scan.source_model
        region = pointing_scan.pointing

        # no op if not an astro map
        pointing_scan.source_model = None
        result = pointing_scan.get_source_asymmetry(region)
        assert result is None

        # returns asym object if valid source model
        pointing_scan.source_model = src
        result = pointing_scan.get_source_asymmetry(region)
        assert isinstance(result, Asymmetry2D)
        assert np.allclose(result.x, 0.0033648)
        assert np.allclose(result.y, 0.00104956)

        # not ground based, equatorial - angle set to zero
        mocker.patch.object(pointing_scan.astrometry, 'ground_based',
                            False)
        result = pointing_scan.get_source_asymmetry(region)
        assert isinstance(result, Asymmetry2D)
        assert np.allclose(result.x, 0.0034936)
        assert np.allclose(result.y, -0.00046702)

        # horizontal
        mocker.patch.object(src.grid, 'is_horizontal',
                            return_value=True)
        result = pointing_scan.get_source_asymmetry(region)
        assert isinstance(result, Asymmetry2D)
        assert np.allclose(result.x, 0.0027889)
        assert np.allclose(result.y, -0.0010091)

    def test_get_focus_string(self, capsys, pointing_scan):
        ptg = pointing_scan.pointing

        result = pointing_scan.get_focus_string()
        assert 'Elongation' in result
        assert 'FWHM unrealistically low' not in capsys.readouterr().err

        # fwhm too low
        ptg.model.x_stddev = 0 * units.arcsec
        ptg.model.y_stddev = 0 * units.arcsec
        result = pointing_scan.get_focus_string()
        assert 'Elongation' in result
        assert 'FWHM unrealistically low' in capsys.readouterr().err

        # fwhm too high
        ptg.model.x_stddev = 100 * units.arcsec
        ptg.model.y_stddev = 100 * units.arcsec
        result = pointing_scan.get_focus_string()
        assert 'Elongation' in result
        assert 'Source is either too extended ' \
               'or too defocused' in capsys.readouterr().err

        # source is not elliptical
        pointing_scan.pointing = GaussianSource()
        result = pointing_scan.get_focus_string(asymmetry=Asymmetry2D())
        assert 'Elongation' not in result

    def test_get_pointing_string_from_increment(self, mocker, capsys,
                                                pointing_scan):
        s = pointing_scan

        # no increment
        result = s.get_pointing_string_from_increment(None)
        assert result == ''

        # with increment
        inc = s.get_native_pointing_increment(s.pointing)
        result = s.get_pointing_string_from_increment(inc)
        assert result.count(' Offset:') == 1
        assert '(dAZ, dEL)' in result

        # with nasmyth mount: prints two offsets
        mount = pointing_scan.info.instrument.mount
        pointing_scan.info.instrument.mount = Mount.RIGHT_NASMYTH
        result = s.get_pointing_string_from_increment(inc)
        assert result.count(' Offset:') == 2

        # mock error in CS retrieval
        pointing_scan.info.instrument.mount = mount
        mocker.patch.object(inc, 'get_coordinate_class',
                            side_effect=ValueError('test'))
        result = s.get_pointing_string_from_increment(inc)
        assert result.count(' Offset:') == 1
        assert '(dAZ, dEL)' not in result
        assert '(x, y)' in result
        assert 'Could not retrieve' in capsys.readouterr().err

    def test_get_equatorial_pointing(self, pointing_scan):
        s = pointing_scan
        ptg = pointing_scan.pointing
        ptg_copy = ptg.copy()

        # default, already equatorial
        result = s.get_equatorial_pointing(ptg)
        assert np.allclose(result.x, 0.181621 * units.arcsec)
        assert np.allclose(result.y, -0.040425 * units.arcsec)

        # different epoch
        ptg.coordinates.epoch = B1950
        result = s.get_equatorial_pointing(ptg)
        # works, returns different offset value
        assert not np.allclose(result.x, 0.181621 * units.arcsec)
        assert not np.allclose(result.y, -0.040425 * units.arcsec)
        ptg = ptg_copy

        # non-equatorial source coordinates
        eqc = s.source_model.reference
        gal = eqc.get_instance('galactic')
        gal.from_equatorial(eqc)
        s.source_model.reference = gal

        # raises error if pointing isn't galactic too
        with pytest.raises(ValueError) as err:
            s.get_equatorial_pointing(ptg)
        assert 'different coordinate system' in str(err)

        # convert pointing
        eqc = ptg.coordinates
        gal = eqc.get_instance('galactic')
        gal.from_equatorial(eqc)
        ptg.coordinates = gal
        result = s.get_equatorial_pointing(ptg)
        # works, returns different offset value
        assert not np.allclose(result.x, 0.181621 * units.arcsec)
        assert not np.allclose(result.y, -0.040425 * units.arcsec)

    def test_get_native_pointing(self, pointing_scan):
        ptg = pointing_scan.pointing
        result = pointing_scan.get_native_pointing(ptg)
        assert np.allclose(result.x, -0.14764 * units.arcsec)
        assert np.allclose(result.y, -0.11323 * units.arcsec)

        # add in a default offset
        c = Coordinate2D()
        c.x = -1 * units.arcsec
        c.y = -2 * units.arcsec
        pointing_scan.pointing_correction = c
        result = pointing_scan.get_native_pointing(ptg)
        assert np.allclose(result.x, -1.14764 * units.arcsec)
        assert np.allclose(result.y, -2.11323 * units.arcsec)

    def test_get_native_pointing_increment(self, pointing_scan,
                                           focal_pointing_scan):
        s = pointing_scan
        ptg_copy = s.pointing.copy()
        src_copy = s.source_model.copy()
        ptg = ptg_copy.copy()
        src = src_copy.copy()

        # default: both equatorial
        result = s.get_native_pointing_increment(ptg)
        assert np.allclose(result.x, -0.14764 * units.arcsec)
        assert np.allclose(result.y, -0.11323 * units.arcsec)

        # one horizontal: raises error
        ptg.coordinates = ptg.coordinates.to_horizontal(s.site, s.lst)
        with pytest.raises(ValueError) as err:
            s.get_native_pointing_increment(ptg)
        assert 'different coordinate system' in str(err)

        # both horizontal
        s.source_model.reference = src.reference.to_horizontal(s.site, s.lst)
        result = s.get_native_pointing_increment(ptg)
        assert np.allclose(result.x, -0.14764 * units.arcsec, rtol=.005)
        assert np.allclose(result.y, -0.11323 * units.arcsec, rtol=.005)

        ptg = ptg_copy.copy()
        s.source_model = src_copy.copy()

        # both galactic
        eqc = s.source_model.reference
        gal = eqc.get_instance('galactic')
        gal.from_equatorial(eqc)
        s.source_model.reference = gal

        eqc = ptg.coordinates
        gal = eqc.get_instance('galactic')
        gal.from_equatorial(eqc)
        ptg.coordinates = gal

        result = s.get_native_pointing_increment(ptg)
        assert np.allclose(result.x, -0.14764 * units.arcsec, rtol=.5)
        assert np.allclose(result.y, -0.11323 * units.arcsec, rtol=.5)

        # both focal plane
        ptg = focal_pointing_scan.pointing
        result = focal_pointing_scan.get_native_pointing_increment(ptg)
        assert np.allclose(result.x, 0.2607 * units.arcsec, rtol=.001)
        assert np.allclose(result.y, -0.091 * units.arcsec, rtol=.001)

    def test_offset_errors(self, mocker, initialized_scan):
        # raises error if not equatorial
        offset = Offset2D(Coordinate2D([0.0, 0.0]), Coordinate2D([0.0, 0.0]))
        with pytest.raises(ValueError) as err:
            initialized_scan.get_native_offset_of(offset)
        assert 'Not an equatorial offset' in str(err)

        with pytest.raises(ValueError) as err:
            initialized_scan.get_nasmyth_offset(offset)
        assert 'Non-native pointing offset' in str(err)

    def test_str(self):
        channels = ExampleChannels()
        scan = DemoScanClass(channels)
        scan.serial_number = 4
        assert str(scan) == 'Scan 4'

    def test_segment_to(self, populated_scan):
        integ1 = populated_scan.integrations[0].copy()
        assert len(populated_scan) == 1
        populated_scan.segment_to(10 * units.s)
        assert len(populated_scan) == 11

        # multiple integrations: merges first, same outcome
        integ2 = integ1.copy()
        # add a gap between integrations
        integ2.frames.mjd += 0.0015
        populated_scan.integrations = [integ1, integ2]
        assert len(populated_scan) == 2

        populated_scan.segment_to(10 * units.s)
        assert len(populated_scan) == 11

        # too big a time: no op
        populated_scan.integrations = [integ1]
        assert len(populated_scan) == 1
        populated_scan.segment_to(2000 * units.s)
        assert len(populated_scan) == 1
        assert populated_scan.integrations[0] is integ1

    def test_time_order_scans(self):
        info = ExampleInfo()
        info.read_configuration()
        channels = info.get_channels_instance()
        s1 = DemoScanClass(channels)
        s2 = DemoScanClass(channels)
        s3 = DemoScanClass(channels)

        s1.mjd = 12345
        s2.mjd = 12346
        s3.mjd = 12347

        assert Scan.time_order_scans([s3, s1, s2]) == [s1, s2, s3]

    def test_calculate_precessions(self, mocker, initialized_scan):
        s = initialized_scan
        m1 = mocker.patch.object(s.info.astrometry, 'calculate_precessions')
        s.calculate_precessions('test')
        m1.assert_called_once()

    def test_frame_midpoint_value(self, populated_scan):
        assert np.allclose(populated_scan.frame_midpoint_value('mjd'),
                           59554.78426)
        assert populated_scan.frame_midpoint_value('sign') == 1

        with pytest.raises(ValueError) as err:
            populated_scan.frame_midpoint_value('test')
        assert 'does not contain a test field' in str(err)
