# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.horizontal_coordinates import \
    HorizontalCoordinates
from sofia_redux.scan.custom.sofia.integration.integration import \
    SofiaIntegration
from sofia_redux.scan.source_models.beams.gaussian_source import GaussianSource
from sofia_redux.scan.custom.sofia.scan.scan import SofiaScan
from sofia_redux.scan.reduction.reduction import Reduction
from sofia_redux.scan.source_models.astro_intensity_map import \
    AstroIntensityMap


s = units.Unit('second')
degree = units.Unit('degree')


class Scan(SofiaScan):

    def __init__(self, channels, reduction=None):
        super().__init__(channels, reduction=reduction)
        self.stored_hdul = None

    def add_integrations_from_hdul(self, hdul):
        self.stored_hdul = hdul


@pytest.fixture
def sofia_scan():
    reduction = Reduction('hawc_plus')
    channels = reduction.info.get_channels_instance()
    scan = Scan(channels, reduction=reduction)
    return scan


@pytest.fixture
def populated_sofia_scan(sofia_scan):
    scan = sofia_scan.copy()
    scan.hdul = fits.HDUList()
    integration = SofiaIntegration(scan=scan)
    integration.frames.integration = integration
    integration.frames.set_frame_size(2)
    integration.frames.telescope_vpa = [1, 2] * degree
    integration.frames.instrument_vpa = [3, 4] * degree
    integration.frames.horizontal = HorizontalCoordinates([[10, 10], [11, 11]])
    scan.integrations = [integration]
    scan.astrometry.horizontal = HorizontalCoordinates([10, 31])
    scan.info.detector_array.pixel_sizes = Coordinate2D([1, 1], unit='arcsec')

    return scan


@pytest.fixture
def scan_with_model(populated_sofia_scan):
    scan = populated_sofia_scan.copy()
    source = GaussianSource()
    source.coordinates = EquatorialCoordinates([20, 30])
    model = AstroIntensityMap(scan.info)
    model.reference = EquatorialCoordinates([20, 30.1])
    model.map.data = np.ones((10, 10))
    model.map.weight.data = np.full((10, 10), 0.25)
    source.grid = model.grid
    source.center_index = Coordinate2D([4, 4])
    scan.pointing = source
    scan.source_model = model
    return scan


def test_init():
    reduction = Reduction('hawc_plus')
    channels = reduction.info.get_channels_instance()
    scan = Scan(channels, reduction=reduction)
    assert scan.reduction is reduction
    assert scan.hdul is None
    assert scan.header_extension == 0
    assert scan.header is None
    assert scan.history is None


def test_copy(sofia_scan):
    scan = sofia_scan
    scan2 = scan.copy()
    assert scan is not scan2 and scan.reduction is scan2.reduction


def test_referenced_attributes(sofia_scan):
    assert 'hdul' in sofia_scan.referenced_attributes
    assert 'header' in sofia_scan.referenced_attributes


def test_info(sofia_scan):
    assert sofia_scan.info.instrument.name == 'hawc_plus'


def test_astrometry(sofia_scan):
    assert sofia_scan.astrometry is sofia_scan.info.astrometry


def test_get_lowest_quality(sofia_scan):
    scan1 = sofia_scan.copy()
    scan2 = sofia_scan.copy()
    scan2.channels = scan2.channels.copy()
    scan3 = sofia_scan.copy()
    scan3.channels = scan3.channels.copy()
    flagspace = scan1.info.processing.flagspace
    q1 = flagspace.convert_flag(2)
    q2 = flagspace.convert_flag(4)
    q3 = flagspace.convert_flag(8)
    scan1.info.processing.quality_level = q3
    scan2.info.processing.quality_level = q1
    scan3.info.processing.quality_level = q2
    assert SofiaScan.get_lowest_quality([scan1, scan2, scan3]) == q1
    assert SofiaScan.get_lowest_quality([]) is None


def test_get_total_exposure_time(sofia_scan):
    scan1 = sofia_scan.copy()
    scan2 = scan1.copy()
    scan2.channels = scan2.channels.copy()
    scan1.info.instrument.exposure_time = 2 * s
    scan2.info.instrument.exposure_time = 3 * s
    assert SofiaScan.get_total_exposure_time([scan1, scan2]) == 5 * s
    assert SofiaScan.get_total_exposure_time(None) == 0 * s


def test_has_tracking_error(sofia_scan):
    scan1 = sofia_scan.copy()
    scan2 = scan1.copy()
    scan2.channels = scan2.channels.copy()
    assert not SofiaScan.has_tracking_error(None)
    assert not SofiaScan.has_tracking_error([scan1, scan2])
    scan2.info.telescope.has_tracking_error = True
    assert SofiaScan.has_tracking_error([scan1, scan2])


def test_get_earliest_scan(sofia_scan):
    scan1 = sofia_scan.copy()
    scan2 = sofia_scan.copy()
    scan1.mjd = 1.0
    scan2.mjd = 2.0
    assert SofiaScan.get_earliest_scan([scan1, scan2]) is scan1


def test_get_latest_scan(sofia_scan):
    scan1 = sofia_scan.copy()
    scan2 = sofia_scan.copy()
    scan1.mjd = 1.0
    scan2.mjd = 2.0
    assert SofiaScan.get_latest_scan([scan1, scan2]) is scan2


def test_read(sofia_scan, tmpdir):
    fits_file = str(tmpdir.mkdir('test_read').join('test.fits'))
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU(data=np.ones(1)))
    hdul.writeto(fits_file)
    hdul.close()
    scan = sofia_scan
    assert scan.stored_hdul is None
    scan.read(fits_file)
    assert isinstance(scan.stored_hdul, fits.HDUList)
    assert scan.hdul is None


def test_close_fits(sofia_scan):
    scan = sofia_scan
    scan.close_fits()
    assert scan.hdul is None
    scan.hdul = fits.HDUList()
    scan.close_fits()
    assert scan.hdul is None


def test_read_hdul(sofia_scan, hawc_scan_file, initialized_hawc_scan):
    scan = initialized_hawc_scan
    hdul = fits.open(hawc_scan_file)
    scan.hdul = hdul
    scan.read_hdul(hdul)
    assert np.isclose(scan.info.sampling_interval, 0.00492 * s)
    assert scan.info.integration_time == scan.info.sampling_interval
    assert scan.size == 1
    hdul.close()
    scan = sofia_scan
    hdul = fits.open(hawc_scan_file)
    scan.hdul = hdul
    scan.read_hdul(hdul)
    hdul.close()
    assert scan.size == 0


def test_is_aor_valid(sofia_scan):
    scan = sofia_scan.copy()
    assert not scan.is_aor_valid()
    scan.info.observation.aor_id = '123'
    assert scan.is_aor_valid()


def test_is_coordinate_valid(sofia_scan):
    scan = sofia_scan
    c = Coordinate2D([1, 1], unit='arcsec')
    assert scan.is_coordinate_valid(c)
    c.x = -9999 * units.Unit('arcsec')
    assert not scan.is_coordinate_valid(c)


def test_is_requested_valid(sofia_scan):
    scan = sofia_scan
    h = fits.Header()
    h['OBSRA'] = 20.0
    h['OBSDEC'] = 30.0
    assert scan.is_requested_valid(h)
    h['OBSRA'] = -9999.0
    assert not scan.is_requested_valid(h)


def test_guess_reference_coordinates(sofia_scan):
    scan = sofia_scan
    h = fits.Header()
    h['OBSRA'] = 20.0
    h['OBSDEC'] = 30.0
    reference = scan.guess_reference_coordinates(header=h)
    assert reference.ra == 20 * degree
    assert reference.dec == 30 * degree


def test_edit_scan_header(populated_hawc_scan):
    h = fits.Header()
    scan = populated_hawc_scan
    scan.validate()
    scan.info.origin.checksum = 'abc'
    scan.info.origin.checksum_version = 'md5sum'
    scan.edit_scan_header(h)
    assert 'DATE' in h
    assert h['DATASUM'] == 'abc'
    assert h['CHECKVER'] == 'md5sum'


def test_validate(sofia_scan, capsys):
    scan = sofia_scan.copy()
    with pytest.raises(ValueError) as err:
        scan.validate()
    assert 'No integrations exist' in str(err.value)
    scan.configuration.parse_key_value('lab', 'True')
    scan.validate()
    assert 'No integrations to validate' in capsys.readouterr().err


def test_get_telescope_vpa(populated_sofia_scan):
    scan = populated_sofia_scan
    assert np.isclose(scan.get_telescope_vpa(), 1.5 * degree)


def test_get_instrument_vpa(populated_sofia_scan):
    scan = populated_sofia_scan
    assert np.isclose(scan.get_instrument_vpa(), 3.5 * degree)


def test_get_id(sofia_scan):
    scan = sofia_scan.copy()
    assert scan.get_id() == 'None.UNKNOWN'
    scan.info.observation.obs_id = 'UNKNOWN-FOO'
    assert scan.get_id() == 'None.-FOO'
    scan.info.observation.obs_id = 'FOO'
    assert scan.get_id() == 'FOO'


def test_get_pointing_data(scan_with_model):
    scan = scan_with_model.copy()
    pointing = scan.get_pointing_data()
    assert 'dSIBSX' in pointing and 'dSIBSY' in pointing


def test_get_flight_number(sofia_scan):
    scan = sofia_scan.copy()
    scan.info.mission.mission_id = None
    assert scan.get_flight_number() == -1
    scan.info.mission.mission_id = 'abc'
    assert scan.get_flight_number() == -1
    scan.info.mission.mission_id = 'A_F123'
    assert scan.get_flight_number() == 123


def test_get_scan_number(sofia_scan):
    scan = sofia_scan.copy()
    scan.info.observation.obs_id = None
    assert scan.get_scan_number() == -1
    scan.info.observation.obs_id = 'abc'
    assert scan.get_scan_number() == -1
    scan.info.observation.obs_id = 'a-123'
    assert scan.get_scan_number() == 123


def test_get_table_entry(sofia_scan):
    scan = sofia_scan.copy()
    scan.info.observation.obs_type = 'a'
    scan.info.observation.obs_id = 'b-456'
    scan.info.mission.mission_id = 'A_F123'
    scan.astrometry.date = 'today'
    assert scan.get_table_entry('obstype') == 'a'
    assert scan.get_table_entry('flight') == 123
    assert scan.get_table_entry('scanno') == 456
    assert scan.get_table_entry('date') == 'today'
    scan.info.available_info['bar'] = None
    assert scan.get_table_entry('inst.name') == 'hawc_plus'
    assert scan.get_table_entry('foo') is None


def test_get_nominal_pointing_offset(sofia_scan):
    scan = sofia_scan.copy()
    native_pointing = Coordinate2D([1, 2], unit='arcsec')
    scan.configuration.parse_key_value('pointing', '3,4')
    offset = scan.get_nominal_pointing_offset(native_pointing)
    assert offset == Coordinate2D([4, 6], unit='arcsec')


def test_get_si_arcseconds_offset(sofia_scan):
    scan = sofia_scan.copy()
    scan.configuration.parse_key_value('pointing', '0,0')
    native_pointing = Coordinate2D([0.1, 0.2], unit='arcmin')
    offset = scan.get_si_arcseconds_offset(native_pointing)
    assert offset == Coordinate2D([6, 12], unit='arcsec')


def test_get_si_pixel_offset(populated_sofia_scan):
    scan = populated_sofia_scan.copy()
    scan.configuration.parse_key_value('pointing', '0,0')

    native_pointing = Coordinate2D([1, 2], unit='arcsec')
    offset = scan.get_si_pixel_offset(native_pointing)
    assert np.allclose(offset.coordinates, [1.06918982, 1.96388216], atol=1e-5)


def test_get_pointing_string_from_increment(populated_sofia_scan):
    scan = populated_sofia_scan.copy()
    scan.configuration.parse_key_value('pointing', '0,0')
    native_pointing = Coordinate2D([1, 2], unit='arcsec')
    s = scan.get_pointing_string_from_increment(native_pointing)
    assert 'Offset: 1.000 arcsec, 2.000 arcsec (x, y)' in s
    assert 'SIBS offset --> 1.0692, 1.9639 pixels' in s


def test_edit_pointing_header_info(scan_with_model):
    scan = scan_with_model.copy()
    h = fits.Header()
    scan.edit_pointing_header_info(h)
    for key in ['SIBS_DX', 'SIBS_DY', 'SIBS_DXE', 'SIBS_DE']:
        assert key in h
