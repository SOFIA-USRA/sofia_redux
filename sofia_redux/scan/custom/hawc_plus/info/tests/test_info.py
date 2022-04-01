# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
from astropy.table import Table
from copy import deepcopy
import numpy as np
import os
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.info.base import InfoBase
from sofia_redux.scan.custom.hawc_plus.info.info import HawcPlusInfo
from sofia_redux.scan.custom.sofia.info.gyro_drifts import GyroDrift


degree = units.Unit('degree')
second = units.Unit('second')


class DummySource(object):  # pragma: no cover
    @staticmethod
    def get_default_core_name():
        return 'source.fits'


class DummyReduction(object):  # pragma: no cover

    def __init__(self, configuration=None):
        self.configuration = configuration.copy()
        self.parallel_read = 1
        self.sub_reductions = None
        self.parent_reduction = None
        self.reduction_files = None
        self.reduction_number = -1
        self.is_read = False
        self.is_validated = False
        self.is_reduced = False
        self.source = DummySource()

    def blank_copy(self):
        return deepcopy(self)

    def update_parallel_config(self, reset=True):
        pass

    def read_scans(self, *args, **kwargs):
        self.is_read = True

    def validate(self):
        self.is_validated = True

    def reduce(self):
        self.is_reduced = True

    @staticmethod
    def create_hwp_file(path, angle):
        hwp_step = 0.25 * degree
        hwp_counts = int((angle / hwp_step).decompose().value)
        filename = os.path.join(path, f'angle_{angle.value:.2f}.fits')
        hdul = fits.HDUList()
        table = Table({'hwpCounts': np.full(10, hwp_counts)})
        hdu = fits.BinTableHDU(table)
        hdu.header['EXTNAME'] = 'Timestream'
        hdul.append(hdu)
        hdul.writeto(filename, overwrite=True)
        hdul.close()
        return filename


@pytest.fixture
def hawc_header():
    h = fits.Header()
    h['DETECTOR'] = 'HAWC'
    h['DETSIZE'] = '64,40'
    h['PIXSCAL'] = 9.43
    h['SUBARRNO'] = 3
    h['SIBS_X'] = 15.5
    h['SIBS_Y'] = 19.5
    h['CTYPE1'] = 'RA---TAN'
    h['CTYPE2'] = 'DEC--TAN'
    h['SUBARR01'] = 10
    h['SUBARR02'] = 11
    h['MCEMAP'] = '0,2,1,-1'
    return h


@pytest.fixture
def hawc_configuration(hawc_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(hawc_header)
    c.parse_key_value('darkcorrect', 'True')
    c.parse_key_value('hwp', '2')
    return c


@pytest.fixture
def gyro_header():
    h = fits.Header()
    h['DBRA0'] = '00:00:00.0'
    h['DBDEC0'] = 30.0
    h['DARA0'] = '00:08:00.0'
    h['DADEC0'] = 31.0
    h['EQUINOX'] = 2000.0
    h['DBTIME0'] = 10.0
    h['DATIME0'] = 11.0
    return h


def test_init():
    info = HawcPlusInfo()
    assert info.name == 'hawc_plus'
    for attribute in ['astrometry', 'gyro_drifts', 'chopping',
                      'detector_array', 'instrument', 'telescope',
                      'observation', 'scanning']:
        assert isinstance(getattr(info, attribute), InfoBase)
    assert info.spectroscopy is None
    assert info.hwp_grouping_angle == 2 * degree


def test_get_file_id():
    assert HawcPlusInfo().get_file_id() == 'HAW'


def test_edit_header(hawc_configuration):
    h = fits.Header()
    info = HawcPlusInfo()
    info.detector_array.configuration = hawc_configuration.copy()
    info.detector_array.apply_configuration()
    info.edit_header(h)
    assert h['SMPLFREQ'] == -9999
    assert h['DETHWPAG'] == 2
    info.sampling_interval = 0.1 * second
    info.detector_array.subarrays_requested = 'R0, R1, T0'
    info.edit_header(h)
    assert h['SMPLFREQ'] == 10
    assert h['SUBARRAY'] == 'R0, R1, T0'


def test_validate_scans(populated_hawc_scan, gyro_header, capsys):
    scan1 = populated_hawc_scan.copy()
    channels = scan1.channels.copy()
    channels.info.unlink_configuration()
    scan2 = scan1.copy()
    scan2.channels = channels

    scans = [scan1, scan2]
    scan_list = scans.copy()
    info = scan1.info

    info.validate_scans(None)  # Does nothing

    scan2.info.configuration.purge('gyrocorrect.max')
    info.validate_scans(scans)
    assert scans == scan_list

    scan2.info.configuration.parse_key_value('gyrocorrect.max', '10.0')
    drift = GyroDrift(gyro_header, 0)
    scan2.info.gyro_drifts.drifts.append(drift)
    info.validate_scans(scans)
    assert len(scans) == 1 and scan2 not in scans and scan1 in scans
    assert 'too large gyro' in capsys.readouterr().err

    scans = scan_list.copy()
    scan2.info.instrument.instrument_config = 'FOO'
    info.validate_scans(scans)
    assert len(scans) == 1 and scan2 not in scans and scan1 in scans
    assert 'different instrument configuration' in capsys.readouterr().err

    scans = scan_list.copy()
    scan2.info.instrument.wavelength = -1 * units.Unit('um')
    info.validate_scans(scans)
    assert len(scans) == 1 and scan2 not in scans and scan1 in scans
    assert 'different band' in capsys.readouterr().err


def test_max_pixels():
    info = HawcPlusInfo()
    assert info.max_pixels() == 0
    info.instrument.n_store_channels = 10
    assert info.max_pixels() == 10


def test_get_si_pixel_size():
    info = HawcPlusInfo()
    p = Coordinate2D([2, 3], unit='arcsec')
    info.detector_array.pixel_sizes = p
    assert info.get_si_pixel_size() == p


def test_perform_reduction(tmpdir):
    path = str(tmpdir.mkdir('test_perform_reduction'))

    info = HawcPlusInfo()
    angles = np.arange(6) * degree
    filenames = []
    for angle in angles:
        filenames.append(DummyReduction.create_hwp_file(path, angle))
    groups = info.group_files_by_hwp(filenames, jobs=1)

    # Single file
    r = DummyReduction(configuration=info.configuration)
    info.perform_reduction(r, filenames[0])
    assert r.is_read and r.is_validated and r.is_reduced
    assert r.sub_reductions is None

    # Single group
    files = groups[0 * degree]
    r = DummyReduction(configuration=info.configuration)
    info.perform_reduction(r, files)
    assert r.sub_reductions is None
    assert r.is_read and r.is_validated and r.is_reduced

    r = DummyReduction(configuration=info.configuration)
    info.perform_reduction(r, filenames)
    assert r.is_read and r.is_validated and r.is_reduced
    assert r.sub_reductions is not None
    assert len(r.sub_reductions) == 4
    for s in r.sub_reductions:
        name = s.configuration['name']
        assert name.startswith('source_') and name.endswith('.fits')


def test_group_files_by_hwp(tmpdir):
    path = str(tmpdir.mkdir('test_group_files_by_hwp'))
    info = HawcPlusInfo()
    assert info.hwp_grouping_angle == 2 * degree
    angles = np.arange(6) * degree
    filenames = []
    for angle in angles:
        filenames.append(DummyReduction.create_hwp_file(path, angle))
    filenames.append(os.path.join(path, 'foo.fits'))
    filenames.append(os.path.join(path, 'bar.fits'))
    groups = info.group_files_by_hwp(filenames, jobs=1)

    assert len(groups) == 3
    assert 0 * degree in groups
    assert 3 * degree in groups
    assert None in groups
    files = [os.path.basename(x) for x in groups[0 * degree]]
    assert files == ['angle_0.00.fits', 'angle_1.00.fits', 'angle_2.00.fits']
    files = [os.path.basename(x) for x in groups[3 * degree]]
    assert files == ['angle_3.00.fits', 'angle_4.00.fits', 'angle_5.00.fits']
    files = [os.path.basename(x) for x in groups[None]]
    assert files == ['foo.fits', 'bar.fits']

    groups = info.group_files_by_hwp(filenames[0], jobs=1)
    assert len(groups) == 1 and 0 * degree in groups
    assert groups[0 * degree][0].endswith('angle_0.00.fits')


def test_parallel_safe_determine_hwp_angle(tmpdir):
    path = str(tmpdir.mkdir('test_parallel_safe_determine_hwp_angle'))
    info = HawcPlusInfo()
    filename = DummyReduction.create_hwp_file(path, 5 * degree)
    args = ([filename], 0.25 * degree)
    angle = info.parallel_safe_determine_hwp_angle(args, 0)
    assert np.isclose(angle, 5 * degree)


def test_determine_hwp_angle(tmpdir):
    info = HawcPlusInfo()
    path = str(tmpdir.mkdir('test_determine_hwp_angle'))
    filename = DummyReduction.create_hwp_file(path, 5 * degree)
    hwp_step = 0.25 * degree
    angle = info.determine_hwp_angle(filename, hwp_step)
    assert np.isclose(angle, 5 * degree)
    bad_file = os.path.join(path, 'foo.fits')
    angle = info.determine_hwp_angle(bad_file, hwp_step)
    assert np.isclose(angle, np.nan * degree, equal_nan=True)

    bad_hdul = fits.HDUList()
    data = Table({'foo': np.arange(10)})
    hdu1 = fits.BinTableHDU(data)
    hdu1.header['EXTNAME'] = 'FooBar'
    bad_hdul.append(hdu1)
    hdu2 = fits.BinTableHDU(data)
    hdu2.header['EXTNAME'] = 'Timestream'
    bad_hdul.append(hdu2)
    file2 = os.path.join(path, 'empty_file.fits')
    bad_hdul.writeto(file2)
    bad_hdul.close()
    angle = info.determine_hwp_angle(file2, hwp_step)
    assert np.isclose(angle, np.nan * degree, equal_nan=True)


def test_split_reduction(tmpdir):
    info = HawcPlusInfo()
    path = str(tmpdir.mkdir('test_split_reduction'))

    angles = np.arange(6) * degree
    filenames = []
    for angle in angles:
        filenames.append(DummyReduction.create_hwp_file(path, angle))
    groups = info.group_files_by_hwp(filenames, jobs=1)
    reduction = DummyReduction(configuration=info.configuration)

    info.split_reduction(reduction, groups)
    reductions = reduction.sub_reductions
    assert len(reductions) == 4
    for i, r in enumerate(reductions):
        assert r.parent_reduction is reduction
        assert r.reduction_number == i + 1
        assert len(r.reduction_files) == 3
        angle = '0.00' if i < 2 else '3.00'
        subarray = 'R0' if i % 2 == 0 else 'T0'
        assert r.configuration['hwp'] == angle
        assert r.configuration['subarray'] == subarray
        assert r.configuration['name'] == f'{angle}{subarray}.fits'

    info.configuration.parse_key_value('name', 'foo')
    info.configuration.parse_key_value('subarray', 'R0,R1,T0')
    reduction = DummyReduction(configuration=info.configuration)
    info.split_reduction(reduction, groups)
    for i, r in enumerate(reduction.sub_reductions):
        angle = '0.00' if i < 2 else '3.00'
        subarray = 'R0' if i % 2 == 0 else 'T0'
        name = f'foo.{angle}{subarray}.fits'
        assert r.configuration['name'] == name
        assert r.configuration['subarray'] == 'R0,R1,T0'
