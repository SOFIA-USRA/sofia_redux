# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from configobj import ConfigObj
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.horizontal_coordinates import \
    HorizontalCoordinates
from sofia_redux.scan.coordinate_systems.geodetic_coordinates import \
    GeodeticCoordinates
from sofia_redux.scan.channels.division.division import ChannelDivision
from sofia_redux.scan.channels.mode.mode import Mode
from sofia_redux.scan.channels.modality.modality import Modality
from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.example.channels.channel_group.channel_group \
    import ExampleChannelGroup
from sofia_redux.scan.custom.example.info.info import ExampleInfo
from sofia_redux.scan.custom.example.scan.scan import ExampleScan
from sofia_redux.scan.custom.hawc_plus.info.info import HawcPlusInfo
from sofia_redux.scan.custom.hawc_plus.scan.scan import HawcPlusScan
from sofia_redux.scan.custom.hawc_plus.simulation.simulation \
    import HawcPlusSimulation
from sofia_redux.scan.custom.hawc_plus.channels.channel_data.channel_data \
    import HawcPlusChannelData
from sofia_redux.scan.utilities.utils import safe_sidereal_time
from sofia_redux.scan.reduction.reduction import Reduction
from sofia_redux.scan.custom.fifi_ls.channels.channel_data.channel_data \
    import FifiLsChannelData
from sofia_redux.scan.custom.fifi_ls.simulation.simulation import \
    FifiLsSimulation


# Configuration fixtures

@pytest.fixture
def config_options():
    """
    A sample set of options for configuration testing.

    Returns
    -------
    ConfigObj
    """
    options = {
        'rounds': '5',
        'switch1': 'True',
        'switch2': 'False',
        'testvalue1': 'foo',
        'testvalue2': 'bar',
        'testvalue3': 'lock_me',
        'lock': 'testvalue3',
        'refra': '{?fits.OBSRA}',
        'refdec': '{?fits.OBSDEC}',
        'refsrc': '{?fits.OBJECT}',
        'options1': {'value': 'True',
                     'suboptions1': {'v1': '1', 'v2': '2'},
                     'suboptions2': {'v1': '3', 'v2': '3'}},
        'options2': {'value1': 'a',
                     'value2': 'b',
                     'suboptions1': {'v1': 'c', 'v2': 'd'},
                     'suboptions2': {'v1': 'e', 'v2': 'f'}},

        'aliases': {'o1s1': 'options1.suboptions1',
                    'o1s2': 'options1.suboptions2',
                    'o2s1': 'options2.suboptions1',
                    'o2s2': 'options2.suboptions2',
                    'i': 'iteration',
                    'i1': 'iteration.1',
                    'final': 'iteration.-1'},

        'iteration': {'2': {'o1s1.v1': '10'},
                      '0.6': {'o1s1.v1': '20'}},

        'date': {'*--*': {'add': 'alldates'},
                  '2020-07-01--2020-07-31': {'add': 'jul2020'},
                  '2021-08-01--2021-08-31': {'add': 'aug2021'}},

        'object': {'source1': {'add': 'src1'},
                   'source2': {'testvalue1': 'baz'},
                   'Asphodel': {'add': 'finally'}},

        'serial': {'1-5': {'add': 'scans1to5'},
                   '2-3': {'add': 'scans2to3'}},

        'fits': {'addkeys': ['STORE1', 'STORE2']},

        'conditionals': {
            'switch2': {'final': {'forget': 'testvalue1'}},
            'refra>12': {'add': 'switch3'},
            'switch3': {'o2s2.v2': 'z'},
            'jul2020': {'final': {'forget': 'testvalue2'}},
            'aug2021': {'final': {'add': 'lastiteration'}}
        }
    }
    return ConfigObj(options)


@pytest.fixture
def fits_header():
    h = fits.Header()
    h['STORE1'] = 1, 'Stored value 1'
    h['STORE2'] = 2, 'Stored value 2'
    h['STORE3'] = 3, 'Stored value 3'
    h['REFVAL'] = 'a_fits_value', 'The referenced FITS value'
    h['OBSRA'] = 12.5, 'The observation right ascension (hours)'
    h['OBSDEC'] = 45.0, 'The observation declination (degree)'
    h['OBJECT'] = 'Asphodel'
    return h


@pytest.fixture
def fits_file(fits_header, tmpdir):
    h = fits_header
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU(data=np.zeros((5, 5)), header=h))
    for extension in range(3):
        he = h.copy()
        for val in range(1, 4):
            he[f'STORE{val}'] += 10 * val
        hdul.append(fits.ImageHDU(data=np.zeros((5, 5)), header=he))

    filename = str(tmpdir.mkdir('test_fits_file_').join('test_file.fits'))
    hdul.writeto(filename)
    hdul.close()
    return filename


@pytest.fixture
def config_file(tmpdir, config_options):
    c = config_options
    filename = str(tmpdir.mkdir('test_configuration_').join('config.cfg'))
    c.filename = filename
    c.write()
    return filename


@pytest.fixture
def initialized_configuration(config_options):
    configuration = Configuration()
    configuration.read_configuration(config_options)
    return configuration


@pytest.fixture
def fits_configuration(initialized_configuration, fits_header):
    c = initialized_configuration
    c.read_fits(fits_header)
    return c


@pytest.fixture
def initialized_conditions(fits_configuration):
    """
    Return an initialized Conditions from fully initialized Configuration.

    Parameters
    ----------
    fits_configuration : Configuration

    Returns
    -------
    Conditions
    """
    return fits_configuration.conditions


# ChannelData fixtures

@pytest.fixture
def overlaps():
    values = np.arange(100)
    valid = np.arange(25)
    rows = np.tile(np.arange(5), (5, 1)).ravel()
    cols = np.tile(np.arange(5), (5, 1)).T.ravel()
    overlap_values = csr_matrix((values[valid],
                                 (rows, cols)),
                                shape=(10, 10))
    return overlap_values


# Example instrument simulation

@pytest.fixture
def scan_file(tmpdir):
    reduction = Reduction('example')
    fname = str(tmpdir.join('test.fits'))
    reduction.info.write_simulated_hdul(fname, fwhm=10 * units.arcsec)
    return fname


@pytest.fixture
def bad_file(tmpdir):
    hdul = fits.HDUList(fits.PrimaryHDU(data=[1, 2, 3]))
    fname = str(tmpdir.join('bad.fits'))
    hdul.writeto(fname, overwrite=True)
    hdul.close()
    return fname


@pytest.fixture
def initialized_scan():
    info = ExampleInfo()
    info.read_configuration()
    channels = info.get_channels_instance()
    scan = ExampleScan(channels)
    return scan


@pytest.fixture
def populated_scan(scan_file):
    info = ExampleInfo()
    info.read_configuration()
    channels = info.get_channels_instance()
    scan = channels.read_scan(scan_file)
    return scan


@pytest.fixture
def reduced_scan(scan_file, tmpdir):
    with tmpdir.as_cwd():
        reduction = Reduction('example')
        reduction.run(scan_file)
    return reduction.scans[0]


@pytest.fixture
def pointing_scan(scan_file, tmpdir):
    with tmpdir.as_cwd():
        reduction = Reduction('example')
        reduction.configuration.set_option('point', True)
        reduction.run(scan_file)
    return reduction.scans[0]


@pytest.fixture
def focal_pointing_scan(scan_file, tmpdir):
    with tmpdir.as_cwd():
        reduction = Reduction('example')
        reduction.configuration.set_option('point', True)
        reduction.configuration.set_option('focalplane', True)
        reduction.run(scan_file)
    return reduction.scans[0]


@pytest.fixture
def populated_integration(populated_scan):
    return populated_scan.integrations[0]


@pytest.fixture
def populated_data(populated_scan):
    return populated_scan.channels.data


@pytest.fixture
def example_modality(populated_data):
    g1 = ExampleChannelGroup(populated_data, name='test_g1')
    g2 = ExampleChannelGroup(populated_data, name='test_g2')
    division = ChannelDivision('test_division', groups=[g1, g2])
    modality = Modality(name='test', mode_class=Mode,
                        channel_division=division, gain_provider='gain')
    return modality


@pytest.fixture
def skydip_file(tmpdir):
    reduction = Reduction('example')
    fname = str(tmpdir.join('skydip.fits'))
    reduction.info.write_simulated_hdul(fname, scan_pattern='skydip',
                                        source_type='sky',
                                        start_elevation=80,  # degrees
                                        end_elevation=10,  # degrees
                                        scan_time=45)  # seconds
    return fname


@pytest.fixture
def skydip_scan(skydip_file):
    info = ExampleInfo()
    info.read_configuration()
    channels = info.get_channels_instance()
    scan = channels.read_scan(skydip_file)
    return scan


# HAWC+ simulation

@pytest.fixture
def hawc_scan_file(tmpdir):
    reduction = Reduction('hawc_plus')
    sim = HawcPlusSimulation(reduction.info)

    header_options = fits.Header()
    header_options['CHPNOISE'] = 3.0  # Chopper noise (arcsec)
    header_options['SRCAMP'] = 20.0  # NEFD estimate
    header_options['SRCS2N'] = 30.0  # source signal to noise
    header_options['OBSDEC'] = 7.406657  # declination (degree)
    header_options['OBSRA'] = 1.272684  # ra (hours)
    header_options['SPECTEL1'] = 'HAW_C'  # sets band
    header_options['SRCSIZE'] = 20  # source FWHM (arcsec)
    header_options['ALTI_STA'] = 41993.0
    header_options['ALTI_END'] = 41998.0
    header_options['LON_STA'] = -108.182373
    header_options['LAT_STA'] = 47.043457
    header_options['EXPTIME'] = 30.0  # scan length (seconds)
    header_options['DATE-OBS'] = '2016-12-14T06:41:30.450'

    hdul = sim.create_simulated_hdul(header_options=header_options)

    fname = str(tmpdir.join('test.fits'))
    hdul.writeto(fname, overwrite=True)
    hdul.close()
    return fname


@pytest.fixture
def hawc_chopscan_file(tmpdir):
    reduction = Reduction('hawc_plus')
    sim = HawcPlusSimulation(reduction.info)

    header_options = fits.Header()
    header_options['CHOPPING'] = True
    header_options['CHPNOISE'] = 3.0  # Chopper noise (arcsec)
    header_options['SRCAMP'] = 20.0  # NEFD estimate
    header_options['SRCS2N'] = 30.0  # source signal to noise
    header_options['OBSDEC'] = 7.406657  # declination (degree)
    header_options['OBSRA'] = 1.272684  # ra (hours)
    header_options['SPECTEL1'] = 'HAW_C'  # sets band
    header_options['SRCSIZE'] = 20  # source FWHM (arcsec)
    header_options['ALTI_STA'] = 41993.0
    header_options['ALTI_END'] = 41998.0
    header_options['LON_STA'] = -108.182373
    header_options['LAT_STA'] = 47.043457
    header_options['EXPTIME'] = 30.0  # scan length (seconds)
    header_options['DATE-OBS'] = '2016-12-14T06:41:30.450'

    hdul = sim.create_simulated_hdul(header_options=header_options)

    fname = str(tmpdir.join('test.fits'))
    hdul.writeto(fname, overwrite=True)
    hdul.close()
    return fname


@pytest.fixture
def initialized_hawc_scan():
    info = HawcPlusInfo()
    info.read_configuration()
    channels = info.get_channels_instance()
    scan = HawcPlusScan(channels)
    return scan


@pytest.fixture
def populated_hawc_scan(hawc_scan_file):
    info = HawcPlusInfo()
    info.read_configuration()
    channels = info.get_channels_instance()
    scan = channels.read_scan(hawc_scan_file)
    return scan


@pytest.fixture
def populated_hawc_chopscan(hawc_chopscan_file):
    info = HawcPlusInfo()
    info.read_configuration()
    channels = info.get_channels_instance()
    scan = channels.read_scan(hawc_chopscan_file)
    return scan


@pytest.fixture
def reduced_hawc_scan(hawc_scan_file, tmpdir):
    with tmpdir.as_cwd():
        reduction = Reduction('hawc_plus')
        reduction.run(hawc_scan_file, shift=0.0, blacklist='correlated.bias')
    return reduction.scans[0]


@pytest.fixture
def populated_hawc_integration(populated_hawc_scan):
    return populated_hawc_scan.integrations[0]


@pytest.fixture
def populated_hawc_chop_integration(populated_hawc_chopscan):
    return populated_hawc_chopscan.integrations[0]


@pytest.fixture
def hawc_plus_channel_data():
    info = HawcPlusInfo()
    info.read_configuration()
    header = fits.Header(HawcPlusSimulation.default_values)
    info.configuration.read_fits(header)
    info.apply_configuration()
    channels = info.get_channels_instance()
    data = HawcPlusChannelData(channels=channels)
    data.configuration.parse_key_value('pixelsize', '5.0')
    data.info.detector_array.load_detector_configuration()
    data.info.detector_array.initialize_channel_data(data)
    data.info.detector_array.set_boresight()
    data.channels.subarray_gain_renorm = np.full(4, 1.0)
    data.calculate_sibs_position()
    center = data.info.detector_array.get_sibs_position(
        sub=0,
        row=39 - data.info.detector_array.boresight_index.y,
        col=data.info.detector_array.boresight_index.x)
    data.position.subtract(center)
    return data


@pytest.fixture
def jump_file_zeros(tmpdir):
    filename = str(tmpdir.mkdir('fake_jump_data').join('jump.dat'))
    data = np.zeros((32, 123), dtype=int)
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU(data=data))
    hdul.writeto(filename)
    return filename


@pytest.fixture
def hawc_plus_channels(jump_file_zeros):
    info = HawcPlusInfo()
    info.read_configuration()
    h = HawcPlusSimulation.default_values.copy()
    h['SPECTEL1'] = 'HAW_A'
    h['SPECTEL2'] = 'HAW_HWP_Open'
    h['WAVECENT'] = 53.0
    h['CHOPPING'] = True
    h['CHPNOISE'] = 3.0  # Chopper noise (arcsec)
    h['SRCAMP'] = 20.0  # NEFD estimate
    h['SRCS2N'] = 30.0  # source signal to noise
    h['OBSDEC'] = 7.406657  # declination (degree)
    h['OBSRA'] = 1.272684  # ra (hours)
    h['SRCSIZE'] = 20  # source FWHM (arcsec)
    h['ALTI_STA'] = 41993.0
    h['ALTI_END'] = 41998.0
    h['LON_STA'] = -108.182373
    h['LAT_STA'] = 47.043457
    h['EXPTIME'] = 20  # scan length (seconds)
    h['DATE-OBS'] = '2016-12-14T06:41:30.450'
    header = fits.Header(h)
    info.configuration.parse_key_value('subarray', 'R0,T0,R1')
    info.configuration.parse_key_value('jumpdata', jump_file_zeros)
    info.configuration.lock('subarray')
    info.configuration.lock('jumpdata')
    info.configuration.read_fits(header)
    info.apply_configuration()
    channels = info.get_channels_instance()
    channels.load_channel_data()
    channels.initialize()
    channels.normalize_array_gains()
    return channels


@pytest.fixture
def no_data_scan(hawc_plus_channels):
    scan = hawc_plus_channels.get_scan_instance()
    scan.hdul = fits.HDUList()
    scan.integrations = [scan.get_integration_instance()]
    scan.hdul = None
    integration = scan.integrations[0]
    integration.frames.initialize(integration, 10)
    return scan


@pytest.fixture
def full_hdu(no_data_scan):
    degree = units.Unit('degree')
    hourangle = units.Unit('hourangle')
    frames = no_data_scan[0].frames
    n_frames = frames.size
    row, col = frames.channels.data.fits_row, frames.channels.data.fits_col
    n_row, n_col = row.max() + 1, col.max() + 1
    dac = np.ones((n_frames, n_row, n_col))
    jump = np.zeros((n_frames, n_row, n_col), dtype=int)
    sn = np.arange(n_frames)
    dt = frames.info.instrument.sampling_interval
    t0 = Time('2022-03-28T12:00:00.000')
    t = t0 + np.arange(n_frames) * dt
    utc = t.unix
    zeros = np.zeros(n_frames)
    ra = np.full(n_frames, 12.0)
    dec = np.full(n_frames, 30.0)
    lon = np.full(n_frames, 15.0)
    lat = np.full(n_frames, 20.0)
    lst = safe_sidereal_time(t, 'mean', longitude=lon * units.Unit('degree'))
    equatorial = EquatorialCoordinates([ra * hourangle, dec * degree])
    site = GeodeticCoordinates([lon * degree, lat * degree])
    horizontal = equatorial.to_horizontal(site, lst)
    table = Table(
        {'Timestamp': utc,
         'FrameCounter': sn,
         'FluxJumps': jump,
         'SQ1Feedback': dac,
         'Flag': zeros.astype(int),
         'AZ': horizontal.az.to('degree').value,
         'EL': horizontal.el.to('degree').value,
         'RA': ra,
         'DEC': dec,
         'NonSiderealRA': ra,
         'NonSiderealDec': dec,
         'LST': lst.value,
         'SIBS_VPA': zeros + 1,
         'TABS_VPA': zeros + 2,
         'Chop_VPA': zeros + 3,
         'LON': lon,
         'LAT': lat,
         'sofiaChopR': zeros + 4,
         'sofiaChopS': zeros + 5,
         'PWV': zeros + 6,
         'LOS': zeros + 7,
         'ROLL': zeros + 8,
         'hwpCounts': zeros.astype(int) + 9}
    )
    hdu = fits.BinTableHDU(table)
    hdu.header['EXTNAME'] = 'Timestream'
    return hdu


@pytest.fixture
def valid_frames(no_data_frames, full_hdu):
    frames = no_data_frames.copy()
    frames.scan.is_nonsidereal = True
    frames.configuration.parse_key_value('lab', 'False')
    frames.configuration.parse_key_value('chopper.invert', 'False')
    frames.apply_hdu(full_hdu)
    return frames


@pytest.fixture
def small_integration(no_data_scan, full_hdu):
    scan = no_data_scan
    scan.is_nonsidereal = True
    integration = scan[0]
    c = integration.configuration
    c.parse_key_value('lab', 'False')
    c.parse_key_value('chopper.invert', 'False')
    for key in ['shift', 'frames', 'fillgaps', 'notch', 'vclip', 'aclip',
                'positions.smooth', 'subscan.minlength']:
        c.purge(key)
    integration.frames.data = None
    integration.read([full_hdu])
    integration.channels.census()

    c.set_option('tau', 'pwv')
    c.set_option('tau.pwv', '62.49575959996947')
    c.set_option('tau.pwv.a', '1.0')
    c.set_option('tau.pwv.b', '0.0')
    c.set_option('tau.hawc_plus.a', '0.0020')
    c.set_option('tau.hawc_plus.b', '0.181')
    scan.info.astrometry.horizontal = HorizontalCoordinates([30, 60])
    scan.info.instrument.resolution = 5 * units.Unit('arcsec')
    speed = 10 * units.Unit('arcsec/s')
    integration.average_scan_speed = speed, 1 / speed ** 2
    integration.info.instrument.resolution = 5 * units.Unit('arcsec')
    integration.info.chopping.chopping = False
    return integration


# FIFI-LS simulation


@pytest.fixture
def fifi_channels():
    simulation = FifiLsSimulation()
    header = simulation.create_primary_header()
    info = simulation.info
    info.configuration.read_fits(header)
    info.apply_configuration()
    return info.get_channels_instance()


@pytest.fixture
def fifi_channel_data(fifi_channels):
    return FifiLsChannelData(channels=fifi_channels)


@pytest.fixture
def fifi_initialized_channel_data(fifi_channel_data):
    data = fifi_channel_data.copy()
    data.info.detector_array.initialize_channel_data(data)
    return data


@pytest.fixture
def fifi_simulated_hdul():
    simulation = FifiLsSimulation()
    hdul = simulation.create_simulated_hdul()
    return hdul


@pytest.fixture
def fifi_simulated_reduction(fifi_simulated_hdul):
    reduction = Reduction('fifi_ls')
    reduction.read_scans([fifi_simulated_hdul])
    return reduction


@pytest.fixture
def fifi_simulated_channels(fifi_simulated_reduction):
    return fifi_simulated_reduction.scans[0][0].channels


@pytest.fixture
def fifi_simulated_scan(fifi_simulated_reduction):
    return fifi_simulated_reduction.scans[0]


@pytest.fixture
def fifi_simulated_integration(fifi_simulated_scan):
    return fifi_simulated_scan[0]


@pytest.fixture
def fifi_simulated_frames(fifi_simulated_integration):
    return fifi_simulated_integration.frames
