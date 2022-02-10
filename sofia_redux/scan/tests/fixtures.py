# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
from configobj import ConfigObj
import numpy as np
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.example.info.info import ExampleInfo
from sofia_redux.scan.custom.example.scan.scan import ExampleScan
from sofia_redux.scan.custom.hawc_plus.info.info import HawcPlusInfo
from sofia_redux.scan.custom.hawc_plus.scan.scan import HawcPlusScan
from sofia_redux.scan.custom.hawc_plus.simulation.simulation \
    import HawcPlusSimulation
from sofia_redux.scan.reduction.reduction import Reduction


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
