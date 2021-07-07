# Licensed under a 3-clause BSD style license - see LICENSE.rst

import glob
import os

from astropy import log
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.instruments import fifi_ls
from sofia_redux.instruments.fifi_ls.make_header import make_header

seed = 42


class FIFITestCase(object):
    """Base class for FIFI test cases.

    Sets the log level to DEBUG on test entry.
    """
    @pytest.fixture(autouse=True, scope='function')
    def set_debug_level(self):
        # set log level to debug
        orig_level = log.level
        log.setLevel('DEBUG')
        # let tests run
        yield
        # reset log level
        log.setLevel(orig_level)


# classes used to mock a FIFI data HDU

class MockCols(object):
    def __init__(self):
        self.names = ['DATA', 'HEADER']


class MockData(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.columns = MockCols()


class MockHDU(object):
    def __init__(self):
        self.data = MockData()
        self.header = {}
        self.data['DATA'] = np.zeros((10, 10))
        self.data['HEADER'] = np.chararray((10, 10))


def raw_testdata(nod='A', obsid=None):
    """
    Make raw FIFI-LS data fpr testing purposes

    Returns
    -------
    HDUList
    """
    # seed the random module for consistent tests
    global seed
    rand = np.random.RandomState(seed)
    # modify for next call to ensure some difference between data sets
    seed += 1

    n = 3840
    data = np.recarray(n, dtype=[('header', '>i2', (8,)),
                                 ('data', '>i2', (18, 26))])

    # header array:
    # channel, sample, ramp (index 3, 4, 5) matter
    # for chop split
    data['header'][:, 3] = 1
    if nod == 'A':
        floor = 500
    else:
        floor = 0
    for i in range(n // 32):
        i32 = i * 32
        data['header'][i32:i32 + 32, 4] = np.arange(32)
        data['header'][i32:i32 + 32, 5] = i
        # put a noisy line in each ramp
        data['data'][i32:i32 + 32, :, :] = \
            (250 * rand.rand()
             * np.arange(32) - 20000 + floor)[:, None, None] \
            + rand.randint(10, size=(32, 18, 26))

    # set empty spexel to zero
    data['data'][:, 0, :] = -2**15

    header = fits.Header()
    header['SIMPLE'] = True
    header['EXTEND'] = True

    # different values for A and B
    header['NODBEAM'] = nod
    if nod == 'A':
        if obsid is None:
            header['OBS_ID'] = 'R001'
        else:
            header['OBS_ID'] = obsid
        header['DATE-OBS'] = '2016-03-01T10:38:39'
        header['UTCSTART'] = '10:38:39'
        header['UTCEND'] = '10:38:44'
    else:
        if obsid is None:
            header['OBS_ID'] = 'R002'
        else:
            header['OBS_ID'] = obsid
        header['DATE-OBS'] = '2016-03-01T10:38:51'
        header['UTCSTART'] = '10:38:51'
        header['UTCEND'] = '10:38:56'

    # standardize some defaults
    header = make_header([header])

    # reduction required keys
    header['CHANNEL'] = 'RED'
    header['C_SCHEME'] = '2POINT'
    header['C_AMP'] = 60.0
    header['C_CHOPLN'] = 64
    header['RAMPLN_B'] = 32
    header['RAMPLN_R'] = 32
    header['G_PSUP_B'] = 4
    header['G_PSUP_R'] = 4
    header['G_SZUP_B'] = 200
    header['G_SZUP_R'] = 510
    header['G_STRT_B'] = 713285
    header['G_STRT_R'] = 463923
    header['G_PSDN_B'] = 0
    header['G_PSDN_R'] = 0
    header['G_SZDN_B'] = 0
    header['G_SZDN_R'] = 0
    header['G_CYC_B'] = 1
    header['G_CYC_R'] = 1
    header['C_CYC_B'] = 10
    header['C_CYC_R'] = 10
    header['G_ORD_B'] = 2
    header['PRIMARAY'] = 'BLUE'
    header['DICHROIC'] = 105
    header['NODSTYLE'] = 'NMC'
    header['DLAM_MAP'] = -14.1
    header['DBET_MAP'] = -5.1
    header['DLAM_OFF'] = 0.
    header['DBET_OFF'] = 0.
    header['OBSLAM'] = 0.
    header['OBSBET'] = 0.
    header['OBSRA'] = 0.
    header['OBSDEC'] = 0.
    header['OBJ_NAME'] = 'Mars'
    header['OBJECT'] = 'Mars'
    header['G_WAVE_B'] = 51.807
    header['G_WAVE_R'] = 162.763
    header['TELRA'] = 15.7356
    header['TELDEC'] = -18.4388
    header['PLATSCAL'] = 4.2331334
    header['DET_ANGL'] = 70.0

    # other important keys
    header['AOR_ID'] = '90_0001_01'
    header['MISSN-ID'] = '2016-03-01_FI_F282'
    header['ALTI_STA'] = 41000.
    header['ALTI_END'] = 41000.
    header['ZA_START'] = 45.
    header['ZA_END'] = 45.
    header['LAT_STA'] = 40.
    header['LON_STA'] = -120.
    header['CHPFREQ'] = 1.
    header['DATASRC'] = 'ASTRO'
    header['DETCHAN'] = 'RED'
    header['EXPTIME'] = 2.56
    header['OBSTYPE'] = 'OBJECT'
    header['SPECTEL1'] = 'FIF_BLUE'
    header['SPECTEL2'] = 'FIF_RED'
    header['ALPHA'] = header['EXPTIME'] / n
    header['START'] = 1456857531.0
    header['FIFISTRT'] = 0

    hdulist = fits.HDUList([fits.PrimaryHDU(header=header),
                            fits.BinTableHDU(data)])
    hdulist[1].header['EXTNAME'] = 'FIFILS_rawdata'
    return hdulist


def create_files():
    data_path = os.path.join(os.path.dirname(fifi_ls.__file__),
                             'tests', 'data', '')

    nfiles = 4
    for i in range(nfiles):
        obsid = 'R{:03d}'.format(i + 1)
        if i % 2 == 0:
            nod = 'A'
        else:
            nod = 'B'
        fn = '0000{}_123456_00001_TEST_{}_lw.fits'.format(i + 1, nod)

        hdul = raw_testdata(nod=nod, obsid=obsid)
        hdul[0].header['FILENAME'] = fn

        if i > 1:
            hdul[0].header['DLAM_MAP'] *= -1
            hdul[0].header['DBET_MAP'] *= -1

        datafile = os.path.join(data_path, fn)
        hdul.writeto(datafile, overwrite=True)

    # run default steps in order
    current_dir = os.getcwd()
    os.chdir(data_path)

    input_files = glob.glob(data_path + '00*.fits')
    from sofia_redux.instruments.fifi_ls.split_grating_and_chop \
        import wrap_split_grating_and_chop
    wrap_split_grating_and_chop(input_files, write=True)

    split_files = glob.glob(data_path + '*CP*.fits')
    from sofia_redux.instruments.fifi_ls.fit_ramps \
        import wrap_fit_ramps
    wrap_fit_ramps(split_files, write=True)

    ramp0_files = glob.glob(data_path + '*RP0*.fits')
    from sofia_redux.instruments.fifi_ls.subtract_chops \
        import wrap_subtract_chops
    wrap_subtract_chops(ramp0_files, write=True)

    csb_files = glob.glob(data_path + '*CSB*.fits')
    from sofia_redux.instruments.fifi_ls.combine_nods \
        import combine_nods
    combine_nods(csb_files, write=True)

    ncm_files = glob.glob(data_path + '*NCM*.fits')
    from sofia_redux.instruments.fifi_ls.lambda_calibrate \
        import wrap_lambda_calibrate
    wrap_lambda_calibrate(ncm_files, write=True)

    wav_files = glob.glob(data_path + '*WAV*.fits')
    from sofia_redux.instruments.fifi_ls.spatial_calibrate \
        import wrap_spatial_calibrate
    wrap_spatial_calibrate(wav_files, rotate=True, write=True)

    xyc_files = glob.glob(data_path + '*XYC*.fits')
    from sofia_redux.instruments.fifi_ls.apply_static_flat \
        import wrap_apply_static_flat
    wrap_apply_static_flat(xyc_files, write=True)

    flf_files = glob.glob(data_path + '*FLF*.fits')
    from sofia_redux.instruments.fifi_ls.combine_grating_scans \
        import wrap_combine_grating_scans
    wrap_combine_grating_scans(flf_files, write=True)

    scm_files = glob.glob(data_path + '*SCM*.fits')
    from sofia_redux.instruments.fifi_ls.telluric_correct \
        import wrap_telluric_correct
    wrap_telluric_correct(scm_files, write=True)

    tel_files = glob.glob(data_path + '*TEL*.fits')
    from sofia_redux.instruments.fifi_ls.flux_calibrate \
        import wrap_flux_calibrate
    wrap_flux_calibrate(tel_files, write=True)

    cal_files = glob.glob(data_path + '*CAL*.fits')
    from sofia_redux.instruments.fifi_ls.correct_wave_shift \
        import wrap_correct_wave_shift
    wrap_correct_wave_shift(cal_files, write=True)

    wsh_files = glob.glob(data_path + '*WSH*.fits')
    from sofia_redux.instruments.fifi_ls.resample import resample
    resample(wsh_files, write=True)

    # cd back to original directory
    os.chdir(current_dir)


def test_files(prodtype=None):
    data_path = os.path.join(os.path.dirname(fifi_ls.__file__),
                             'tests', 'data')

    raw_files = glob.glob('{}/0*.fits'.format(data_path))

    if len(raw_files) == 0:
        create_files()
        raw_files = glob.glob('{}/0*.fits'.format(data_path))
    else:
        # check that version matches current
        pipevers = fits.getval(raw_files[0], 'PIPEVERS')
        if pipevers != fifi_ls.__version__.replace('.', '_'):
            # if not, recreate the files
            for filename in glob.glob('{}/*.fits'.format(data_path)):
                os.remove(filename)
            create_files()
            raw_files = glob.glob('{}/0*.fits'.format(data_path))

    if prodtype is None:
        test_file_names = raw_files
    else:
        test_file_names = glob.glob('{}/*{}*.fits'.format(data_path, prodtype))
    return sorted(test_file_names)


def get_split_files():
    return test_files('CP')


def get_chop0_file():
    return test_files('RP0')[0]


def get_chop_files():
    return test_files('RP0')


def get_csb_files():
    return test_files('CSB')


def get_ncm_files():
    return test_files('NCM')


def get_wav_files():
    return test_files('WAV')


def get_xyc_files():
    return test_files('XYC')


def get_flf_files():
    return test_files('FLF')


def get_scm_files():
    return test_files('SCM')


def get_tel_files():
    return test_files('TEL')


def get_cal_files():
    return test_files('CAL')


def get_wsh_files():
    return test_files('WSH')


def get_wxy_files():
    return test_files('WXY')
