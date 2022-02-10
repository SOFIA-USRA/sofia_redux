# Licensed under a 3-clause BSD style license - see LICENSE.rst

import datetime as dt
import os
import re

from astropy import log
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from astropy.stats import gaussian_fwhm_to_sigma
import numpy as np
import pandas as pd
import pytest

from sofia_redux.instruments import hawc


class DRPTestCase(object):
    """
    Base class for DRP test cases.

    Sets/resets the DPS_HAWCPIPE environment variable for
    default configuration for each test setup/teardown.
    """
    @pytest.fixture(autouse=True, scope='function')
    def set_debug_level(self):
        # set hawc env
        try:
            old_pipe = os.environ['DPS_HAWCPIPE']
        except KeyError:
            old_pipe = None
        os.environ['DPS_HAWCPIPE'] = os.path.dirname(hawc.__file__)

        # set log level to debug
        orig_level = log.level
        log.setLevel('DEBUG')

        # let tests run
        yield

        # reset log level and env
        log.setLevel(orig_level)
        if old_pipe is None:
            try:
                del os.environ['DPS_HAWCPIPE']
            except KeyError:
                pass
        else:
            os.environ['DPS_HAWCPIPE'] = old_pipe


def basic_header():
    header = fits.header.Header()

    # required for successful reduction
    header['TELESCOP'] = 'SOFIA'
    header['INSTRUME'] = 'HAWC_PLUS'
    header['OBJECT'] = 'Uranus'
    header['OBSTYPE'] = 'OBJECT'
    header['CALMODE'] = 'UNKNOWN'
    header['SRCTYPE'] = 'point_source'
    # set the date well in the future so no date configs apply
    header['OBS_ID'] = '2100-02-20_HA_F001_001'
    header['DATE-OBS'] = '2100-02-20T02:00:00.000'
    header['UTCSTART'] = '02:00:00.000'
    header['MISSN-ID'] = '2100-02-20_HA_F001'
    header['SPECTEL1'] = 'HAW_D'
    header['PIXSCAL'] = 6.93
    header['SIBS_X'] = 14.5
    header['SIBS_Y'] = 15.5
    header['TELRA'] = 22.740806
    header['TELDEC'] = -8.915143
    header['OBSRA'] = -9999
    header['OBSDEC'] = -9999
    header['TELVPA'] = 194.9

    header['EXPTIME'] = 111.0
    header['AOR_ID'] = '90_0001_01'
    header['ALTI_STA'] = 40000
    header['ALTI_END'] = 40005
    header['ZA_START'] = 44.0
    header['ZA_END'] = 44.5
    header['WVZ_STA'] = 4.0
    header['WVZ_END'] = 5.0
    header['EQUINOX'] = 2000.0
    header['SMPLFREQ'] = 203.2520325203252
    header['MCEMAP'] = '0,2,1,-1'

    # for header compliance
    header['CHOPPING'] = True
    header['NODDING'] = True
    header['DITHER'] = True
    header['FOCUS_ST'] = 370.
    header['FOCUS_EN'] = 370.
    header['SCRIPTID'] = '12345'
    header['FILEGPID'] = '12345'

    # basic chop/nod pol data
    header['INSTCFG'] = 'POLARIZATION'
    header['INSTMODE'] = 'C2N (NMC)'
    header['SPECTEL2'] = 'HAW_HWP_D'
    header['NHWP'] = 4
    header['HWPSTART'] = 5.0
    header['CHPAMP1'] = 100.0
    header['CHPANGLE'] = 90.0
    header['CHPCRSYS'] = 'sirf'
    header['CHPFREQ'] = 10.2
    header['CHPONFPA'] = False
    header['DTHINDEX'] = 0
    header['DTHSCALE'] = 3.0
    header['DTHXOFF'] = -3.0
    header['DTHYOFF'] = -3.0
    header['NODPATT'] = 'ABBA'

    return header


def basic_raw_data(nframe=320, smplfreq=2.0):
    header = basic_header()
    header['EXTNAME'] = 'PRIMARY'
    header['CHPFREQ'] = 1.0
    header['SMPLFREQ'] = smplfreq
    header['EXPTIME'] = nframe / header['SMPLFREQ']

    # primary and config extensions
    primary = fits.PrimaryHDU(header=header, data=np.array([]))
    configuration = fits.ImageHDU(data=np.array([0], dtype=np.int32),
                                  name='Configuration')

    # some useful indices
    div32 = []
    if smplfreq > 1:
        tot = 32
    else:
        tot = 16
    for i in range(tot + 1):
        div32.append(nframe * i // tot)

    # seed the random module for consistent tests
    rand = np.random.RandomState(42)

    # flux data
    baseval = 1000.
    data = rand.normal(baseval, baseval, (nframe, 41, 128))
    feedback = fits.Column(name='SQ1Feedback', array=data,
                           format='5248J', dim='(128,41)')

    # jumps -- set to zero
    jumps = fits.Column(name='FluxJumps',
                        array=np.zeros((nframe, 41, 128)),
                        format='5248I', dim='(128,41)', unit='jumps')

    # frame counter - set to index array
    counter = fits.Column(name='FrameCounter', array=np.arange(nframe),
                          format='1K', unit='frames')

    # 4 HWP angles
    hwp = np.zeros(nframe)
    hwp[div32[0]:div32[4]] = 5.0 * 4.0
    hwp[div32[4]:div32[8]] = 50.0 * 4.0
    hwp[div32[8]:div32[12]] = 27.0 * 4.0
    hwp[div32[12]:div32[16]] = 72.0 * 4.0
    # add a bad data margin at the beginning
    hwp[0:3] = 72.0 * 4.0
    hwp = fits.Column(name='hwpCounts', array=hwp, format='1J', unit='counts')

    # time stamp, lst, lat, lon
    start = re.split(r'[:\-T]', header['DATE-OBS'])
    start_time = dt.datetime(*[int(float(s)) for s in start]).timestamp()
    inttime = 1. / header['SMPLFREQ']
    timestamp = np.full(nframe, start_time) + np.arange(nframe) * inttime
    timestamp = fits.Column(name='Timestamp', array=timestamp, format='1D',
                            unit='seconds')

    lst = np.full(nframe, 23.35778) + np.arange(nframe) * 0.026 / nframe
    lst = fits.Column(name='LST', array=lst, format='1D', unit='hours')

    lat = np.full(nframe, 38.0535) + np.arange(nframe) * 0.1459 / nframe
    lat = fits.Column(name='LAT', array=lat, format='1D', unit='degrees')

    lon = np.full(nframe, -130.467) + np.arange(nframe) * -0.5382 / nframe
    lon = fits.Column(name='LON', array=lon, format='1D', unit='degrees')

    # ra, dec
    # on nods
    ra_on = 22.7429
    ra_off = 22.7396
    dec_on = -8.9175
    dec_off = -8.9045
    ra = np.full(nframe, ra_on)
    dec = np.full(nframe, dec_on)
    # off nods - ABBA
    ra[div32[1]:div32[3]] = ra_off
    ra[div32[5]:div32[7]] = ra_off
    ra[div32[9]:div32[11]] = ra_off
    ra[div32[13]:div32[15]] = ra_off
    dec[div32[1]:div32[3]] = dec_off
    dec[div32[5]:div32[7]] = dec_off
    dec[div32[9]:div32[11]] = dec_off
    dec[div32[13]:div32[15]] = dec_off
    # add noise
    ra += rand.rand(nframe) * 1e-4
    dec += rand.rand(nframe) * 1e-4
    # add zeros at transitions
    ra[div32[:-1]] = header['TELRA']
    dec[div32[:-1]] = header['TELDEC']

    nsra = fits.Column(name='NonSiderealRA', array=ra, format='1D',
                       unit='hours')
    nsdec = fits.Column(name='NonSiderealDec', array=dec, format='1D',
                        unit='degrees')
    ra = fits.Column(name='RA', array=ra, format='1D', unit='hours')
    dec = fits.Column(name='DEC', array=dec, format='1D', unit='degrees')

    # az, el, vpa, pwv, los, roll, temperature
    az = fits.Column(name='AZ', array=np.full(nframe, 193.0),
                     format='1D', unit='degrees')
    az_er = fits.Column(name='AZ_Error', array=np.zeros(nframe),
                        format='1D', unit='degrees')
    el = fits.Column(name='EL', array=np.full(nframe, 43.0),
                     format='1D', unit='degrees')
    el_er = fits.Column(name='EL_Error', array=np.zeros(nframe),
                        format='1D', unit='degrees')
    vpa = fits.Column(name='SIBS_VPA', array=np.full(nframe, 14.643),
                      format='1D', unit='degrees')
    tvpa = fits.Column(name='TABS_VPA', array=np.full(nframe, 14.938),
                       format='1D', unit='degrees')
    cvpa = fits.Column(name='Chop_VPA', array=np.full(nframe, 14.942),
                       format='1D', unit='degrees')
    pwv = fits.Column(name='PWV', array=np.full(nframe, 19.7),
                      format='1D', unit='um')
    los = fits.Column(name='LOS', array=np.full(nframe, -1.244),
                      format='1D', unit='degrees')
    roll = fits.Column(name='ROLL', array=np.full(nframe, 0.627),
                       format='1D', unit='degrees')
    ai23 = fits.Column(name='ai23', array=np.full(nframe, 3.342),
                       format='1D', unit='volts')

    # on nods
    nod = np.full(nframe, 90.0)
    # off nods
    nod[div32[1]:div32[3]] *= -1
    nod[div32[5]:div32[7]] *= -1
    nod[div32[9]:div32[11]] *= -1
    nod[div32[13]:div32[15]] *= -1
    # add zeros at transitions
    nod[div32[:-1]] = 0.0
    nod_off = fits.Column(name='NOD_OFF', array=nod, format='1D',
                          unit='arcsec')

    # chops
    chopr = fits.Column(name='sofiaChopR',
                        array=rand.rand(nframe) * 1e-5 - 1e-5,
                        format='1E', unit='volts')
    chop = rand.rand(nframe) * 1e-5 + 3.0
    if tot == 32:
        chop[::4] *= -1.0
        chop[1::4] *= -1.0
    else:
        chop[::2] *= -1.0
    chops = fits.Column(name='sofiaChopS', array=chop, format='1E',
                        unit='volts')
    chopsync = fits.Column(name='sofiaChopSync', array=chop, format='1E',
                           unit='volts')
    criochop = fits.Column(name='crioAnalogChopOut', array=chop, format='1E',
                           unit='volts')

    # flag - zero is normal
    flag = fits.Column(name='Flag', array=np.zeros(nframe), format='1J',
                       unit='flag')

    # tracking errors: small random noise, jumps at nod transitions
    track = rand.rand(nframe) * 0.5
    track[div32[:-1]] = 100.0
    track3 = fits.Column(name='TrackErrAoi3', array=track, format='1D',
                         unit='arcsec')
    track4 = fits.Column(name='TrackErrAoi4', array=track, format='1D',
                         unit='arcsec')

    ctrack = rand.rand(nframe) + 20.0
    ctrack[div32[:-1]] = 2.0
    cent = fits.Column(name='CentroidExpMsec', array=ctrack, format='1D',
                       unit='millisecond')

    timestream = fits.BinTableHDU.from_columns(
        [feedback, jumps, counter, hwp,
         chopr, chops, chopsync, criochop,
         ai23, timestamp,
         ra, dec, nsra, nsdec,
         az, el, az_er, el_er, vpa, tvpa, cvpa,
         pwv, los, roll, lat, lon, lst, nod_off, flag,
         track3, track4, cent],
        name='Timestream')

    hdul = fits.HDUList([primary, configuration, timestream])
    return hdul


def pol_raw_data(nframe=320):
    return basic_raw_data(nframe=nframe)


def scan_raw_data(nframe=90):
    # modify raw data to scan in RA/Dec
    hdul = basic_raw_data(nframe=nframe, smplfreq=1)
    header = hdul[0].header
    header['INSTCFG'] = 'TOTAL_INTENSITY'
    header['INSTMODE'] = 'OTFMAP'
    header['SCANNING'] = True

    timestream = hdul[2].data
    ra = timestream['RA']
    dec = timestream['DEC']
    nframe = ra.size
    timestream['RA'] = np.full(nframe, ra[0]) + np.arange(nframe) * 5e-5
    timestream['DEC'] = np.full(nframe, dec[0]) + np.arange(nframe) * 5e-5

    hdul[2].data = timestream
    return hdul


def add_col(hdul, colname, copy_from, fill=None):
    tabhdu = hdul[2]
    new_data = tabhdu.data[copy_from].copy()
    if fill is not None:
        new_data[:] = fill

    idx = tabhdu.data.names.index(copy_from)
    fmt = tabhdu.data.columns[idx].format

    new_col = fits.Column(name=colname,
                          array=new_data, format=fmt)

    new_hdu = fits.BinTableHDU.from_columns(tabhdu.columns
                                            + fits.ColDefs([new_col]))
    hdul[2] = new_hdu
    return hdul


def del_col(hdul, colname):
    tabhdu = hdul[2]
    new_col = []
    for col in tabhdu.data.columns:
        if col.name != colname:
            new_col.append(col)
    new_hdu = fits.BinTableHDU.from_columns(new_col)
    hdul[2] = new_hdu
    return hdul


def intcal_raw_data(nframe=80):
    hdul = basic_raw_data(nframe=nframe)

    # mark as intcal
    hdul[0].header['CALMODE'] = 'INT_CAL'
    return hdul


def basic_reduced_data(x_off=0.0, y_off=0.0):
    """Synthetic test data with one Gaussian peak."""
    # parameters
    nx, ny = 64, 40
    mfwhm = 2.0

    header = basic_header()
    header['NAXIS1'] = nx
    header['NAXIS2'] = ny
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CRPIX1'] = nx / 4. + x_off
    header['CRPIX2'] = ny / 2. + y_off
    header['CRVAL1'] = 341.143635
    header['CRVAL2'] = -8.917363999
    header['CDELT1'] = 0.001925
    header['CDELT2'] = 0.001925
    header['CROTA2'] = -194.6432304

    # seed the random module for consistent tests
    rand = np.random.RandomState(42)

    sigma = mfwhm * gaussian_fwhm_to_sigma
    data = rand.rand(ny, nx) * 50
    y, x = np.mgrid[:ny, :nx]
    amplitude = 6000.0
    gp = {'amplitude': amplitude,
          'x_mean': nx / 4. + x_off,
          'y_mean': ny / 2. + y_off,
          'x_stddev': sigma,
          'y_stddev': sigma}
    g = Gaussian2D(**gp)
    data += g(x, y)

    # peak the error a little over the source
    errdata = rand.rand(ny, nx) * 50
    gp['amplitude'] *= 0.05
    g = Gaussian2D(**gp)
    errdata += g(x, y)

    primary = fits.PrimaryHDU(data=data,
                              header=header)
    err = fits.ImageHDU(errdata, name='ERROR')
    hdul = fits.HDUList([primary, err])

    return hdul


def pol_bgs_data(idx=0, empty=False):
    """Synthetic HAWC+ polarimetry data, ready for merge."""
    if idx % 4 == 0:
        x_off = 0.0
        y_off = 0.0
    elif idx % 4 == 1:
        x_off = 2.0
        y_off = 0.0
    elif idx % 4 == 2:
        x_off = 2.0
        y_off = 2.0
    else:
        x_off = 0.0
        y_off = 2.0

    # seed the random module for consistent tests
    rand = np.random.RandomState(42)

    hdul = basic_reduced_data(x_off=x_off, y_off=y_off)
    if empty:
        hdul[0].data = rand.rand(*hdul[0].data.shape) * 10 + 50.
    header = hdul[0].header

    # add configuration keys
    header['INSTCFG'] = 'POLARIZATION'
    header['INSTMODE'] = 'C2N (NMC)'
    header['SPECTEL2'] = 'HAW_HWP_D'
    header['NHWP'] = 4
    header['PRODTYPE'] = 'bgsubtract'
    header['ALNGAPX'] = 4.0
    header['ALNGAPY'] = 0.0
    header['ALNROTA'] = 0.0

    new_hdul = fits.HDUList()
    for i, stokes in enumerate(['I', 'Q', 'U']):
        im = hdul[0].copy()
        im.data[:, 32:] = np.nan
        im.header['EXTNAME'] = 'STOKES {}'.format(stokes)
        err = hdul[1].copy()
        err.data[:, 32:] = np.nan
        err.header['EXTNAME'] = 'ERROR {}'.format(stokes)
        if i == 0:
            new_hdul.append(im)
        else:
            if stokes == 'Q':
                im.data *= -0.02
            else:
                im.data *= 0.02
            new_hdul.append(fits.ImageHDU(im.data, im.header))
        new_hdul.append(err)

    # also append covar
    covar = hdul[1].copy()
    covar.data[:, 32:] = np.nan
    covar.data = covar.data**2
    covar.header['EXTNAME'] = 'COVAR Q I'
    new_hdul.append(covar)
    covar = covar.copy()
    covar.header['EXTNAME'] = 'COVAR U I'
    new_hdul.append(covar)
    covar = covar.copy()
    covar.header['EXTNAME'] = 'COVAR Q U'
    new_hdul.append(covar)

    # and a bad pixel mask
    bpm = hdul[1].copy()
    bpm.data = bpm.data.astype(np.int32)
    bpm.data[:, 0:32] = 0
    bpm.data[:, 32:] = 3
    bpm.header['EXTNAME'] = 'BAD PIXEL MASK'
    new_hdul.append(bpm)

    return new_hdul


def scan_smp_data():
    """Synthetic HAWC+ image scan data, as if from scanmap."""
    hdul = basic_reduced_data()

    primary = hdul[0].copy()
    primary.header['INSTCFG'] = 'TOTAL_INTENSITY'
    primary.header['INSTMODE'] = 'OTFMAP'
    primary.header['SCANNING'] = True

    expmap = hdul[1].copy()
    expmap.header['EXTNAME'] = 'EXPOSURE'
    noise = hdul[1].copy()
    noise.header['EXTNAME'] = 'NOISE'
    sn = hdul[1].copy()
    sn.header['EXTNAME'] = 'S/N'

    new_hdul = fits.HDUList([primary, expmap, noise, sn])
    return new_hdul


def scanpol_crh_data():
    """Synthetic HAWC+ scanpol data, as if from scanmap."""
    hdul = basic_reduced_data()

    primary = hdul[0].copy()
    primary.header['INSTCFG'] = 'POLARIZATION'
    primary.header['INSTMODE'] = 'OTFMAP'
    primary.header['SCANNING'] = True
    primary.header['EXTNAME'] = 'DATA R HWP0'

    data = {'R': primary.data,
            'T': primary.data * 0.9}
    noise = hdul[1].data
    header = primary.header

    new_hdul = fits.HDUList(primary)
    angle = [5.0, 50.0, 27.0, 72.0]
    first = True
    for i, a in enumerate(angle):
        for typ in ['DATA', 'ERROR', 'EXPOSURE']:
            for ext in ['R', 'T']:
                if first:
                    new_hdul[0].header['HWPINIT'] = a
                    first = False
                else:
                    hdr = header.copy()
                    hdr['HWPINIT'] = a
                    if typ == 'DATA':
                        d = data[ext].copy()
                    else:
                        d = noise.copy()
                    hdu = fits.ImageHDU(d, header=hdr,
                                        name=f'{typ} {ext} HWP{i}')
                    new_hdul.append(hdu)

    return new_hdul


def flat_data(rval=1.0, tval=2.0, seed=42):
    """Synthetic HAWC+ flat data."""
    rand = np.random.RandomState(seed)
    hdul = basic_reduced_data()

    primary = hdul[0].copy()
    primary.header['EXTNAME'] = 'R ARRAY GAIN'
    primary.data = rand.rand(40, 64) + rval

    tgain = fits.ImageHDU(header=fits.Header())
    tgain.header['EXTNAME'] = 'T ARRAY GAIN'
    tgain.data = rand.rand(40, 64) + tval

    rbad = fits.ImageHDU(header=fits.Header())
    rbad.header['EXTNAME'] = 'R BAD PIXEL MASK'
    rbad.data = rand.choice([0, 1], size=(40, 64))

    tbad = fits.ImageHDU(header=fits.Header())
    tbad.header['EXTNAME'] = 'T BAD PIXEL MASK'
    tbad.data = rand.choice([0, 2], size=(40, 64))

    new_hdul = fits.HDUList([primary, tgain, rbad, tbad])
    return new_hdul


def pixel_data(scan='File-001', seed=42):
    """Synthetic HAWC+ gain data in scanmap format."""
    rand = np.random.RandomState(seed)
    npix = 41 * 32 * 3
    gain = rand.rand(npix) + 1.0
    weight = np.ones(npix, dtype=float)
    flag = rand.choice(['B', 'n', 'g', '-'], npix)
    ch, idx, sub, row, col = [], [], [], [], []
    arrays = ['R0', 'R1', 'T0']
    n = 0
    for i in range(3):
        for j in range(41):
            for k in range(32):
                ch.append(f'{arrays[i]}[{j},{k}]')
                idx.append(n)
                sub.append(i)
                row.append(j)
                col.append(k)
                n += 1

    header = f"# Scan: {scan}\n#\n" \
             f"# ch	gain	weight		flag	eff	Gmux	idx	sub	row	col\n"

    ptable = pd.DataFrame({'ch': ch, 'gain': gain, 'weight': weight,
                           'flag': flag, 'eff': weight, 'Gmux1': weight,
                           'Gmux2': weight, 'idx': idx, 'sub': sub,
                           'row': row, 'col': col})
    ptable['gain'] = ptable['gain'].map('{:.3f}'.format)

    return header, ptable
