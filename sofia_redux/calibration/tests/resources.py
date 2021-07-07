# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from astropy.stats import gaussian_fwhm_to_sigma
import numpy as np


def basic_data():
    """Synthetic test data with positive and negative Gaussian peaks."""
    # parameters
    nx, ny = 500, 500
    angle = 30.0
    dx = -40.17
    dy = 69.57
    mfwhm = 4.5

    header = fits.header.Header()
    header['NAXIS1'] = nx
    header['NAXIS2'] = ny
    header['TELESCOP'] = 'SOFIA'
    header['AOR_ID'] = '90_0001_01'
    header['ALTI_STA'] = 40000
    header['ALTI_END'] = 40005
    header['ZA_START'] = 44.0
    header['ZA_END'] = 44.5
    header['WVZ_STA'] = 4.0
    header['WVZ_END'] = 5.0
    header['OBSTYPE'] = 'STANDARD_FLUX'
    header['CRPIX1'] = nx // 2
    header['CRPIX2'] = ny // 2
    header['CROTA2'] = 180 - angle

    # generic instrument keys
    header['INSTRUME'] = 'UNKNOWN'
    header['SPECTEL1'] = 'UNKNOWN'
    header['SPECTEL2'] = 'UNKNOWN'
    header['INSTCFG'] = 'UNKNOWN'
    header['OBJECT'] = 'UNKNOWN'
    header['DATE-OBS'] = 'UNKNOWN'

    # build data
    npeaks = 5
    mid = npeaks // 2
    cx, cy = nx // 2, ny // 2
    header['SRCPOSX'] = cx
    header['SRCPOSY'] = cy
    peaks = []
    for i in range(npeaks):
        peaks.append([cx + dx * (i - mid), cy + dy * (i - mid)])
    peaks = np.array(peaks)
    sigma = mfwhm * gaussian_fwhm_to_sigma
    data = np.zeros((ny, nx))
    y, x = np.mgrid[:ny, :nx]
    mult = 1
    amplitude = 10.0
    gp = {'amplitude': amplitude, 'x_stddev': sigma, 'y_stddev': sigma}
    for idx, peak in enumerate(peaks):
        gp['x_mean'] = peak[0]
        gp['y_mean'] = peak[1]
        gp['amplitude'] = amplitude * mult
        if idx == 0 or idx == 4:
            gp['amplitude'] /= 2
        g = Gaussian2D(**gp)
        data += g(x, y)
        mult *= -1
    data += 1e-6
    errdata = np.full_like(data, 1.0)

    primary = fits.PrimaryHDU(data=data,
                              header=header)
    err = fits.ImageHDU(errdata, name='ERROR')
    hdul = fits.HDUList([primary, err])

    return hdul


def flitecam_data():
    """Synthetic FLITECAM test data."""
    hdul = basic_data()
    header = hdul[0].header

    # add FLITECAM keys
    header['INSTRUME'] = 'FLITECAM'
    header['INSTCFG'] = 'IMAGING'
    header['INSTMODE'] = 'NOD_OFFARRAY'
    header['SPECTEL1'] = 'FLT_H'
    header['SPECTEL2'] = 'NONE'

    header['OBJECT'] = 'SA92336'
    header['DATE-OBS'] = '2018-12-31T23:59:59.999'
    header['MISSN-ID'] = '2018-12-31_FC_F001'

    # add error as second plane, instead of extension
    primary = hdul[0]
    err_data = hdul[1].data
    stacked = np.array([primary.data, err_data])
    primary.data = stacked
    primary.header['NAXIS3'] = 2
    new_hdul = fits.HDUList(primary)
    return new_hdul


def flipo_data():
    """Synthetic FLITECAM test data, in FLIPO config."""
    hdul = flitecam_data()
    header = hdul[0].header

    # change the mission id
    header['MISSN-ID'] = '2018-12-31_FP_F001'

    return hdul


def flitecam_new_data():
    hdul = flitecam_data()

    # separate error into new extension
    cube = hdul[0].data
    hdul[0].data = cube[0]
    hdul[0].header['EXTNAME'] = 'FLUX'
    hdul.append(fits.ImageHDU(cube[1]))
    hdul[1].header['EXTNAME'] = 'ERROR'
    return hdul


def forcast_data():
    """Synthetic FORCAST test data."""
    hdul = basic_data()
    header = hdul[0].header

    # add FORCAST keys
    header['INSTRUME'] = 'FORCAST'
    header['INSTCFG'] = 'IMAGING_SWC'
    header['INSTMODE'] = 'C2N'
    header['SKYMODE'] = 'NMC'
    header['SPECTEL1'] = 'FOR_F197'
    header['SPECTEL2'] = 'NONE'
    header['DETCHAN'] = 'SW'
    header['SLIT'] = 'NONE'
    header['DICHROIC'] = 'Mirror (swc)'

    header['OBJECT'] = 'Alpha Boo'
    header['DATE-OBS'] = '2018-12-31T23:59:59.999'
    header['MISSN-ID'] = '2018-12-31_FO_F001'

    # add an exposure map
    exp = hdul[1].copy()
    exp.header['EXTNAME'] = 'EXPOSURE'
    exp.data[:, :] = 1.0
    hdul.append(exp)

    return hdul


def forcast_legacy_data():
    """Synthetic FORCAST data from early pipeline versions (<2.0)"""
    hdul = forcast_data()

    # add variance as second plane, instead of error extension
    primary = hdul[0]
    err_data = hdul[1].data
    exp_data = hdul[2].data
    stacked = np.array([primary.data, err_data**2, exp_data])
    primary.data = stacked
    primary.header['NAXIS3'] = 2
    new_hdul = fits.HDUList(primary)
    return new_hdul


def hawc_pol_data():
    """Synthetic HAWC+ polarimetry data"""
    hdul = basic_data()
    header = hdul[0].header

    # add HAWC keys
    header['INSTRUME'] = 'HAWC_PLUS'
    header['INSTCFG'] = 'POLARIZATION'
    header['INSTMODE'] = 'C2N (NMC)'
    header['SPECTEL1'] = 'HAW_D'
    header['SPECTEL2'] = 'HAW_HWP_D'

    header['OBJECT'] = 'Uranus'
    header['DATE-OBS'] = '2019-02-20T02:00:00.000'
    header['MISSN-ID'] = '2019-02-20_HA_F001'

    new_hdul = fits.HDUList()
    for i, stokes in enumerate(['I', 'Q', 'U']):
        im = hdul[0].copy()
        im.header['EXTNAME'] = 'STOKES {}'.format(stokes)
        err = hdul[1].copy()
        err.header['EXTNAME'] = 'ERROR {}'.format(stokes)
        if i == 0:
            new_hdul.append(im)
        else:
            new_hdul.append(fits.ImageHDU(im.data, im.header))
        new_hdul.append(err)

    return new_hdul


def hawc_im_data():
    """Synthetic HAWC+ scanning imaging data"""
    hdul = basic_data()
    header = hdul[0].header

    # add HAWC keys
    header['INSTRUME'] = 'HAWC_PLUS'
    header['INSTCFG'] = 'TOTAL_INTENSITY'
    header['INSTMODE'] = 'OTFMAP'
    header['SPECTEL1'] = 'HAW_D'
    header['SPECTEL2'] = 'HAW_HWP_Open'

    header['OBJECT'] = 'Uranus'
    header['DATE-OBS'] = '2019-02-20T02:00:00.000'
    header['MISSN-ID'] = '2019-02-20_HA_F001'

    new_hdul = fits.HDUList()

    im = hdul[0].copy()
    im.header['EXTNAME'] = 'PRIMARY IMAGE'
    new_hdul.append(im)

    im = hdul[1].copy()
    im.header['EXTNAME'] = 'EXPOSURE'
    new_hdul.append(im)

    im = hdul[1].copy()
    im.header['EXTNAME'] = 'NOISE'
    new_hdul.append(im)

    im = hdul[1].copy()
    im.header['EXTNAME'] = 'S/N'
    new_hdul.append(im)

    return new_hdul
