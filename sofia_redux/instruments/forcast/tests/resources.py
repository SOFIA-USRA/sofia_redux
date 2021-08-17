# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from astropy.stats import gaussian_fwhm_to_sigma
from sofia_redux.instruments.forcast.imgshift_header import imgshift_header
import numpy as np


def npc_testdata():
    nx, ny = 500, 500
    dr = 80.0 + 1 / 3  # for fractional testing
    angle = 30.0
    header = fits.header.Header()

    header['NAXIS1'] = nx
    header['NAXIS2'] = ny
    header['SPECTEL1'] = 'FOR_F197'
    header['SPECTEL2'] = 'NONE'
    header['DETCHAN'] = 'SW'
    header['SLIT'] = 'NONE'

    header['INSTMODE'] = 'C2N'
    header['TELESCOP'] = 'PIXELS'
    header['MFWHM'] = 4.5
    header['BORDER'] = 0
    header['NAXIS1'] = nx
    header['NAXIS2'] = ny
    header['CRPIX1'] = nx // 2
    header['CRPIX2'] = ny // 2
    header['ANGLCONV'] = 'negative'
    header['SKY_ANGL'] = angle
    header['CROTA2'] = 180 - angle
    header['CHOPPING'] = True
    header['NODDING'] = True
    header['CHPAMP1'] = dr / 2
    header['NODAMP'] = dr
    header['CHPANGLR'] = 330
    header['NODANGLR'] = 240
    header['CHPANGLE'] = -330
    header['NODANGLE'] = 120
    header['CHPCRSYS'] = 'array'
    header['NODCRSYS'] = 'EQUATORIAL'
    header['CHPCOORD'] = 0
    header['NODCOORD'] = 0
    header['DITHER'] = True
    header['DITHERX'] = 1.0
    header['DITHERY'] = 1.0
    shifts = imgshift_header(header)
    peaks, sign = [], []
    cx, cy = nx // 2, ny // 2
    nodx, nody = shifts['nodx'] / 2, shifts['nody'] / 2
    chopx, chopy = shifts['chopx'] / 2, shifts['chopy'] / 2
    header['SRCPOSX'] = cx + nodx + chopx
    header['SRCPOSY'] = cy + nody + chopy
    mult = 1
    for chop in [(chopx, chopy), (-chopx, -chopy)]:
        mult *= -1
        for nod in [(nodx, nody), (-nodx, -nody)]:
            mult *= -1
            peaks.append((cx + chop[0] + nod[0], cy + chop[1] + nod[1]))
            sign.append(mult)

    peaks = np.array(peaks)
    sigma = header['MFWHM'] * gaussian_fwhm_to_sigma
    data = np.zeros((ny, nx))
    y, x = np.mgrid[:ny, :nx]
    gp = {'amplitude': 10, 'x_stddev': sigma, 'y_stddev': sigma}
    for mult, peak in zip(sign, peaks):
        gp['x_mean'] = peak[0]
        gp['y_mean'] = peak[1]
        gp['amplitude'] = mult * 10
        g = Gaussian2D(**gp)
        data += g(x, y)
    return {'data': data + 1e-6, 'header': header, 'peaks': peaks}


def nmc_testdata():
    nx, ny = 500, 500
    dr = 80.0 + 1 / 3  # for fractional testing
    angle = 30.0
    header = fits.header.Header()

    header['NAXIS1'] = nx
    header['NAXIS2'] = ny
    header['SPECTEL1'] = 'FOR_F197'
    header['SPECTEL2'] = 'NONE'
    header['DETCHAN'] = 'SW'
    header['SLIT'] = 'NONE'

    header['INSTMODE'] = 'NMC'
    header['TELESCOP'] = 'PIXELS'
    header['MFWHM'] = 4.5
    header['BORDER'] = 0
    header['CRPIX1'] = nx // 2
    header['CRPIX2'] = ny // 2
    header['CRVAL1'] = 50.0
    header['CRVAL2'] = 50.0
    header['CDELT1'] = -1
    header['CDELT2'] = 1
    header['ANGLCONV'] = 'negative'
    header['SKY_ANGL'] = angle
    header['CROTA2'] = 180 - angle
    header['CHOPPING'] = True
    header['NODDING'] = True
    header['CHPAMP1'] = dr / 2
    header['NODAMP'] = dr
    header['CHPANGLE'] = -330
    header['NODANGLE'] = 210
    header['CHPANGLR'] = 330
    header['NODANGLR'] = 330
    header['CHPCRSYS'] = 'SIRF'
    header['NODCRSYS'] = 'SIRF'
    header['CHPCOORD'] = 0
    header['NODCOORD'] = 0
    header['DITHER'] = True
    header['DITHERX'] = 1.0
    header['DITHERY'] = 1.0
    shifts = imgshift_header(header)
    dx = shifts['chopx']
    dy = shifts['chopy']

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
    sigma = header['MFWHM'] * gaussian_fwhm_to_sigma
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

    return {'data': data + 1e-6, 'header': header, 'peaks': peaks}


def raw_testdata(spec=False):
    rand = np.random.RandomState(42)
    nx, ny, nz = 256, 256, 4
    dr = 60.0
    angle = 30.0
    header = fits.header.Header()

    header['NAXIS1'] = nx
    header['NAXIS2'] = ny
    header['NAXIS3'] = nz
    header['TELESCOP'] = 'SOFIA'
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
    header['MISSN-ID'] = '2018-12-31_FO_F001'
    header['DATE-OBS'] = '2018-12-31T23:59:59.999'
    header['AOR_ID'] = '90_0001_01'
    header['OBS_ID'] = '2018-12-31_FO_F001B0001'
    header['ALTI_STA'] = 40000
    header['ALTI_END'] = 40005
    header['ZA_START'] = 44.0
    header['ZA_END'] = 44.5
    header['LAT_STA'] = 40.
    header['LON_STA'] = -120.
    header['CHPNPOS'] = 2
    header['DATASRC'] = 'astro'
    header['DETECTOR'] = 'As-010'
    header['DETITIME'] = 1.6384
    header['EPERADU'] = 136
    header['FRMRATE'] = 24.2203
    header['ILOWCAP'] = True
    header['INTTIME'] = 1.6384
    header['NODBEAM'] = 'A'
    header['OBSTYPE'] = 'STANDARD_FLUX'
    header['OTMODE'] = 'AD'
    header['OTNBUFS'] = 2
    header['OTSTACKS'] = 1
    header['SRCTYPE'] = 'POINT_SOURCE'
    header['WAVELNTH'] = 19.7
    header['MFWHM'] = 4.5
    header['BORDER'] = 0
    header['CRPIX1'] = nx // 2
    header['CRPIX2'] = ny // 2
    header['CRVAL1'] = 50.0
    header['CRVAL2'] = 50.0
    header['CDELT1'] = -0.000218419
    header['CDELT2'] = 0.000205914
    header['TELRA'] = 50.0 / 15.
    header['TELDEC'] = 50.0
    header['ANGLCONV'] = 'negative'
    header['SKY_ANGL'] = angle
    header['CROTA2'] = 180 - angle
    header['CHOPPING'] = True
    header['NODDING'] = True
    header['CHPAMP1'] = dr / 2
    header['NODAMP'] = dr
    header['CHPANGLE'] = -330
    header['NODANGLE'] = 210
    header['CHPANGLR'] = 330
    header['NODANGLR'] = 330
    header['CHPCRSYS'] = 'SIRF'
    header['NODCRSYS'] = 'SIRF'
    header['CHPCOORD'] = 0
    header['NODCOORD'] = 0
    header['DITHER'] = True
    header['DITHERX'] = 1.0
    header['DITHERY'] = 1.0
    header['FILENAME'] = 'bFT001_0001.fits'

    shifts = imgshift_header(header)
    dx = shifts['chopx']
    dy = shifts['chopy']

    # build data with Gaussians
    cx, cy = nx // 2, ny // 2
    sigma = header['MFWHM'] * gaussian_fwhm_to_sigma
    amplitude = 2500.0
    gp = {'amplitude': amplitude, 'x_stddev': sigma,
          'y_stddev': sigma, 'x_mean': cx, 'y_mean': cy}
    y, x = np.mgrid[:ny, :nx]

    # A
    data1 = rand.rand(ny, nx) * 200 + 5000
    data4 = rand.rand(ny, nx) * 200 + 5000
    g = Gaussian2D(**gp)
    data1 += g(x, y)
    data4 += g(x, y)

    # B1
    data2 = rand.rand(ny, nx) * 200 + 5000
    gp['amplitude'] /= 2
    gp['x_mean'] = cx + dx
    gp['y_mean'] = cy + dy
    g = Gaussian2D(**gp)
    data2 += g(x, y)

    # B2
    data3 = rand.rand(ny, nx) * 200 + 5000
    gp['x_mean'] = cx - dx
    gp['y_mean'] = cy - dy
    g = Gaussian2D(**gp)
    data3 += g(x, y)

    if spec:
        # make a spectral trace like the center of the gaussian, for all x
        data1 = np.column_stack([data1[:, cx]] * nx)
        data4 = np.column_stack([data4[:, cx]] * nx)
        data2 = np.column_stack([data2[:, int(cx + dx)]] * nx)
        data3 = np.column_stack([data3[:, int(cx - dx)]] * nx)

    data = np.array([data1, data2,
                     data3, data4])

    primary = fits.PrimaryHDU(data=data,
                              header=header)
    hdul = fits.HDUList([primary])

    return hdul


def raw_specdata():
    hdul = raw_testdata(spec=True)
    hdul[0].header['SPECTEL1'] = 'FOR_G111'
    hdul[0].header['SLIT'] = 'FOR_LS24'
    hdul[0].header['SRCTYPE'] = 'POINT_SOURCE'
    hdul[0].header['SKYMODE'] = 'NXCAC'
    hdul[0].header['OBSTYPE'] = 'OBJECT'
    return hdul


def random_mask(shape, frac=0.5):
    rand = np.random.RandomState(42)
    mask = np.full(np.product(shape), False)
    mask[:int(np.product(shape) * frac)] = True
    rand.shuffle(mask)
    return np.reshape(mask, shape)


def add_jailbars(data, level=10):
    if len(data.shape) == 3:
        for frame in data:
            for i in range((frame.shape[1] // 16)):
                frame[:, i * 16] += level
    else:
        for i in range((data.shape[1] // 16)):
            data[:, i * 16] += level
