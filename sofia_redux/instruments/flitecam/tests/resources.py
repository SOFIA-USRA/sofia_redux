# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from astropy.stats import gaussian_fwhm_to_sigma
import numpy as np


def raw_testdata(spec=False, dthindex=1, nodbeam='A', nx=1024, ny=1024,
                 clean=False):
    rand = np.random.RandomState(42 + dthindex)
    angle = 102.099085
    header = fits.header.Header()
    fn = f'{dthindex:04d}'
    dthstep = 5

    cx = nx // 2 + dthstep * (dthindex - 1) - dthstep * 3
    cy = ny // 2 + dthstep * (dthindex - 1)

    header['NAXIS1'] = nx
    header['NAXIS2'] = ny
    header['TELESCOP'] = 'SOFIA'
    header['INSTRUME'] = 'FLITECAM'
    header['INSTCFG'] = 'IMAGING'
    header['INSTMODE'] = 'NOD_OFFARRAY'
    header['SPECTEL1'] = 'FLT_H'
    header['SPECTEL2'] = 'NONE'
    header['SLIT'] = 'NONE'
    header['OBJECT'] = 'HD71264'
    header['MISSN-ID'] = '2014-02-19_FP_F146'
    header['DATE-OBS'] = f'2014-02-19T04:16:{dthindex:02d}.000'
    header['AOR_ID'] = '90_0001_01'
    header['OBS_ID'] = f'140219_000_00FL{fn}'
    header['ALTI_STA'] = 40000
    header['ALTI_END'] = 40005
    header['ZA_START'] = 44.0
    header['ZA_END'] = 44.5
    header['LAT_STA'] = 40.
    header['LON_STA'] = -120.
    header['DATASRC'] = 'astro'
    header['ITIME'] = 1.5
    header['EXPTIME'] = 15.0
    header['COADDS'] = 10
    header['DIVISOR'] = 10
    header['NDR'] = 1
    header['TABLE_MS'] = 197.256
    header['CYCLES'] = 1
    header['SRCTYPE'] = 'POINT_SOURCE'
    header['CRPIX1'] = cx
    header['CRPIX2'] = cy
    header['CRVAL1'] = 50.0
    header['CRVAL2'] = 50.0
    header['CDELT1'] = -0.000218419
    header['CDELT2'] = 0.000205914
    header['CD1_1'] = 2.7656E-05
    header['CD1_2'] = -0.000129014
    header['CD2_1'] = 0.000129014
    header['CD2_2'] = 2.7656E-05
    header['TELRA'] = 50.0 / 15.
    header['TELDEC'] = 50.0
    header['ROT_ANGL'] = angle
    header['NODDING'] = True
    header['NODANGLE'] = 210
    header['NODBEAM'] = nodbeam
    header['NODANGLR'] = 330
    header['NODCRSYS'] = 'SIRF'
    header['DITHER'] = True
    header['DTHINDEX'] = dthindex
    header['FILENAME'] = f'Feb-19-2014-{fn}.a.fits'

    # add noise
    data1 = rand.rand(ny, nx) * 10000 + 3000

    # add random bad pixels
    hot = rand.choice(np.arange(nx * ny), 1000)
    cold = rand.choice(np.arange(nx * ny), 500)
    if clean:
        data1.flat[hot] = np.nan
        data1.flat[cold] = np.nan
    else:
        data1.flat[hot] *= 1e6
        data1.flat[cold] *= -1e6

    # add Gaussians for on-source image
    if nodbeam == 'A':
        header['OBSTYPE'] = 'STANDARD_FLUX'
        sigma = 6.0 * gaussian_fwhm_to_sigma
        amplitude = 28000.0
        gp = {'amplitude': amplitude, 'x_stddev': sigma,
              'y_stddev': sigma, 'x_mean': cx, 'y_mean': cy}
        y, x = np.mgrid[:ny, :nx]

        g = Gaussian2D(**gp)
        data1 += g(x, y)

        if spec:
            # make a spectral trace like the center of the
            # gaussian, for all y
            data1 = np.row_stack([data1[cy, :]] * ny)
    else:
        header['OBSTYPE'] = 'SKY'

    primary = fits.PrimaryHDU(data=data1,
                              header=header)
    hdul = fits.HDUList([primary])

    return hdul


def raw_specdata(dthindex=1, nodbeam='A', add_ext=False):
    hdul = raw_testdata(spec=True, dthindex=dthindex, nodbeam=nodbeam)
    hdul[0].header['SPECTEL1'] = 'FLT_B3_J'
    hdul[0].header['SPECTEL2'] = 'FLT_SS20'
    hdul[0].header['SLIT'] = 'FLT_SS20'
    hdul[0].header['INSTCFG'] = 'SPECTROSCOPY'
    hdul[0].header['INSTMODE'] = 'NOD_OFF_SLIT'
    hdul[0].header['OBJECT'] = 'HD370'

    if nodbeam == 'A':
        hdul[0].header['OBSTYPE'] = 'OBJECT'

    # add expected intermediate extensions if desired
    if add_ext:
        hdul[0].header['EXTNAME'] = 'FLUX'
        hdul.append(fits.ImageHDU(np.abs(hdul[0].data) * 0.1, name='ERROR'))
        hdul.append(fits.ImageHDU(np.zeros(hdul[0].data.shape, dtype=int),
                                  name='BADMASK'))

    return hdul


def intermediate_data(dthindex=1, nodbeam='A', nx=500, ny=500, clean=True):
    hdul = raw_testdata(dthindex=dthindex, nodbeam=nodbeam,
                        nx=nx, ny=ny, clean=clean)

    # flux and error extensions
    hdul[0].header['EXTNAME'] = 'FLUX'
    err = hdul[0].data * 0.1
    hdul.append(fits.ImageHDU(err, name='ERROR'))
    return hdul
