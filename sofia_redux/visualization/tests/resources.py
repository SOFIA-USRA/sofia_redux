import astropy.io.fits as pf
from astropy.modeling.models import Gaussian2D, Gaussian1D
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

    header = pf.header.Header()
    header['NAXIS1'] = nx
    header['NAXIS2'] = ny
    header['TELESCOP'] = 'SOFIA'
    header['AOR_ID'] = '90_0001_01'
    header['PIPEVERS'] = '2_0_0'
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
    header['FILENUM'] = 'UNKNOWN'

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

    primary = pf.PrimaryHDU(data=data,
                            header=header)
    err = pf.ImageHDU(errdata, name='ERROR')
    hdul = pf.HDUList([primary, err])

    return hdul


def cube_data():
    hdul = basic_data()
    header = hdul[0].header

    primary = pf.PrimaryHDU(header=header)
    flux = pf.ImageHDU(name='FLUX')
    error = pf.ImageHDU(name='ERROR')
    uncorrected_flux = pf.ImageHDU(name='UNCORRECTED_FLUX')
    uncorrected_error = pf.ImageHDU(name='UNCORRECTED_ERROR')
    wavelength = pf.ImageHDU(name='WAVELENGTH')
    x = pf.ImageHDU(name='X')
    y = pf.ImageHDU(name='Y')
    transmission = pf.ImageHDU(name='TRANSMISSION')
    response = pf.ImageHDU(name='RESPONSE')
    exposure_map = pf.ImageHDU(name='EXPOSURE_MAP')
    unsmoothed_transmission = pf.ImageHDU(name='UNSMOOTHED_TRANSMISSION')

    x_size = 55
    y_size = 74
    wave_size = 56
    for hdu in [flux, error, uncorrected_error, uncorrected_flux,
                exposure_map, unsmoothed_transmission]:
        hdu.data = np.random.random((x_size, y_size, wave_size))

    x.data = np.arange(x_size)
    y.data = np.arange(y_size)
    wavelength.data = np.linspace(50, 300, wave_size)
    transmission.data = np.ones(wave_size)
    response.data = np.ones(wave_size)

    hdul = pf.HDUList([primary, flux, error, uncorrected_flux,
                       uncorrected_error, wavelength, x, y,
                       transmission, response, exposure_map,
                       unsmoothed_transmission])
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
    new_hdul = pf.HDUList(primary)
    return new_hdul


def flipo_data():
    """Synthetic FLITECAM test data, in FLIPO config."""
    hdul = flitecam_data()
    header = hdul[0].header

    # change the mission id
    header['MISSN-ID'] = '2018-12-31_FP_F001'

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
    header['TOTINT'] = 120

    header['PROCSTAT'] = 'LEVEL_3'
    header['PRODTYPE'] = 'STDPHOTCAL'
    header['NAXIS1'] = hdul[0].data.shape[0]
    header['NAXIS2'] = hdul[0].data.shape[1]

    # add an exposure map
    exp = hdul[1].copy()
    exp.header['EXTNAME'] = 'EXPOSURE'
    exp.data[:, :] = 1.0
    hdul.append(exp)

    return hdul


def forcast_legacy_data():
    """Synthetic FORCAST data from early pipeline versions (<2.0)."""
    hdul = forcast_data()

    # add variance as second plane, instead of error extension
    primary = hdul[0]
    err_data = hdul[1].data
    exp_data = hdul[2].data
    stacked = np.array([primary.data, err_data ** 2, exp_data])
    primary.data = stacked
    primary.header['NAXIS3'] = 2
    new_hdul = pf.HDUList(primary)
    return new_hdul


def hawc_pol_data():
    """Synthetic HAWC+ polarimetry data."""
    hdul = basic_data()
    header = hdul[0].header

    # add HAWC keys
    header['INSTRUME'] = 'HAWC_PLUS'
    header['INSTCFG'] = 'POLARIZATION'
    header['INSTMODE'] = 'C2N (NMC)'
    header['SPECTEL1'] = 'HAW_D'
    header['SPECTEL2'] = 'HAW_HWP_D'
    header['EXPTIME'] = 120

    header['OBJECT'] = 'Uranus'
    header['DATE-OBS'] = '2019-02-20T02:00:00.000'
    header['MISSN-ID'] = '2019-02-20_HA_F001'

    new_hdul = pf.HDUList()
    for i, stokes in enumerate(['I', 'Q', 'U']):
        im = hdul[0].copy()
        im.header['EXTNAME'] = 'STOKES {}'.format(stokes)
        err = hdul[1].copy()
        err.header['EXTNAME'] = 'ERROR {}'.format(stokes)
        if i == 0:
            new_hdul.append(im)
        else:
            new_hdul.append(pf.ImageHDU(im.data, im.header))
        new_hdul.append(err)

    return new_hdul


def hawc_im_data():
    """Synthetic HAWC+ scanning imaging data."""
    hdul = basic_data()
    header = hdul[0].header

    # add HAWC keys
    header['INSTRUME'] = 'HAWC_PLUS'
    header['INSTCFG'] = 'TOTAL_INTENSITY'
    header['INSTMODE'] = 'OTFMAP'
    header['SPECTEL1'] = 'HAW_D'
    header['SPECTEL2'] = 'HAW_HWP_Open'
    header['EXPTIME'] = 120
    header['PROCSTAT'] = 'LEVEL_2'

    header['OBJECT'] = 'Uranus'
    header['DATE-OBS'] = '2019-02-20T02:00:00.000'
    header['MISSN-ID'] = '2019-02-20_HA_F001'
    header['CALQUAL'] = 'Nominal'

    new_hdul = pf.HDUList()

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


def fifi_cube_data():
    """Synthetic FORCAST test data."""
    hdul = cube_data()
    header = hdul[0].header

    # add FORCAST keys
    header['INSTRUME'] = 'FIFI-LS'
    header['INSTCFG'] = 'DUAL_CHANNEL'
    header['INSTMODE'] = 'SYMMETRIC_CHOP'
    header['SKYMODE'] = ''
    header['SPECTEL1'] = 'FIF_BLUE'
    header['SPECTEL2'] = 'NONE'
    header['DETCHAN'] = 'BLUE'
    header['SLIT'] = 'UNKNOWN'
    header['DICHROIC'] = '105'

    header['OBJECT'] = 'Alpha Boo'
    header['DATE-OBS'] = '2019-10-30T11:02:27'
    header['MISSN-ID'] = '2019-10-30_FI_F631'
    header['EXPTIME'] = 120

    header['PROCSTAT'] = 'LEVEL_4'
    header['PRODTYPE'] = 'resampled'

    return hdul


def great_data():
    hdul = basic_data()
    header = hdul[0].header

    header['INSTRUME'] = 'GREAT'
    header['INSTCFG'] = 'DUAL_CHANNEL'
    header['SPECTEL1'] = 'GRE_HFA'
    header['SPECTEL2'] = 'NONE'

    header['IMAGFREQ'] = 4744777.49
    header['RESTFREQ'] = 4744777.49
    header['ASSC_FRQ'] = 4744777.49
    header['VELRES'] = 0.015
    header['FREQRES'] = -0.23441

    header['OBJECT'] = 'Alpha Boo'
    header['DATE-OBS'] = '2019-10-30T11:02:27'
    header['MISSN-ID'] = '2019-10-30_FI_F631'
    header['EXPTIME'] = 120

    header['PROCSTAT'] = 'LEVEL_4'
    header['PRODTYPE'] = 'resampled'


def exes_data():
    hdul = basic_data()
    header = hdul[0].header

    header['INSTRUME'] = 'EXES'
    header['INSTMODE'] = 'STARE'
    header['INSTCFG'] = 'HIGH_MED'
    header['SPECTEL1'] = 'EXE_ELON'
    header['SPECTEL2'] = 'EXE_ECHL'

    return hdul


def exes_spec_data(n_orders=5, n_apertures=1):
    min_waves = np.linspace(1480, 1500, n_orders)
    flux_levels = np.linspace(0.5, 2, n_apertures)
    full_data = np.zeros((n_orders * n_apertures, 100, 5))
    index = 0
    for i in range(n_orders):
        for j in range(n_apertures):
            hdul = exes_merged_data(min_wave=min_waves[i],
                                    max_wave=min_waves[i] + 2,
                                    mean_flux=flux_levels[j],
                                    emission=j % 2 == 0)
            full_data[index] = hdul[0].data
            index += 1

    hdul[0].data = full_data

    hdul[0].header['INSTRUME'] = 'EXES'
    hdul[0].header['INSTMODE'] = 'HIGH_MED'
    hdul[0].header['PRODTYPE'] = 'spectra_1d'
    hdul[0].header['NORDERS'] = n_orders
    hdul[0].header['NAPS'] = n_apertures
    return hdul


def exes_merged_data(min_wave=1, max_wave=10, mean_flux=1,
                     emission=True):
    hdul = exes_data().copy()
    hdul.pop(1)
    n_pix = 100
    amplitude = 0.4
    sigma = 0.1
    peak = 1484
    mean_flux = 0.6

    gp = {'amplitude': amplitude, 'stddev': sigma, 'mean': peak}
    g = Gaussian1D(**gp)
    wavelength = np.linspace(min_wave, max_wave, n_pix)
    if emission:
        flux = np.ones_like(n_pix) * mean_flux + g(wavelength)
    else:
        flux = np.ones_like(n_pix) * mean_flux - g(wavelength)
    error = np.random.normal(0, 1, n_pix)
    transmission = np.ones_like(wavelength)
    response = np.ones_like(wavelength)

    data = np.zeros((5, n_pix))
    data[0] = wavelength
    data[1] = flux
    data[2] = error
    data[3] = transmission
    data[4] = response

    hdul[0].data = data.T
    hdul[0].header['norders'] = 1

    return hdul


def npc_testdata():
    nx, ny = 500, 500
    dr = 80.0 + 1 / 3  # for fractional testing
    angle = 30.0
    header = pf.header.Header()

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
    # shifts = imgshift_header(header)
    peaks, sign = [], []
    cx, cy = nx // 2, ny // 2
    # nodx, nody = shifts['nodx'] / 2, shifts['nody'] / 2
    # chopx, chopy = shifts['chopx'] / 2, shifts['chopy'] / 2
    nodx, nody = 1, 1,
    chopx, chopy = 1, 1,
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
    header = pf.header.Header()

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
    # shifts = imgshift_header(header)
    # dx = shifts['chopx']
    # dy = shifts['chopy']
    dx = 1
    dy = 1

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
    header = pf.header.Header()

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

    # shifts = imgshift_header(header)
    # dx = shifts['chopx']
    # dy = shifts['chopy']
    dx, dy = 1, 1

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

    primary = pf.PrimaryHDU(data=data,
                            header=header)
    hdul = pf.HDUList([primary])

    return hdul


def raw_specdata():
    hdul = raw_testdata(spec=True)
    hdul[0].header['SPECTEL1'] = 'FOR_G111'
    hdul[0].header['SLIT'] = 'FOR_LS24'
    hdul[0].header['SRCTYPE'] = 'POINT_SOURCE'
    hdul[0].header['SKYMODE'] = 'NXCAC'
    hdul[0].header['OBSTYPE'] = 'OBJECT'
    hdul[0].header['PRODTYPE'] = 'CALIBRATE'
    return hdul


def exes_image_data():
    nx = 100
    ny = 100
    data = np.random.normal(10, 0.3, (nx, ny))
    return data


def exes_flux_data():
    return exes_image_data()


def exes_error_data():
    return exes_image_data() / 10


def exes_mask_data():
    mask = (exes_image_data() > 9).astype(int)
    return mask


def exes_flat_data():
    return exes_image_data()


def exes_wavecal_data():
    return exes_image_data()


def exes_spatcal_data():
    return exes_image_data()


def exes_order_mask_data():
    mask = (exes_image_data() > 9).astype(int)
    return mask


def exes_flux_order_data():
    data = np.random.normal(10, 0.3, (10, 100))
    return data


def exes_error_order_data():
    return exes_flux_order_data() / 10


def exes_flat_order_data():
    return exes_flux_order_data()


def exes_badmask_order_data():
    return (exes_flux_order_data() < 0.9).astype(int)


def exes_wavepos_order_data():
    return np.linspace(5, 10, 100)


def exes_slitpos_order_data():
    return np.arange(10)


def exes_spatial_map_order_data():
    return exes_flux_order_data()


def exes_spatial_profile_order_data():
    g = Gaussian1D(amplitude=0.05, mean=2, stddev=0.2)
    x = np.linspace(0, 10, 10)
    return g(x)


def exes_aperture_mask_order_data():
    return exes_flux_order_data()


def exes_intermediate_pre_orders(prodtype='calibration_corrected',
                                 file_code='CCR', file_num=1,
                                 n_orders=5, include_cal=True,
                                 bunit='erg s-1 cm-2 sr-1 (cm-1)-1'):
    hdul = exes_data()
    header = hdul[0].header.copy()

    header['PRODTYPE'] = prodtype
    header['NORDERS'] = n_orders
    filename = f'F0158_EX_SPE_86000114_EXEELONEXEECHL_' \
               f'{file_code}_{file_num}.fits'
    header['FILENAME'] = filename
    header['EXTNAME'] = 'FLUX'
    header['BUNIT'] = bunit

    hdus = list()
    hdus.append(pf.PrimaryHDU(exes_flux_data(), header))
    hdus.append(pf.ImageHDU(exes_error_data(), name='ERROR'))
    hdus.append(pf.ImageHDU(exes_mask_data(), name='MASK'))
    hdus.append(pf.ImageHDU(exes_flat_data(), name='FLAT'))
    hdus.append(pf.ImageHDU(exes_error_data(), name='FLAT_ERROR'))
    hdus.append(pf.ImageHDU(exes_flat_data(),
                            name='FLAT_ILLUMINATION'))
    if include_cal:
        hdus.append(pf.ImageHDU(exes_wavecal_data(), name='WAVECAL'))
        hdus.append(pf.ImageHDU(exes_spatcal_data(), name='SPATCAL'))
        hdus.append(pf.ImageHDU(exes_order_mask_data(),
                                name='ORDER_MASK'))

    hdul = pf.HDUList(hdus)
    return hdul


def exes_intermediate_with_orders(prodtype='apertures_set',
                                  file_code='APS', file_num=1,
                                  n_orders=5, n_aps=1,
                                  bunit='Jy/pixel'):
    hdul = exes_data()
    header = hdul[0].header.copy()

    header['PRODTYPE'] = prodtype
    header['NORDERS'] = n_orders
    header['NAPS'] = n_aps
    filename = f'F0158_EX_SPE_86000114_EXEELONEXEECHL_' \
               f'{file_code}_{file_num}.fits'
    header['FILENAME'] = filename
    header['EXTNAME'] = 'FLUX'
    header['BUNIT'] = bunit

    hdus = list()
    hdus.append(pf.PrimaryHDU(exes_flux_data(), header))
    hdus.append(pf.ImageHDU(exes_error_data(), name='ERROR'))
    hdus.append(pf.ImageHDU(exes_mask_data(), name='MASK'))
    hdus.append(pf.ImageHDU(exes_flat_data(), name='FLAT'))
    hdus.append(pf.ImageHDU(exes_wavecal_data(), name='WAVECAL'))
    hdus.append(pf.ImageHDU(exes_spatcal_data(), name='SPATCAL'))
    hdus.append(pf.ImageHDU(exes_order_mask_data(),
                            name='ORDER_MASK'))

    for x in range(n_orders):
        i = x + 1
        hdus.append(pf.ImageHDU(exes_flux_order_data(),
                                name=f'FLUX_ORDER_{i:02d}'))
        hdus.append(pf.ImageHDU(exes_error_order_data(),
                                name=f'ERROR_ORDER_{i:02d}'))
        hdus.append(pf.ImageHDU(exes_flat_order_data(),
                                name=f'FLAT_ORDER_{i:02d}'))
        hdus.append(pf.ImageHDU(exes_badmask_order_data(),
                                name=f'BADMASK_ORDER_{i:02d}'))
        hdus.append(pf.ImageHDU(exes_wavepos_order_data(),
                                name=f'WAVEPOS_ORDER_{i:02d}'))
        hdus.append(pf.ImageHDU(exes_slitpos_order_data(),
                                name=f'SLITPOS_ORDER_{i:02d}'))
        hdus.append(pf.ImageHDU(exes_spatial_map_order_data(),
                                name=f'SPATIAL_MAP_ORDER_{i:02d}'))
        hdus.append(pf.ImageHDU(exes_spatial_profile_order_data(),
                                name=f'SPATIAL_PROFILE_ORDER_{i:02d}'))
        hdus.append(pf.ImageHDU(exes_aperture_mask_order_data(),
                                name=f'APERTURE_MASK_ORDER_{i:02d}'))

    hdul = pf.HDUList(hdus)

    return hdul


def exes_intermediate_with_spectra(prodtype='spectra',
                                   file_code='SPM', file_num=1,
                                   n_orders=5, n_aps=1, merge=False,
                                   bunit='Jy/pixel'):
    hdul = exes_data()
    header = hdul[0].header.copy()

    header['PRODTYPE'] = prodtype
    header['NORDERS'] = n_orders
    header['NAPS'] = n_aps
    filename = f'F0158_EX_SPE_86000114_EXEELONEXEECHL_' \
               f'{file_code}_{file_num}.fits'
    header['FILENAME'] = filename
    header['EXTNAME'] = 'FLUX'
    header['BUNIT'] = bunit

    spec_data = exes_spec_data(n_orders=n_orders, n_apertures=n_aps)
    spec_flux = spec_data[0].data[..., 1]
    spec_err = spec_data[0].data[..., 2]
    spec_trans = spec_data[0].data[..., 3]
    spec_resp = spec_data[0].data[..., 4]
    # modify transmission to look like multi-species array
    shape = (11, spec_trans[0].size)
    spec_trans = np.tile(spec_trans[0], 11).reshape(shape)

    hdus = list()
    hdus.append(pf.PrimaryHDU(exes_flux_data(), header))
    hdus.append(pf.ImageHDU(exes_error_data(), name='ERROR'))
    if not merge:
        # these extensions are dropped after combining spectra
        hdus.append(pf.ImageHDU(exes_mask_data(), name='MASK'))
        hdus.append(pf.ImageHDU(exes_flat_data(), name='FLAT'))
        hdus.append(pf.ImageHDU(exes_wavecal_data(), name='WAVECAL'))
        hdus.append(pf.ImageHDU(exes_spatcal_data(), name='SPATCAL'))
        hdus.append(pf.ImageHDU(exes_order_mask_data(), name='ORDER_MASK'))
    hdus.append(pf.ImageHDU(spec_trans, name='TRANSMISSION'))

    for x in range(n_orders):
        i = x + 1
        if n_orders > 1:
            if n_aps > 1:
                j = np.arange(x * n_aps, x * n_aps + n_aps)
            else:
                j = x
        else:
            j = None

        hdus.append(pf.ImageHDU(exes_flux_order_data(),
                                name=f'FLUX_ORDER_{i:02d}'))
        hdus.append(pf.ImageHDU(exes_error_order_data(),
                                name=f'ERROR_ORDER_{i:02d}'))
        if not merge:
            # attach flat, mask information
            hdus.append(pf.ImageHDU(exes_flat_order_data(),
                                    name=f'FLAT_ORDER_{i:02d}'))
            hdus.append(pf.ImageHDU(exes_badmask_order_data(),
                                    name=f'BADMASK_ORDER_{i:02d}'))

        hdus.append(pf.ImageHDU(exes_wavepos_order_data(),
                                name=f'WAVEPOS_ORDER_{i:02d}'))
        if not merge:
            # attach extraction information
            hdus.append(pf.ImageHDU(exes_slitpos_order_data(),
                                    name=f'SLITPOS_ORDER_{i:02d}'))
            hdus.append(pf.ImageHDU(exes_spatial_map_order_data(),
                                    name=f'SPATIAL_MAP_ORDER_{i:02d}'))
            hdus.append(pf.ImageHDU(exes_spatial_profile_order_data(),
                                    name=f'SPATIAL_PROFILE_ORDER_{i:02d}'))
            hdus.append(pf.ImageHDU(exes_aperture_mask_order_data(),
                                    name=f'APERTURE_MASK_ORDER_{i:02d}'))

            # preserve the empty first dimension
            hdus.append(pf.ImageHDU(spec_flux[j].reshape(n_aps, shape[1]),
                                    name=f'SPECTRAL_FLUX_ORDER_{i:02d}'))
            hdus.append(pf.ImageHDU(spec_err[j].reshape(n_aps, shape[1]),
                                    name=f'SPECTRAL_ERROR_ORDER_{i:02d}'))
            hdus.append(pf.ImageHDU(spec_trans,
                                    name=f'TRANSMISSION_ORDER_{i:02d}'))
            hdus.append(pf.ImageHDU(spec_resp[j].reshape(n_aps, shape[1]),
                                    name=f'RESPONSE_ORDER_{i:02d}'))
        else:
            # extraction-related extensions are dropped,
            # no extra dimension in merged/combined spectra
            hdus.append(pf.ImageHDU(np.squeeze(spec_flux[j]),
                                    name=f'SPECTRAL_FLUX_ORDER_{i:02d}'))
            hdus.append(pf.ImageHDU(np.squeeze(spec_err[j]),
                                    name=f'SPECTRAL_ERROR_ORDER_{i:02d}'))
            hdus.append(pf.ImageHDU(np.squeeze(spec_trans),
                                    name=f'TRANSMISSION_ORDER_{i:02d}'))
            hdus.append(pf.ImageHDU(np.squeeze(spec_resp[j]),
                                    name=f'RESPONSE_ORDER_{i:02d}'))

    hdul = pf.HDUList(hdus)

    return hdul


def exes_final(prodtype='orders_merged', file_code='MRM',
               file_num=1, n_aps=1, bunit='Jy/pixel'):
    hdul = exes_intermediate_with_spectra(
        prodtype=prodtype, file_code=file_code, file_num=file_num,
        n_orders=1, n_aps=n_aps, merge=True, bunit=bunit)
    header = hdul[0].header

    new_hdul = pf.HDUList()
    for hdu in hdul:
        extname = hdu.header['EXTNAME']
        if 'ORDER' in extname:
            if not new_hdul:
                hdu = pf.PrimaryHDU(hdu.data, header)
            hdu.header['EXTNAME'] = extname.replace('_ORDER_01', '')
            new_hdul.append(hdu)

    return new_hdul


def make_1d_exes(hdul):
    """
    Make 1d spectrum file.

    Borrowed and modified from the EXES Redux method.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        File containing 2D and 1D spectral extensions.

    Returns
    -------
    spectrum : astropy.io.fits.HDUList
        New HDU list in Spextool style, containing 1D spectra.
    """
    header = hdul[0].header
    spechdr = header.copy()

    # add some spextool-required header keywords
    bunit = hdul[0].header.get('BUNIT', 'UNKNOWN').replace('/pixel', '')
    spechdr['XUNITS'] = ('cm-1', 'Spectral wavelength units')
    spechdr['YUNITS'] = (bunit, 'Spectral flux units')

    try:
        del spechdr['BUNIT']
    except KeyError:  # pragma: no cover
        pass

    # loop over orders
    specset = []
    first = True
    max_length = 0
    orders = []
    for j in range(header['NORDERS']):
        ordnum = f'{j + 1:02d}'

        # keep actually used orders
        orders.append(j + 1)

        suffix = f'_ORDER_{ordnum}'
        if (header['NORDERS'] == 1
                and f'SPECTRAL_FLUX{suffix}' not in hdul):
            suffix = ''

        if hdul[f'SPECTRAL_FLUX{suffix}'].data.ndim > 1:
            naps = hdul[f'SPECTRAL_FLUX{suffix}'].data.shape[0]
        else:
            naps = 1
        if first:
            spechdr['NAPS'] = (naps, 'Number of apertures')
            first = False

        wave = hdul[f'WAVEPOS{suffix}'].data
        disp = np.nanmean(wave[1:] - wave[:-1])
        spechdr[f'DISPO{ordnum}'] = (disp, 'Dispersion [cm-1 pixel-1]')
        if len(wave) > max_length:
            max_length = len(wave)

        # transmission rows include additional data for various
        # molecular species -- use the first only,
        # which contains total transmission.
        trans = hdul[f'TRANSMISSION{suffix}'].data
        if trans.ndim > 1:
            trans = trans[0]

        for n in range(naps):
            if naps > 1:
                speclist = [hdul[f'WAVEPOS{suffix}'].data,
                            hdul[f'SPECTRAL_FLUX{suffix}'].data[n],
                            hdul[f'SPECTRAL_ERROR{suffix}'].data[n],
                            trans,
                            hdul[f'RESPONSE{suffix}'].data[n]]
            else:
                speclist = [hdul[f'WAVEPOS{suffix}'].data,
                            hdul[f'SPECTRAL_FLUX{suffix}'].data,
                            hdul[f'SPECTRAL_ERROR{suffix}'].data,
                            trans,
                            hdul[f'RESPONSE{suffix}'].data]

            specdata = np.vstack(speclist)
            specset.append(specdata)

    # update ORDERS and NORDERS in header to actually used values
    spechdr['ORDERS'] = ','.join([str(n) for n in orders])
    spechdr['NORDERS'] = len(orders)

    nspec = len(specset)
    spec_array = np.full((nspec, 5, max_length), np.nan)
    for i, data in enumerate(specset):
        spec_array[i, :, :data.shape[1]] = data

    # remove any empty dimensions
    spec_array = np.squeeze(spec_array)

    spec = pf.HDUList(pf.PrimaryHDU(data=spec_array,
                                    header=spechdr))
    return spec
