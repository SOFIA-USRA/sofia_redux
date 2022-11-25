# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy.io import fits
from astropy.modeling.models import Gaussian1D
from astropy.stats import gaussian_fwhm_to_sigma
import numpy as np

from sofia_redux.instruments.exes.readhdr import readhdr
from sofia_redux.instruments.exes.makeflat import makeflat
from sofia_redux.toolkit.image.adjust import rotate


def low_header(filenum=1, coadded=False):
    header = fits.header.Header()
    fn = f'{filenum:02d}'

    # two patterns, two integrations, two nods, ND reads
    # 16 frames in the raw:
    #   BN BD BN BD | AN AD AN AD | BN BD BN BD | AN AD AN AD
    header['OTPAT'] = 'N0D0'
    header['NODN'] = 2
    header['NINT'] = 2
    if coadded:
        header['RAWFNAME'] = f'test.sci.100{fn}.fits'
        header['FILENAME'] = f'F0001_EX_SPE_01000101_NONEEXEECHL_' \
                             f'RDC_100{fn}.fits'
        header['EXTNAME'] = 'FLUX'
        header['PRODTYPE'] = 'readouts_coadded'
        header['BEAMTIME'] = 2.0
    else:
        header['FILENAME'] = f'test.sci.100{fn}.fits'

    # low mode, nod-on-slit
    header['DATASRC'] = 'ASTRO'
    header['OBSTYPE'] = 'OBJECT'
    header['INSTCFG'] = 'LOW'
    header['INSTMODE'] = 'NOD_ON_SLIT'
    header['WAVENO0'] = 547.0
    header['EXPTIME'] = 100.0
    header['DATE-OBS'] = f'2015-01-01T00:{fn}:00.000'
    header['UTCSTART'] = f'00:{fn}:00.000'
    header['UTCEND'] = f'00:{filenum + 1:02d}:00.000'
    header['MISSN-ID'] = '2015-01-01_EX_F001'
    header['OBS_ID'] = f'2015-01-01_EX_F001100{fn}'
    header['AOR_ID'] = '01_0001_01'
    header['SPECTEL1'] = 'NONE'
    header['SPECTEL2'] = 'EXE_ECHL'
    header['NODBEAM'] = 'B'
    header['RESOLUN'] = 2000.
    header['ALTI_STA'] = 40000.
    header['ALTI_END'] = 41000.
    header['ZA_START'] = 40.
    header['ZA_END'] = 45.
    header['FLATTAMB'] = 290.
    header['FLATEMIS'] = 0.1

    # set EXES defaults
    header = readhdr(header, check_header=False)

    return header


def low_flat_header(filenum=1, coadded=False):
    header = fits.header.Header()
    fn = f'{filenum:02d}'

    # two patterns, two integrations, ND reads
    # 8 frames in raw:
    #   AN AD AN AD
    header['OTPAT'] = 'N0D0'
    header['NINT'] = 2
    if coadded:
        header['RAWFNAME'] = f'test.flat.100{fn}.fits'
        header['FILENAME'] = f'F0001_EX_SPE_01000101_NONEEXEECHL_' \
                             f'RDC_100{fn}.fits'
        header['EXTNAME'] = 'FLUX'
        header['PRODTYPE'] = 'readouts_coadded'
        header['BEAMTIME'] = 2.0
    else:
        header['FILENAME'] = f'test.flat.100{fn}.fits'

    # low mode, stare
    header['INSTCFG'] = 'LOW'
    header['INSTMODE'] = 'STARE'
    header['DATASRC'] = 'CALIBRATION'
    header['OBSTYPE'] = 'FLAT'
    header['WAVENO0'] = 547.0
    header['EXPTIME'] = 10.0
    header['DATE-OBS'] = f'2015-01-01T00:{fn}:00.000'
    header['UTCSTART'] = f'00:{fn}:00.000'
    header['UTCEND'] = f'00:{filenum + 1:02d}:00.000'
    header['MISSN-ID'] = '2015-01-01_EX_F001'
    header['OBS_ID'] = f'2015-01-01_EX_F001100{fn}'
    header['AOR_ID'] = '01_0001_01'
    header['SPECTEL1'] = 'NONE'
    header['SPECTEL2'] = 'EXE_ECHL'
    header['RESOLUN'] = 2000
    header['ALTI_STA'] = 40000.
    header['ALTI_END'] = 41000.
    header['ZA_START'] = 40.
    header['ZA_END'] = 45.
    header['FLATTAMB'] = 290.
    header['FLATEMIS'] = 0.1

    # set rotation parameters to zero
    header['SLITROT'] = 0
    header['DETROT'] = 0

    # set EXES defaults
    header = readhdr(header, check_header=False)

    return header


def raw_low_nod_on(coadded=False, use_dark=True, filenum=2, dy=40):
    rand = np.random.RandomState(filenum)
    header = low_header(filenum, coadded)

    # two patterns, two integrations, two nods, ND reads
    # 16 frames in the raw:
    #   BN BD BN BD | AN AD AN AD | BN BD BN BD | AN AD AN AD
    if coadded:
        # 1 frame coadded
        nx, ny, nz = 1024, 1024, 1
    else:
        nx, ny, nz = 1032, 1024, 16

    # build data with Gaussian profile,
    # spectral axis along x
    sigma = 10.0 * gaussian_fwhm_to_sigma
    amplitude = 300
    cy = ny // 2
    gp = {'amplitude': amplitude, 'stddev': sigma,
          'mean': cy - dy}
    y = np.arange(ny)

    # A
    data2 = rand.rand(ny, nx) * 2 + 1000
    data4 = rand.rand(ny, nx) * 2 + 1000
    g = Gaussian1D(**gp)
    data2 += g(y)[:, None]
    data4 += g(y)[:, None]

    # B
    data1 = rand.rand(ny, nx) * 2 + 1000
    data3 = rand.rand(ny, nx) * 2 + 1000
    gp['mean'] = cy + dy
    g = Gaussian1D(**gp)
    data1 += g(y)[:, None]
    data3 += g(y)[:, None]

    if coadded:
        data = np.array([data1, data2,
                         data3, data4])
        # gain factor
        data /= 2.0
        error = 0.001 * np.abs(data)
        mask = np.zeros((ny, nx), dtype=int)

        # add a few spikes
        for n in range(data.shape[0]):
            sy = rand.randint(0, ny, 20)
            sx = rand.randint(0, nx, 20)
            data[n, sy, sx] += 20000

        primary = fits.PrimaryHDU(data=data,
                                  header=header)
        error = fits.ImageHDU(data=error, name='ERROR')
        mask = fits.ImageHDU(data=mask, name='MASK')
        hdul = fits.HDUList([primary, error, mask])
    else:
        dark_file = os.path.join(header['DATAPATH'], 'dark',
                                 'dark_2015.02.13.fits')
        if use_dark and os.path.isfile(dark_file):
            dark = fits.open(dark_file)
            read_pattern = dark[0].data[0]
            dark.close()
        else:
            read_pattern = np.full((ny, nx), 10000.)
            read_pattern[:, ::3] -= 10
            read_pattern[:, 1::3] -= 5

        full_data = np.zeros((nz, ny, nx))
        # set Ns to half signal + read_pattern
        # set Ds to A or B data + read_pattern
        full_data[0] = -data1 / 2 + read_pattern
        full_data[1] = -data1 + read_pattern
        full_data[2] = -data1 / 2 + read_pattern
        full_data[3] = -data1 + read_pattern
        full_data[4] = -data2 / 2 + read_pattern
        full_data[5] = -data2 + read_pattern
        full_data[6] = -data2 / 2 + read_pattern
        full_data[7] = -data2 + read_pattern
        full_data[8] = -data3 / 2 + read_pattern
        full_data[9] = -data3 + read_pattern
        full_data[10] = -data3 / 2 + read_pattern
        full_data[11] = -data3 + read_pattern
        full_data[12] = -data4 / 2 + read_pattern
        full_data[13] = -data4 + read_pattern
        full_data[14] = -data4 / 2 + read_pattern
        full_data[15] = -data4 + read_pattern

        # add a few spikes
        for n in range(full_data.shape[0]):
            sy = rand.randint(0, ny, 20)
            sx = rand.randint(0, nx, 20)
            full_data[n, sy, sx] -= 20000

        primary = fits.PrimaryHDU(data=full_data,
                                  header=header)
        hdul = fits.HDUList(primary)

    return hdul


def raw_low_flat(coadded=False, use_dark=True, filenum=1):
    rand = np.random.RandomState(filenum)
    header = low_flat_header(filenum, coadded)

    # two patterns, two integrations, ND reads
    # 8 frames in raw:
    #   AN AD AN AD
    if coadded:
        # 1 frame coadded
        nx, ny, nz = 1024, 1024, 1
    else:
        nx, ny, nz = 1032, 1024, 4

    # data is simple flat with unilluminated regions,
    # spectral axis along x
    data = rand.rand(ny, nx) * 20 + 10000
    data[:350] -= 9000
    data[700:] -= 9000

    if coadded:
        # gain factor
        data = np.expand_dims(data, axis=0)
        data /= 2.0
        error = 0.001 * np.abs(data)
        mask = np.zeros((ny, nx), dtype=int)

        primary = fits.PrimaryHDU(data=data,
                                  header=header)
        error = fits.ImageHDU(data=error, name='ERROR')
        mask = fits.ImageHDU(data=mask, name='MASK')
        hdul = fits.HDUList([primary, error, mask])
    else:
        dark_file = os.path.join(header['DATAPATH'], 'dark',
                                 'dark_2015.02.13.fits')
        if use_dark and os.path.isfile(dark_file):
            dark = fits.open(dark_file)
            read_pattern = dark[0].data[0]
            dark.close()
        else:
            read_pattern = np.full((ny, nx), 10000.)
            read_pattern[:, ::3] -= 10
            read_pattern[:, 1::3] -= 5

        full_data = np.zeros((nz, ny, nx))
        # set all Ns to read pattern + half signal, Ds to signal + read
        full_data[::2] = -data / 2 + read_pattern
        full_data[1::2] = -data + read_pattern

        primary = fits.PrimaryHDU(data=full_data,
                                  header=header)
        hdul = fits.HDUList(primary)

    return hdul


def raw_med_nod_on(coadded=False, use_dark=True, filenum=4, dy=40):
    hdul = raw_low_nod_on(coadded=coadded, use_dark=use_dark,
                          filenum=filenum, dy=dy)
    header = hdul[0].header
    header['INSTCFG'] = 'MEDIUM'
    header['WAVENO0'] = 2150.0
    header['RESOLUN'] = 10000
    return hdul


def raw_med_flat(coadded=False, use_dark=True, filenum=3):
    hdul = raw_low_flat(coadded=coadded, use_dark=use_dark, filenum=filenum)
    header = hdul[0].header
    header['INSTCFG'] = 'MEDIUM'
    header['WAVENO0'] = 2150.0
    header['RESOLUN'] = 10000
    return hdul


def raw_high_low_nod_off(coadded=False, use_dark=True, filenum=6,
                         spacing=45, angle=0.0, header=None):
    rand = np.random.RandomState(filenum)
    fn = f'{filenum:02d}'
    if header is None:
        # set the waveno and grating angle for high-low
        header = fits.header.Header()
        header['INSTCFG'] = 'HIGH_LOW'
        header['WAVENO0'] = 442.30
        header['ECHELLE'] = 18.941

    # two patterns, two integrations, two nods, ND reads
    # 16 frames in the raw:
    #   BN BD BN BD | AN AD AN AD | BN BD BN BD | AN AD AN AD
    header['OTPAT'] = 'N0D0'
    header['NODN'] = 2
    header['NINT'] = 2
    if coadded:
        header['RAWFNAME'] = f'test.sci.100{fn}.fits'
        header['FILENAME'] = f'F0001_EX_SPE_01000101_EXEELONEXEECHL_' \
                             f'RDC_100{fn}.fits'
        nx, ny, nz = 1024, 1024, 1
    else:
        header['FILENAME'] = f'test.sci.100{fn}.fits'
        nx, ny, nz = 1032, 1024, 16

    # high-low mode, nod-off-slit
    header['DATASRC'] = 'ASTRO'
    header['OBSTYPE'] = 'OBJECT'
    header['INSTMODE'] = 'NOD_OFF_SLIT'
    header['EXPTIME'] = 100.0
    header['DATE-OBS'] = f'2015-01-01T00:{fn}:00.000'
    header['UTCSTART'] = f'00:{fn}:00.000'
    header['UTCEND'] = f'00:{filenum + 1:02d}:00.000'
    header['MISSN-ID'] = '2015-01-01_EX_F001'
    header['OBS_ID'] = f'2015-01-01_EX_F001100{fn}'
    header['AOR_ID'] = '01_0001_01'
    header['SPECTEL1'] = 'EXE_ELON'
    header['SPECTEL2'] = 'EXE_ECHL'
    header['NODBEAM'] = 'B'
    header['RESOLUN'] = 75000
    header['ALTI_STA'] = 40000.
    header['ALTI_END'] = 41000.
    header['ZA_START'] = 40.
    header['ZA_END'] = 45.
    header['FLATTAMB'] = 290.
    header['FLATEMIS'] = 0.1

    # set EXES defaults
    header = readhdr(header, check_header=False)

    # build multi-order data with Gaussian profile
    sigma = spacing // 8 * gaussian_fwhm_to_sigma
    amplitude = 300
    x = np.arange(nx)

    # B A B A
    data1 = rand.rand(ny, nx) * 2 + 1000
    data2 = rand.rand(ny, nx) * 2 + 1000
    data3 = rand.rand(ny, nx) * 2 + 1000
    data4 = rand.rand(ny, nx) * 2 + 1000
    half = spacing // 2 - 4
    for cx in range(half, nx - half, spacing):
        # A nods contain source, spectral axis along y
        gp = {'amplitude': amplitude, 'stddev': sigma,
              'mean': cx}
        g = Gaussian1D(**gp)
        data2 += g(x)
        data4 += g(x)

    # rotate a bit
    if angle != 0:
        data2 = rotate(data2, angle, missing=1000)
        data4 = rotate(data4, angle, missing=1000)

    if coadded:
        data = np.array([data1, data2,
                         data3, data4])
        # gain factor
        data /= 2.0
        error = 0.001 * np.abs(data)
        mask = np.zeros((ny, nx), dtype=int)

        # add a few spikes
        for n in range(data.shape[0]):
            sy = rand.randint(0, ny, 20)
            sx = rand.randint(0, nx, 20)
            data[n, sy, sx] += 20000

        header['EXTNAME'] = 'FLUX'
        header['PRODTYPE'] = 'readouts_coadded'
        header['BEAMTIME'] = 2.0

        primary = fits.PrimaryHDU(data=data,
                                  header=header)
        error = fits.ImageHDU(data=error, name='ERROR')
        mask = fits.ImageHDU(data=mask, name='MASK')
        hdul = fits.HDUList([primary, error, mask])
    else:
        dark_file = os.path.join(header['DATAPATH'], 'dark',
                                 'dark_2015.02.13.fits')
        if use_dark and os.path.isfile(dark_file):
            dark = fits.open(dark_file)
            read_pattern = dark[0].data[0]
            dark.close()
        else:
            read_pattern = np.full((ny, nx), 10000.)
            read_pattern[:, ::3] -= 10
            read_pattern[:, 1::3] -= 5

        full_data = np.zeros((nz, ny, nx))
        # set Ns to half signal + read_pattern
        # set Ds to A or B data + read_pattern
        full_data[0] = -data1 / 2 + read_pattern
        full_data[1] = -data1 + read_pattern
        full_data[2] = -data1 / 2 + read_pattern
        full_data[3] = -data1 + read_pattern
        full_data[4] = -data2 / 2 + read_pattern
        full_data[5] = -data2 + read_pattern
        full_data[6] = -data2 / 2 + read_pattern
        full_data[7] = -data2 + read_pattern
        full_data[8] = -data3 / 2 + read_pattern
        full_data[9] = -data3 + read_pattern
        full_data[10] = -data3 / 2 + read_pattern
        full_data[11] = -data3 + read_pattern
        full_data[12] = -data4 / 2 + read_pattern
        full_data[13] = -data4 + read_pattern
        full_data[14] = -data4 / 2 + read_pattern
        full_data[15] = -data4 + read_pattern

        # add a few spikes
        for n in range(full_data.shape[0]):
            sy = rand.randint(0, ny, 20)
            sx = rand.randint(0, nx, 20)
            full_data[n, sy, sx] -= 20000

        primary = fits.PrimaryHDU(data=full_data,
                                  header=header)
        hdul = fits.HDUList(primary)

    return hdul


def raw_high_low_flat(coadded=False, use_dark=True, filenum=5,
                      spacing=45, angle=0.0, header=None):
    rand = np.random.RandomState(filenum)
    fn = f'{filenum:02d}'
    if header is None:
        # set the waveno and grating angle for high-low
        header = fits.header.Header()
        header['INSTCFG'] = 'HIGH_LOW'
        header['WAVENO0'] = 442.30
        header['ECHELLE'] = 18.941

    # two patterns, two integrations, ND reads
    # 8 frames in raw:
    #   AN AD AN AD
    header['OTPAT'] = 'N0D0'
    header['NINT'] = 2
    if coadded:
        header['RAWFNAME'] = f'test.flat.100{fn}.fits'
        header['FILENAME'] = f'F0001_EX_SPE_01000101_EXEELONEXEECHL_' \
                             f'RDC_100{fn}.fits'
        nx, ny, nz = 1024, 1024, 1
    else:
        header['FILENAME'] = f'test.flat.100{fn}.fits'
        nx, ny, nz = 1032, 1024, 4

    # stare mode, basic obs info
    header['INSTMODE'] = 'STARE'
    header['DATASRC'] = 'CALIBRATION'
    header['OBSTYPE'] = 'FLAT'
    header['EXPTIME'] = 10.0
    header['DATE-OBS'] = f'2015-01-01T00:{fn}:00.000'
    header['UTCSTART'] = f'00:{fn}:00.000'
    header['UTCEND'] = f'00:{filenum + 1:02d}:00.000'
    header['MISSN-ID'] = '2015-01-01_EX_F001'
    header['OBS_ID'] = f'2015-01-01_EX_F001100{fn}'
    header['AOR_ID'] = '01_0001_01'
    header['SPECTEL1'] = 'EXE_ELON'
    header['SPECTEL2'] = 'EXE_ECHL'
    header['RESOLUN'] = 75000
    header['ALTI_STA'] = 40000.
    header['ALTI_END'] = 41000.
    header['ZA_START'] = 40.
    header['ZA_END'] = 45.
    header['FLATTAMB'] = 290.
    header['FLATEMIS'] = 0.1

    # set EXES defaults
    header = readhdr(header, check_header=False)

    # data is multi-order flat, spectral axis along y
    half = spacing // 2 - 4
    data = rand.rand(ny, nx) * 20 + 1000
    for cx in range(half, nx - half, spacing):
        data[:, cx - half:cx + half] += 9000

    # rotate a bit
    if angle != 0:
        data = rotate(data, angle, missing=1000)

    if coadded:
        # gain factor
        data = np.expand_dims(data, axis=0)
        data /= 2.0
        error = 0.001 * np.abs(data)
        mask = np.zeros((ny, nx), dtype=int)

        header['EXTNAME'] = 'FLUX'
        header['PRODTYPE'] = 'readouts_coadded'
        header['BEAMTIME'] = 2.0

        primary = fits.PrimaryHDU(data=data,
                                  header=header)
        error = fits.ImageHDU(data=error, name='ERROR')
        mask = fits.ImageHDU(data=mask, name='MASK')
        hdul = fits.HDUList([primary, error, mask])
    else:
        dark_file = os.path.join(header['DATAPATH'], 'dark',
                                 'dark_2015.02.13.fits')
        if use_dark and os.path.isfile(dark_file):
            dark = fits.open(dark_file)
            read_pattern = dark[0].data[0]
            dark.close()
        else:
            read_pattern = np.full((ny, nx), 10000.)
            read_pattern[:, ::3] -= 10
            read_pattern[:, 1::3] -= 5

        full_data = np.zeros((nz, ny, nx))
        # set all Ns to read pattern + half signal, Ds to signal + read
        full_data[::2] = -data / 2 + read_pattern
        full_data[1::2] = -data + read_pattern

        primary = fits.PrimaryHDU(data=full_data,
                                  header=header)
        hdul = fits.HDUList(primary)

    return hdul


def raw_high_med_nod_off(coadded=False, use_dark=True, filenum=8,
                         spacing=102, angle=2.0):
    # set the waveno and grating angle for high-med
    header = fits.header.Header()
    header['INSTCFG'] = 'HIGH_MED'
    header['WAVENO0'] = 747.00931
    header['ECHELLE'] = 53.923
    header['SDEG'] = 120.0
    header['XDFL0'] = 88.2

    hdul = raw_high_low_nod_off(coadded=coadded, use_dark=use_dark,
                                filenum=filenum, spacing=spacing,
                                angle=angle, header=header)

    return hdul


def raw_high_med_flat(coadded=False, use_dark=True, filenum=7,
                      spacing=102, angle=2.0):
    # set the waveno and grating angle for high-med
    header = fits.header.Header()
    header['INSTCFG'] = 'HIGH_MED'
    header['WAVENO0'] = 747.00931
    header['ECHELLE'] = 53.923
    header['SDEG'] = 120.0
    header['XDFL0'] = 88.2

    hdul = raw_high_low_flat(coadded=coadded, use_dark=use_dark,
                             filenum=filenum, spacing=spacing,
                             angle=angle, header=header)

    return hdul


def flat_hdul(mode='low'):
    if mode == 'medium':
        hdul = raw_med_flat(coadded=True)
    elif mode == 'high_med':
        hdul = raw_high_med_flat(coadded=True)
    elif mode == 'high_low':
        hdul = raw_high_low_flat(coadded=True)
    else:
        hdul = raw_low_flat(coadded=True)

    flat_data = hdul[0].data
    flat_header = hdul[0].header
    flat_var = hdul[1].data**2

    flat_param = makeflat(flat_data, flat_header, flat_var)

    new_hdul = fits.HDUList()
    new_hdul.append(fits.PrimaryHDU(flat_param['flat'], flat_param['header']))
    new_hdul.append(fits.ImageHDU(np.sqrt(flat_param['flat_variance'])))
    new_hdul.append(fits.ImageHDU(flat_param['illum']))

    new_hdul[0].header['PRODTYPE'] = 'flat'
    new_hdul[0].header['FILENAME'] = flat_header['FILENAME'].replace(
        'RDC', 'FLT')

    return new_hdul


def cross_dispersed_flat_header():
    mode = {'INSTCFG': 'HIGH_LOW',
            'WAVENO0': 442.3,
            'WAVECENT': 22.0,
            'NORDERS': 22,
            'ORDERS': '22,21,20,19,18,17,16,15,14,13,12,11,'
                      '10,9,8,7,6,5,4,3,2,1',
            'ORDR_B': '954,909,864,819,774,729,683,638,593,548,503,'
                      '458,412,367,322,277,232,187,141,96,51,6',
            'ORDR_T': '989,944,899,854,809,764,718,673,628,583,538,493,'
                      '447,402,357,312,267,222,176,131,86,41',
            'ORDR_S': '46,42,38,34,30,26,22,17,13,9,5,2,2,2,2,2,2,2,2,2,2,2',
            'ORDR_E': '1021,1021,1021,1021,1021,1021,1021,1021,1021,'
                      '1021,1016,1010,1005,1000,995,990,985,980,976,'
                      '971,966,961',
            'ROTATION': 3,
            'RESOLUN': 100000.0,
            'RP': 98000.0}

    header = readhdr(fits.Header(mode), check_header=False)
    return header


def single_order_flat_header():
    mode = {'INSTCFG': 'LOW',
            'WAVENO0': 547.0,
            'NORDERS': 1,
            'ORDERS': '1',
            'ORDR_B': '324',
            'ORDR_T': '673',
            'ORDR_S': '2',
            'ORDR_E': '1021',
            'ROTATION': 0}
    header = readhdr(fits.Header(mode), check_header=False)
    return header


def nodsub_hdul(mode='low', do_flat=True):
    if mode == 'medium':
        hdul = raw_med_nod_on(coadded=True)
    elif mode == 'high_med':
        hdul = raw_high_med_nod_off(coadded=True)
    elif mode == 'high_low':
        hdul = raw_high_low_nod_off(coadded=True)
    else:
        hdul = raw_low_nod_on(coadded=True)
    data = hdul[0].data
    var = hdul[1].data ** 2

    # for all modes, should have two nods: B A B A
    shape = (2, data.shape[1], data.shape[2])

    new_data = np.zeros(shape)
    new_data[0] = data[1] - data[0]
    new_data[1] = data[3] - data[2]

    new_var = np.zeros(shape)
    new_var[0] = var[1] + var[0]
    new_var[1] = var[3] + var[2]

    hdul[0].data = new_data
    hdul[1].data = np.sqrt(new_var)
    hdul[2].data = np.zeros(shape, dtype=int)
    hdul[0].header['PRODTYPE'] = 'nods_subtracted'
    hdul[0].header['FILENAME'] = (
        hdul[0].header['FILENAME']).replace('RDC', 'NSB')

    if do_flat:
        flat = flat_hdul(mode)
        hdul.append(fits.ImageHDU(flat[0].data, flat[0].header, name='FLAT'))
        hdul.append(fits.ImageHDU(flat[1].data, name='FLAT_ERROR'))
        hdul.append(fits.ImageHDU(flat[2].data, name='FLAT_ILLUMINATION'))

        copy_keys = [
            'DETROT', 'HRFL', 'HRFL0',
            'HRR', 'KROT', 'XDFL', 'XDFL0', 'SLITROT',
            'WNO0', 'WAVENO0',
            'BB_TEMP', 'BNU_T', 'FLATTAMB', 'FLATEMIS',
            'SPACING', 'NT', 'NORDERS',
            'ORDERS', 'XORDER1', 'NBELOW',
            'ORDR_B', 'ORDR_T', 'ORDR_S', 'ORDR_E',
            'SLTH_ARC', 'SLTH_PIX', 'ROTATION', 'RP']
        for key in copy_keys:
            if key in flat[0].header:
                hdul[0].header[key] = flat[0].header[key]

    return hdul
