# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Handle file I/O for model generation"""

import os
import numpy as np
import pandas as pd
import astropy.io.fits as pf

from sofia_redux.calibration.standard_model import thermast
from sofia_redux.calibration.standard_model import isophotal_wavelength as iso
from sofia_redux.calibration.pipecal_error import PipeCalError


def model_spectrum(infile, txt=False, dataframe=False, alpha=None, temp=None,
                   wmin=40., wmax=300., df_index=0):
    """
    Read in or generate model spectrum.

    Parameters
    ----------
    infile : str, pandas.DataFrame
        Defines how to generate the model. It can be the name of
        a file (if model is in an ASCII format or FITS format),
        set to "Blackbody" to generate a blackbody model, set to
        "PowerLaw" to generate a power law mdodel, or it could be
        a straight pandas DataFrame.
    txt : bool
        If set, `infile` is the name of an ASCII formatted file.
    dataframe : bool
        If set, `infile` is a pandas DataFrame.
    alpha : float
        The power law index to use if `infile` is set to "PowerLaw".
    temp : float
        The blackbody temperature to use if `infile` is set
        to "Blackbody".
    wmin : float, optional
        Minimum wavelength of the spectrum. Defaults to 40 microns.
    wmax : float, optional
        Maximum wavelength of the spectrum. Defaults to 300 microns.

    Returns
    -------
    wavelength : numpy.array
        Wavelengths in stellar spectrum.
    flux : numpy.array
        Flux at each wavelength in `wavelength`.
    power_law : bool
        A flag to indicate the model is a generated
        power law model.
    blackbody : bool
        A flag to indicate the model is a generated
        blackbody model.
    """
    if txt:
        power_law = False
        blackbody = False
        wstar, fstar = read_text(infile)
    elif dataframe:
        power_law = False
        blackbody = False
        try:
            filename = infile['model_file'].iloc[df_index]
        except KeyError:
            raise PipeCalError('Input dataframe is improperly formatted.')
        wstar, fstar = read_text(filename)
    elif infile.lower() == 'powerlaw':
        if not alpha:
            raise PipeCalError('Power Law model requires an index alpha')
        else:
            power_law = True
            blackbody = False
            wstar, fstar = generate_power_law(alpha, wmin=wmin, wmax=wmax)
    elif infile.lower() == 'blackbody':
        if not temp:
            raise PipeCalError('Blackbody model requires an input temperature')
        else:
            power_law = False
            blackbody = True
            wstar, fstar = generate_blackbody(temp, wmin=wmin, wmax=wmax)
    else:
        power_law = False
        blackbody = False
        wstar, fstar = read_fits(infile)

    return wstar, fstar, power_law, blackbody


def read_text(filename):
    """
    Read spectrum input file if formatted in plain text.

    Parameters
    ----------
    filename : str
        Name of file to read.

    Returns
    -------
    wavelength : numpy.array
        Wavelengths in stellar spectrum.
    flux : numpy.array
        Flux at each wavelength in `wavelength`.

    """
    with open(filename, 'r') as f:
        com = f.readline()[0]
    wavelength, flux = np.loadtxt(filename, unpack=True,
                                  usecols=(0, 1), comments=com)
    return wavelength, flux


def generate_power_law(alpha, wmin=40., wmax=300.):
    """
    Generate a power law input spectrum.

    Parameters
    ----------
    alpha : float
        Power law index.
    wmin : float, optional
        Minimum wavelength of the spectrum. Defaults to 40 microns.
    wmax : float, optional
        Maximum wavelength of the spectrum. Defaults to 300 microns.

    Returns
    -------
    wavelength : numpy.array
        Wavelengths in stellar spectrum.
    flux : numpy.array
        Flux at each wavelength in `wavelength`.

    """
    c = 2.99792e14  # cm/s
    # cm2mum = 1e4  # Convert cm to microns
    wref = 24.0
    fref = 1.0
    # nuref = c * cm2mum / wref
    nuref = c / wref
    dw = 0.005
    nw = int((wmax - wmin) / dw) + 1
    wstar = np.arange(nw) * dw + wmin
    nuarr = c / wstar
    fstar = fref * (nuarr / nuref) ** alpha

    return wstar, fstar


def generate_blackbody(temp, wmin=40., wmax=300.):
    """
    Generate a simple blackbody spectrum.

    Parameters
    ----------
    temp : float
        Temperature of the asteroid.
    wmin : float, optional
        Minimum wavelength of the spectrum. Defaults to 40 microns.
    wmax : float, optional
        Maximum wavelength of the spectrum. Defaults to 300 microns.

    Returns
    -------
    wavelength : numpy.array
        Wavelengths in stellar spectrum.
    flux : numpy.array
        Flux at each wavelength in `wavelength`.

    """
    c = 2.99792e14  # um/s
    Jy2W = 1e-26  # Convert Jy to W/m2/Hz
    dw = 0.005
    nw = int((wmax - wmin) / dw) + 1
    wstar = np.arange(nw) * dw + wmin
    fstar = np.pi * thermast.planck_function(wstar, temp) \
        * wstar ** 2 / (Jy2W * c)
    return wstar, fstar


def read_fits(infile):
    """
    Read spectrum from a FITS file.

    Parameters
    ----------
    infile : str
        Name of FITS file to read.

    Returns
    -------
    wavelength : numpy.array
        Wavelengths in stellar spectrum.
    flux : numpy.array
        Flux at each wavelength in `wavelength`.

    """
    hdul = pf.open(infile)
    data = hdul[0].data[0]
    wstar = data[0, :]
    fstar = data[1, :]
    return wstar, fstar


def read_atran(atmofile, ws, no_atm=False, wmin=40., wmax=300.):
    """
    Read in atmospheric transmission model.

    Parameters
    ----------
    atmofile : str
        Name of the ATRAN file.
    ws : numpy.array
        Wavelengths of spectrum.
    no_atm : bool
        If set, do not read in a ATRAN file.
    wmin : float, optional
        Minimum wavelength of spectrum to use. Defaults to 40 microns.
    wmax : float, optional
        Maximum wavelength of spectrum to use. Defaults to 300 microns.

    Returns
    -------
    wa : numpy.array
        Wavelengths of atmospheric transmission spectrum in microns.
    ta : numpy.array
        Transmission at each wavelength in `wa`.
    afile : str
        Full path of ATRAN file read in.
    """
    atran_location = '/dps/calibrations/ATRAN/fits/'
    local_apath = './atranfiles/fits/'
    if no_atm:
        wa = ws
        ta = np.ones_like(wa)
        afile = None
    else:
        if not os.path.isfile(atmofile):  # pragma: no cover
            if os.sep not in atmofile:
                afile = os.path.join(atran_location, atmofile)
                if not os.path.isfile(afile):
                    print(f'Cannot find ATRAN file {atmofile} in default '
                          f'location {atran_location}.\nUsing local copy.')
                    afile = os.path.join(local_apath, atmofile)
            else:
                afile = os.path.join(os.getcwd(), atmofile)
        else:
            afile = os.path.join(os.getcwd(), atmofile)
        hdul_a = pf.open(afile)
        watm = hdul_a[0].data[0]
        tatm = hdul_a[0].data[1]

        indx = (watm >= wmin) & (watm <= wmax)
        wa = watm[indx]
        ta = tatm[indx]
    return wa, ta, afile


def calibration_data_path():
    """
    Location of local calibration data.

    Returns
    -------
    caldata : str
        Full path to local data.

    """
    pkgpath = (os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))) + os.path.sep)
    caldata = os.path.join(*[pkgpath, 'data', 'models'])
    return caldata


def open_outfile_and_header(outfile, no_atm=False, afile=None, infile=None,
                            index=0):
    """
    Open the outfile and print all headers.

    Parameters
    ----------
    outfile : str, None
        Filename of ouptut file. If None, pull outfile from
        `infile` or use a default filename.
    no_atm : bool
        If set, no ATRAN file was read in.
    afile : str
        Name of ARAN file read in.

    Returns
    -------
    outf : file
        Handler pointing to open output file.

    """
    if no_atm:
        print('\nNo atmosphere')
    else:
        print(f'\nUsing ATRAN: {afile}')
    print('    lambda_ref   lambda_mean   lambda_1     lambda_pivot'
          '  lambda_eff lambda_eff_jv   lambda_iso   '
          'width       Response  F_mean Fnu_mean    ColorTerm    '
          'ColorTerm   Source_Rate      Source_Size   Source_FWHM'
          'Bkgd_Power           NEP            NEFD         MDCF'
          '     Npix   Filter')
    print('     microns       microns      microns      microns      '
          'microns microns         microns      microns'
          '       W/mJy   W/m^2/mum Jy                                 '
          'Watts            pix          arcsec Watts               '
          'W/sqrt(Hz)      Jy/sqrt(Hz)    mJy')

    if outfile is None:
        if isinstance(infile, pd.DataFrame):
            outfile = infile['cal_file'][index]
        else:
            outfile = 'flux_values.out'
    outf = open(outfile, 'w')
    outf.write(f'{outfile}\n')
    if no_atm:
        outf.write('No atmosphere\n\n')
    else:
        outf.write(f'{os.path.basename(afile)}\n\n')
    outf.write('lambda_ref   lambda_mean  lambda_1     lambda_pivot '
               'lambda_eff  lambda_eff_jv lambda_iso  '
               'width        Response    F_mean       Fnu_mean     '
               'ColorTerm    ColorTerm    Source_Rate Source_Size  '
               'Source_FWHM  Bkgd_Power   NEP          NEFD        '
               'MDCF         Npix                                   Filter\n')
    outf.write('microns      microns      microns      microns      '
               'microns     microns       microns     '
               'microns      W/mJy       W/m^2/mum    '
               'Jy                                     Watts       '
               'pix          arcsec       '
               'Watts        W/sqrt(Hz)   Jy/sqrt(Hz) mJy\n\n')
    return outf


def report_result(result, filter_name, outf):
    """
    Format the result of the calibration and report it.

    Parameters
    ----------
    result : dict
        Collection of calibration results.
    filter_name : str
        Name of the current filter.
    outf : IOStream
        Open output file to write to.

    Returns
    -------
    None

    """
    s = (f"{result['lambda_c']:.5e}  {result['lambda_mean']:.5e}  "
         f"{result['lambda_1']:.5e}  "
         f"{result['lambda_pivot']:.5e}  "
         f"{result['lambda_eff']:.5e}  "
         f"{result['lambda_eff_jv']:.5e}  {result['isophotal_wt']:.5e}  "
         f"{result['width']:.5e}  {result['response']:.5e}  "
         f"{result['flux_mean']:.5e}  {result['flux_nu_mean']:.5e}  "
         f"{result['color_term_k0']:.5e}  "
         f"{result['color_term_k1']:.5e}  {result['source_rate']:.5e}  "
         f"{result['source_size']:.5e} "
         f"{result['source_fwhm']:.5e}  {result['background_power']:.5e}  "
         f"{result['nep']:.5e}  "
         f"{result['nefd']:.5e}  {result['mdcf']:.5e}  "
         f"{result['npix_mean']:.5e}  {result['lambda_prime']:.5e}  "
         f"{result['lamcorr']:.5e}  "
         f"{os.path.basename(filter_name)}")
    print(s)
    outf.write(s + '\n')


def plot_spectrum(model_wave, model_flux, power_law, blackbody,
                  isophotal_weight, calibration_results, outfile=None):
    """
    Generate a plot of the final spectrum.

    Parameters
    ----------
    ws : numpy.array
        Wavelength of spectrum.
    fs : numpy.array
        Flux at each wavelength in `ws`.
    power_law : bool
        If set, spectrum was generated from a power law.
    blackbody : bool
        If set, spectrumw as generated from an ideal blackbody.
    lam_iso_wt : numpy.array
        ISO weighted wavelength in each filter.
    Nf : int
        Number of filters.
    outfile : str
        Name of the output file, that the file the plots
        is saved to is based on.

    Returns
    -------
    None

    """
    from matplotlib.backends.backend_agg \
        import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    print('\nPlotting isophotal wavelengths')
    fig = Figure(figsize=(10, 10))
    FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)

    fmin = np.min(model_flux)
    fmax = np.max(model_flux)
    f_mean = calibration_results['flux_mean']
    isophotal_weight = calibration_results['isophotal_wt']
    if power_law == 1 or blackbody == 1:
        ax.plot(model_wave, model_flux)
    else:
        ax.step(model_wave, model_flux, where='mid')
    ax.plot(isophotal_weight, f_mean, color='red', linestyle='--')
    print('Lambda_iso      <F_lambda>      F_lambda(lambda_iso)')
    for i in range(len(isophotal_weight)):
        ax.scatter(isophotal_weight.iloc[i], f_mean.iloc[i],
                   marker='d', color='k')
        fiso = iso.interpol(model_flux, model_wave, isophotal_weight.iloc[i])
        print(f'{isophotal_weight[i]:.5e}\t{f_mean[i]:.5e}\t{fiso:.5e}')

    ax.set_ylabel([fmin, fmax])
    ax.set_yscale('log')
    ax.set_xlabel('Wavelength (micron)')
    ax.set_ylabel('Flux (W/m2/micron)')

    if outfile is None:
        plotname = 'spectrum.png'
    else:
        plotname = '.'.join(outfile.split('.')[:-1]) + '.png'
    fig.savefig(plotname, bbox_inches='tight', dpi=300)
    print(f'Plotting to {plotname}')


def unique_wavelengths(wavelengths, flux, wmin=40., wmax=300.):
    """
    Select out a window of unique wavelengths.

    Parameters
    ----------
    wavelengths : numpy.array
        Full list of wavelengths in microns.
    flux : numpy.array
        Flux at each wavelength in `wavelengths`.
    wmin : float, optional
        Minimum wavelength to allow. Defaults to 40 microns.
    wmax :float, optional
        Maximum wavelength to allow. Defaults to 300 microns.

    Returns
    -------
    ws : numpy.array
        Unique values of `wavelengths` between `wmin` and `wmax`.
    fs : numpy.array
        Flux at each wavelength in `ws`.

    """

    wsin, indicies = np.unique(wavelengths, return_index=True)
    fsin = flux[indicies]
    indx = (wsin >= wmin) & (wsin <= wmax)
    ws = wsin[indx]
    fs = fsin[indx]
    return ws, fs
