# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Scale models to account for object distance"""

import os
import pandas as pd
import numpy as np
from sofia_redux.calibration.pipecal_error import PipeCalError


def modconvert(infile, outfile, scale_factor=1.0):
    """
    Scale a Herschel model by a constant factor.

    Parameters
    ----------
    infile : str
        Name of file containing Herschel model.
    outfile : str
        Name of file to write scaled model to.
    scale_factor : float
        Factor to scale model in `infile` by.

    Returns
    -------
    model : pandas.DataFrame
        The scaled model.

    """
    index, freq, brightness_temp, flux, rj_temp = read_infile(infile)

    wave, flux, temp = sort_spectrum(freq, flux, brightness_temp)

    scaled_flux = scale_factor * flux

    plot_scaled_spectrum(wave, scaled_flux, scale_factor, infile)

    write_scaled_spectrum(wave, scaled_flux, scale_factor, temp,
                          infile, outfile)

    model = pd.DataFrame({'wavelength': wave, 'flux': scaled_flux,
                          't_br': temp})
    return model


def read_infile(infile):
    """
    Read in a Herschel model.

    Parameters
    ----------
    infile : str
        Name of Herchel file.

    Returns
    -------
    index : numpy.array
        Index value of each row in `infile`.
    freq : numpy.array
        Frequency of each row in `infile`.
    tbr : numpy.array
        Brightness temperature of each row in `infile`.
    flux : numpy.array
        Flux value of each row in `infile`.
    trj : numpy.array
        Rayleigh-Jeans temperature value of each row in `infile`.

    """
    # There are a variable header lengths possible.
    # Loop through and look for when the line starts
    # with '1', the first index.
    nheader = 0
    try:
        with open(infile, 'r') as f:
            for line in f:
                if line.strip().startswith('1'):
                    break
                nheader += 1
    except IOError:
        message = f'Unable to open {infile} in modconvert.'
        raise PipeCalError(message)
    index, freq, tbr, flux, trj = np.genfromtxt(infile, unpack=True,
                                                skip_header=nheader)
    return index, freq, tbr, flux, trj


def sort_spectrum(freq, flux, brightness_temp):
    """
    Sort the flux and brightness temp by increasing wavelength.

    Parameters
    ----------
    freq : numpy.array
        Frequency data.
    flux : numpy.array
        Flux data.
    brightness_temp : numpy.array
        Brightness temperature data.

    Returns
    -------
    w : numpy.array
        Wavelength of each data point.
    f : numpy.array
        Flux at each wavelength in `w`.
    t : numpy.array
        Brightness temperature at each wavelength in `w`.

    """
    # Speed of light in microns/sec
    clight = 2.9979e14
    wave = clight / (freq * 1e9)

    # Sort the data by wavelength
    sortind = np.argsort(wave)
    w = wave[sortind]
    f = flux[sortind]
    t = brightness_temp[sortind]

    return w, f, t


def plot_scaled_spectrum(wave, scaled_flux, scale_factor, infile):
    """
    Plot the scaled Herschel model.

    Parameters
    ----------
    wave : numpy.array
        Wavelength data of spectrum.
    scaled_flux : numpy.array
        Flux data of spectrum.
    scale_factor : numpy.array
        Scale factor applied to Herschel model.
    infile : str
        Name of Herschel model.

    Returns
    -------
    None

    """
    from matplotlib.backends.backend_agg \
        import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 10))
    FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(wave, scaled_flux)
    ax.set_xlim([30, 300])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Wavelength (microns)')
    ax.set_ylabel('Flux (Jy)')
    ax.set_title(f'Scale = {scale_factor:.3f}')
    fig.savefig(f'scaled_flux_{os.path.basename(infile).split(".")[0]}.png',
                bbox_inches='tight')


def write_scaled_spectrum(wave, scaled_flux, scale_factor, temp,
                          infile, outfile):
    """
    Write the scaled Herschel model to file.

    Parameters
    ----------
    wave : numpy.array
        Wavelength data of spectrum.
    scaled_flux : numpy.array
        Flux data of spectrum.
    scale_factor : float
        Scale factor applied to Herschel model.
    temp : numpy.array
        Brightness temperature data of spectrum.
    infile : str
        Name of Herschel file.
    outfile : str
        Name of file to create with scaled spectrum.

    Returns
    -------
    None

    """
    with open(outfile, 'w') as outf:
        outf.write('; {0:s}\n'.format(infile))
        outf.write('; FSCALE = {}\n'.format(scale_factor))
        outf.write('; Wave (microns) Flux (Jy)      T_br (K) '
                   '     Fscale = {0:.3f}\n'.format(scale_factor))

        for i in range(len(wave)):
            outf.write(f'{wave[i]:.6f}\t{scaled_flux[i]:.6f}\t{temp[i]:.6f}\n')
