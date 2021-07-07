# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np
from sofia_redux.spectroscopy.extinction_model import ExtinctionModel

__all__ = ['rieke_unred']


def rieke_unred(wave, flux, ebv, r_v=3.09, **kwargs):
    """
    De-redden a flux vector.

    The default reddening curve is that of Rieke & Lebofsky (1985 ApJ.,
    288, 618); see also the paper by Rieke, Rieke, & Paul (1989 ApJ,
    336, 752) and is valid from 0.365 to 13 microns.

    The reddening curve is output from `extinction_model.ExtinctionModel`.
    Please consult `extinction_model` documentation for additional
    models and optional keyword arguments.

    Notes
    -----
    For the 'rieke1989' model, r_v = 3.09 is recommended by Rieke et al.
    If r_v = 3.07, the Rieke curve will smoothly match the CCM curve at
    3.3 microns.

    Parameters
    ----------
    wave : float or array_like of float (shape)
        Wavelength in Angstroms, for consistency with ccm_unred.
    flux : float or array_like of float (shape)
        Calibrated flux
    ebv : float, optional
        Color excess E(B-V), scalar.  If a negative EBV is supplied
        then fluxes will be reddened rather than de-reddened.
    r_v : float, optional
        The ratio of total selective extinction R(V) = A(V) / E(B-V)
    kwargs : dict, optional
        Optional keyword/values to pass into `Extinction Model`.  Note
        That care must be taken to change `ebv` and `r_v` depending
        on the model used.  The default is to use the 'rieke1989` model.

    Returns
    -------
    unred_flux : float or numpy.ndarray of float (shape)
        The de-reddened flux.
    """
    try:
        model = ExtinctionModel(**kwargs)
    except Exception as err:
        log.error(err)
        return

    reddening = model(wave)
    isarr = hasattr(flux, '__len__')
    if isarr:
        if not isinstance(flux, np.ndarray):
            flux = np.array(flux)
        if isinstance(reddening, np.ndarray) and \
                reddening.shape != flux.shape:
            log.error("Wave and flux shape mismatch")
            return

    unred_wave = ebv * (r_v + reddening)
    unred_flux = flux * 10 ** (0.4 * unred_wave)
    return unred_flux
