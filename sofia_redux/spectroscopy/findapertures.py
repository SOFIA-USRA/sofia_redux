# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
import numpy as np

from sofia_redux.toolkit.fitting.fitpeaks1d import fitpeaks1d
from sofia_redux.toolkit.interpolate import tabinv

__all__ = ['find_apertures']


def find_apertures(profiles, npeaks=1, orders=None, positions=None,
                   fwhm=1.0, fix=False, **kwargs):
    """
    Determine the position of the aperture(s) in a spatial profile.

    Profiles expected are the median profiles produced by
    `sofia_redux.spectroscopy.mkspatprof`.  Peaks are fit using a
    Gaussian model, with initial estimates optionally provided by the user.

    Any additional keyword arguments provided are passed to the
    `sofia_redux.toolkit.fitting.fitpeaks1d` algorithm, used to fit
    the Gaussian model to the profile.

    Parameters
    ----------
    profiles : dict
        order (int) -> profile (numpy.ndarray)
            (n_spatial, 2) spatial profile where profile[:, 0] = spatial
            coordinate and profile[:, 1] = median spatial profile.
    npeaks : int, optional
        Number of peaks to find in each profile.
    orders : list of int, optional
        Orders to extract.  If not present, all orders in `profiles` will
        be extracted.
    positions : dict, optional
        order (int) -> list of float
            Up to (npeaks) positions to use as the starting point(s)
            for the fit.
    fwhm : float, optional
        Starting estimate for Gaussian FWHM of the spatial peak,
        in arcsec up the slit.
    fix : bool, optional
        If set, apertures will be fixed to the input positions.  If input
        positions are not specified, they will be fixed to the center of
        the slit.

    Returns
    -------
    apertures : dict
        order (int) -> list of dict
            Keys and values are as follows:
               position : float
               fwhm : float
               sign : {1, -1}
    """
    apertures = {}
    if orders is None:
        orders = np.unique(list(profiles.keys())).astype(int)
    else:
        orders = np.unique(orders).astype(int)

    # Gaussian width from FHWM
    sigma = gaussian_fwhm_to_sigma * fwhm

    for order in orders:
        # x and y values: slit position vs. normalized profile value
        x = profiles[order][0]
        y = profiles[order][1]
        dx = np.mean(x[1:] - x[:-1])

        # guess positions if provided
        if positions is None or order not in positions:
            guess = None
        else:
            guess = positions[order]

        if fix:
            # if positions not specified, divide up the slit by npeaks
            if guess is None:
                guess = []
                dx = len(x) / (npeaks + 1)
                for i in range(1, npeaks + 1):
                    idx = int(np.round(i * dx))
                    guess.append(x[idx])

            # set the guess positions as the apertures
            # with the given FWHM and a sign derived from the profile
            aplist = []
            for pos in guess:
                sign = 1 if y[int(tabinv(x, pos))] > 0 else -1
                aplist.append({'position': pos,
                               'fwhm': fwhm,
                               'sign': sign})
        else:
            # find highest peaks in the profile

            # bounds on fit position and FHWM: keep inside the slit
            bounds = {}
            for i in range(npeaks):
                bounds['mean_{}'.format(i)] = (x[0], x[-1])
                bounds['stddev_{}'.format(i)] = (dx, x[-1])

            fit_peaks = fitpeaks1d(x, y, npeaks=npeaks, guess=guess,
                                   stddev=sigma, bounds=bounds,
                                   **kwargs)

            # assuming the npeaks are the first n models in the output;
            # background fit is last
            aplist = []
            for i in range(npeaks):
                # fit position
                fit_pos = fit_peaks[i].mean.value

                # fit FHWM
                fit_fwhm = fit_peaks[i].stddev.value * gaussian_sigma_to_fwhm

                # sign at fit position
                sign = 1 if y[int(tabinv(x, fit_pos))] > 0 else -1

                aplist.append({'position': fit_pos,
                               'fwhm': fit_fwhm,
                               'sign': sign})

        apertures[order] = sorted(aplist, key=lambda j: j['position'])

    return apertures
