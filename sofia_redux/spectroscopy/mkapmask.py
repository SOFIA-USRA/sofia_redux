# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np
from sofia_redux.toolkit.interpolate import tabinv

__all__ = ['mkapmask']


def mkapmask(slit, wave, apertures, background=None):
    """
    Constructs a 2D aperture mask.

    First, the mask is set to zero for all values.  Then, the backgrounds
    are set to NaN without regard to the apertures.

    Then, for each aperture, the 'trace' value +/- the 'psf_radius' value
    is set to the aperture number. The full PSF apertures are indexed to
    apnum, inner aperture radii to -apnum.  PSF radius is required,
    aperture radius is optional. It is assumed that the PSF radius is
    larger than the aperture radius if present.

    The edge pixels for the PSF aperture are set to reflect their fractional
    pixel values.  To determine where the full aperture n is:

        z = (abs(mask) > n-1) & (abs(mask) <= n)

    Inner aperture radii are set to whole pixels only.  To determine
    the central region of the aperture n:

        z = (mask == -1 * n)

    Parameters
    ----------
    slit : array_like of float
        (n_values,) 1D array of slit position values (e.g. pixels,
        arcseconds).
    wave : array_like of float
        (n_values,) 1D array of wavelength coordinate values (e.g. um).
    apertures : list of dict
        Required keys and values for the dictionaries are:
            trace : float
            aperture_radius : float
            psf_radius : float
    apertures : list of dict
        Required keys and values for the dictionaries are:
            trace : float
            aperture_radius : float
            psf_radius : float
    background : list of list, optional
        Each element should be a [start, stop] pair of spatial
        coordinates indicating a background region.

    Returns
    -------
    numpy.ndarray
        (ns, nw) array of int
    """
    if not hasattr(slit, '__len__'):
        slit = [slit]
    slit = np.array(slit).astype(float)

    # mask starts as zero -- unused pixels will remain zero
    mask = np.zeros((slit.size, wave.size))

    # set background to nan
    if background is not None:
        regions = np.round(tabinv(slit, background)).astype(int)
        for region in regions:
            if len(region) == 2:
                mask[region[0]: region[1] + 1, :] = np.nan

    for api, aperture in enumerate(apertures):
        pos = aperture['trace']

        # define PSF aperture
        rad = aperture['psf_radius']
        ap = np.array([pos - rad, pos + rad])

        # this gets the effective index for aperture position
        # in the slit array, clipping to 0 at the lower edge,
        # and len(slit)-1 at the upper
        apidxs = tabinv(slit, ap)

        # define aperture radius, if available
        if 'aperture_radius' in aperture:
            aprad = aperture['aperture_radius']
            apradpos = np.array([pos - aprad, pos + aprad])
            aprad_idxs = tabinv(slit, apradpos)
        else:
            aprad_idxs = None

        for wavei, apidx in enumerate(apidxs.T):
            # this takes the floor of the identified indices,
            # so may include a fractional pixel on the lower edge,
            # and will miss any fractional pixels on the upper edge
            apint = apidx.astype(int)

            # check for overlap with previous aperture:
            # identified pixels must be either nan or 0
            maxap = apint[1] + 2 if apint[1] < len(slit) - 2 else len(slit)
            test = mask[apint[0]:maxap, wavei]
            if not np.all(np.isnan(test) | (test == 0)):
                msg = "The extraction apertures overlap. " \
                      "Please lower the aperture radii."
                log.error(msg)
                raise ValueError(msg)

            # set values to aperture number
            mask[apint[0]:apint[1] + 1, wavei] = api + 1

            # fix endpoints to reflect fractional pixels
            # Note that this assumes aperture widths are greater than
            # pixel widths.
            dap = apidx - apint
            if dap[0] > 0:
                # correct the first point to a fractional weight
                # (weight for extraction is mask - api, so full
                # pixels have weight 1)
                mask[apint[0], wavei] = api + dap[0]

            if dap[1] > 0:
                # add the next point up the slit to the aperture,
                # with a fractional weight
                mask[apint[1] + 1, wavei] = api + dap[1]

            # define aperture radius, if available
            if aprad_idxs is not None:
                # for the central aperture region, use only whole pixels
                apidx = aprad_idxs.T[wavei]
                apint[0] = int(np.ceil(apidx[0]))
                apint[1] = int(np.floor(apidx[1]))

                # set value to -1 * aperture number
                mask[apint[0]:apint[1] + 1, wavei] *= -1

    return mask
