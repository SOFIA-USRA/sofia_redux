# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.spectroscopy.mkapmask import mkapmask
from sofia_redux.spectroscopy.mkspatprof import mkspatprof
from sofia_redux.spectroscopy.rectify import rectify


def rectified_data(all_positive=True):
    """Input data for extraction."""
    shape = 100, 100
    spatcal, wavecal = np.mgrid[:100, :100] + 0.5
    ordermask = np.full((100, 100), 0)
    ordermask[1:99, :50] = 1
    ordermask[1:99, 50:] = 2
    image = np.full(shape, 2.0)
    image[12:17] += 1
    image[14] += 1
    image[78:83] += 1
    image[80] += 1
    if not all_positive:
        image[78:83] *= -1
    bitmask = np.full(shape, 1)
    variance = np.full(shape, 1.0)
    mask = np.full(shape, True)

    # rectify image with into two orders
    rectorders = rectify(image, ordermask, wavecal, spatcal,
                         variance=variance, mask=mask, bitmask=bitmask,
                         dw=1, ds=1, xbuffer=1, ybuffer=1, poly_order=1)

    # make spatial profiles
    medprof, spatmap = mkspatprof(rectorders, robust=0, bgsub=True,
                                  return_fit_profile=True)

    # make apertures
    aperture = {'aperture_radius': 2.,
                'psf_radius': 4.}
    background = [[40, 50], [60, 70]]
    for order, rectimg in rectorders.items():
        ap1 = aperture.copy()
        ap2 = aperture.copy()
        ap1['trace'] = np.array([14.] * rectimg['wave'].size)
        ap2['trace'] = np.array([80.] * rectimg['wave'].size)

        apmask = mkapmask(rectimg['spatial'], rectimg['wave'],
                          [ap1, ap2], background)
        rectimg['apmask'] = apmask

    return rectorders, medprof, spatmap
