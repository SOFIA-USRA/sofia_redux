from sofia_redux.toolkit.resampling.resample_polynomial import \
    ResamplePolynomial

import numpy as np
import psutil
import pytest
from skimage.data import chelsea


@pytest.mark.skipif(psutil.cpu_count() < 2, reason='Require multiple CPUs')
def test_parallel():
    # Testing parallel processing is extremely difficult in pytest, so just
    # make sure that the results are consistent.

    image = chelsea().astype(float)
    s = image.shape
    rand = np.random.RandomState(42)
    bad_pix = rand.rand(*s) < 0.7  # 70 percent corruption
    bad_image = image.copy()
    bad_image[bad_pix] = np.nan

    y, x = np.mgrid[:s[0], :s[1]]
    coordinates = np.vstack([c.ravel() for c in [x, y]])

    yout = np.arange(s[0])
    xout = np.arange(s[1])

    # supply data in the form (nsets, ndata)
    data = np.empty((s[2], s[0] * s[1]), dtype=float)
    for frame in range(s[2]):
        data[frame] = bad_image[:, :, frame].ravel()

    resampler = ResamplePolynomial(coordinates, data, window=10, order=2)
    d1 = resampler(xout, yout, smoothing=0.1, relative_smooth=True,
                   order_algorithm='extrapolate', jobs=1)

    d2 = resampler(xout, yout, smoothing=0.1, relative_smooth=True,
                   order_algorithm='extrapolate', jobs=-1)

    assert np.allclose(d1, d2, equal_nan=True)
