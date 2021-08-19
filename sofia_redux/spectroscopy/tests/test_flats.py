# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.spectroscopy.flats import FlatInfo, Flat, FlatBase
import sofia_redux.spectroscopy.tests


@pytest.fixture
def flat_info_file():
    return os.path.join(
        os.path.dirname(sofia_redux.spectroscopy.tests.__file__),
        'data', 'H1_flatinfo.fits')


def test_ordermask(flat_info_file):
    fname = flat_info_file
    flat = FlatInfo(fname)
    norders = flat.orders.size
    assert norders > 0
    assert flat.edgecoeffs.shape == (norders, 2, flat.edgedeg + 1)
    assert flat.xranges.shape == (norders, 2)
    assert not np.allclose(flat.xranges, flat.guesspos)
    assert np.allclose(flat.omask, flat.generate_order_mask(offset=1298))


def test_adjust_guesspos(flat_info_file):
    fname = flat_info_file
    flat = FlatInfo(fname)
    mid_order = int(np.unique(flat.orders[flat.norders // 2]))
    image = np.zeros(flat.shape)
    image[flat.omask == mid_order] = 1.0
    shifted = np.roll(image, 3, axis=0)
    flat.adjust_guess_position(shifted, mid_order)
    assert np.allclose(
        flat.guesspos - flat._default_guesspos, [[0, 3]])
    with pytest.raises(ValueError):
        flat.adjust_guess_position(shifted, 99999999)
        flat.omask[0, 0] = 99999999
        flat.adjust_guess_position(shifted, 99999999)

    flat.adjust_guess_position(shifted, order=mid_order,
                               ybuffer=100000000000)
    assert np.allclose(flat.xranges, -1)

    # test error in order specification
    flat.orders = [311, 312, 313]
    with pytest.raises(ValueError) as err:
        flat.adjust_guess_position(shifted, order=314,
                                   ybuffer=100000000000)
    assert 'Order 314 not present in orders' in str(err)


def test_flat_class(flat_info_file):
    fname = flat_info_file
    flat = Flat(fname)

    # same as for FlatInfo
    norders = flat.orders.size
    assert norders > 0
    assert flat.edgecoeffs.shape == (norders, 2, flat.edgedeg + 1)
    assert flat.xranges.shape == (norders, 2)
    assert not np.allclose(flat.xranges, flat.guesspos)
    assert np.allclose(flat.omask, flat.generate_order_mask())

    # test repr
    repr_flat = repr(flat)
    assert 'filename' in repr_flat
    assert os.path.basename(fname) in repr_flat

    # 2D data: variance, flags None
    shape = flat._data.shape
    assert flat.variance is None
    assert flat.flags is None

    # test 3D data: assumes image, variance, flags
    flat._data = np.zeros((5, *shape))
    flat.parse_info()
    assert flat.variance.shape == shape
    assert flat.flags.shape == shape


def test_load_error(flat_info_file, tmpdir):
    fname = 'badfile.fits'
    with pytest.raises(ValueError) as err:
        Flat(fname)
    assert 'Could not load: badfile.fits' in str(err)

    # data can be empty for base class, as long as required
    # keywords are present in a header
    flatfile = fits.open(flat_info_file)
    hdr = flatfile[0].header
    hdul = fits.HDUList(fits.PrimaryHDU(header=hdr))
    fname = str(tmpdir.join('emptyfile.fits'))
    hdul.writeto(fname)
    flat = FlatBase(fname)
    assert flat._data is None
