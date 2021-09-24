# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

import sofia_redux.instruments.flitecam.mkflat as u
from sofia_redux.instruments.flitecam.tests.resources import intermediate_data


class TestMkflat(object):

    @pytest.mark.parametrize('beam', ['A', 'B'])
    def test_source_mask(self, beam):
        # make data without sources
        data = []
        for i in range(3):
            hdul = intermediate_data(dthindex=i + 1,
                                     nodbeam=beam, nx=256, ny=256)
            data.append(hdul)

        # make flat from sky
        flat = u.mkflat(data)

        # should be normalized, with error and mask
        assert np.allclose(np.nanmedian(flat['FLAT'].data), 1)
        assert np.nansum(flat['FLAT_ERROR'].data) \
            < np.nansum(flat['FLAT'].data)

        # norm value is recorded; should be near the median of
        # all input data
        flatnorm = np.nanmedian([d[0].data for d in data])
        assert np.allclose(flat[0].header['FLATNORM'], flatnorm,
                           rtol=0.1)

        # sources are marked unless all sky frames
        if beam == 'B':
            assert np.sum(flat['FLAT_BADMASK'].data > 0) == 0
        else:
            mask = flat['FLAT_BADMASK'].data
            assert np.sum(mask > 0) > 0

            # one source from each frame, at known locations
            # 5 pixels apart in x and y
            assert mask[128, 113] == 1
            assert mask[133, 118] == 1
            assert mask[138, 123] == 1

            # error is higher over masked sources
            err = flat['FLAT_ERROR'].data
            med_err = np.nanmedian(err)
            assert np.median(err[mask > 0]) > med_err

    def test_one_file(self):
        hdul = intermediate_data(nodbeam='B', nx=256, ny=256)
        flatnorm = np.nanmedian(hdul[0].data)
        flat = u.mkflat([hdul])

        # flat is normalized version of input
        assert np.allclose(np.nanmedian(flat['FLAT'].data), 1)
        assert np.allclose(flat['FLAT'].data,
                           hdul[0].data / flatnorm,
                           equal_nan=True)
        assert np.allclose(flat['FLAT_ERROR'].data,
                           hdul['ERROR'].data / flatnorm,
                           equal_nan=True)
        assert np.sum(flat['FLAT_BADMASK'].data > 0) == 0
        assert np.allclose(flat[0].header['FLATNORM'], flatnorm)

    def test_error(self):
        hdul = intermediate_data(nodbeam='B', nx=256, ny=256)
        hdul[0].data *= np.nan
        with pytest.raises(ValueError) as err:
            u.mkflat([hdul])
        assert "No valid flat data" in str(err)
