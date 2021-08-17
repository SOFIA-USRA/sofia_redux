# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.instruments.flitecam.expmap import expmap
from sofia_redux.instruments.flitecam.tests.resources import intermediate_data


class TestExpmap(object):

    def test_success(self):
        hdul = intermediate_data()
        assert 'EXPOSURE' not in hdul

        result = expmap(hdul)
        assert 'EXPOSURE' in result

        # exposure map matches flux data, has exptime value
        assert result['EXPOSURE'].data.shape == hdul['FLUX'].data.shape
        assert np.allclose(result['EXPOSURE'].data, hdul[0].header['EXPTIME'])
        assert result['EXPOSURE'].header['BUNIT'] == 's'
