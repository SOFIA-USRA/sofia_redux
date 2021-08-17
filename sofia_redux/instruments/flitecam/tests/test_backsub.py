# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.instruments.flitecam.backsub import backsub
from sofia_redux.instruments.flitecam.tests.resources import intermediate_data


class TestBacksub(object):

    def test_flatnorm(self, capsys):
        hdul = intermediate_data()
        flux = hdul['FLUX'].data.copy()
        err = hdul['ERROR'].data.copy()

        # subtract flatnorm value; error not affected
        hdul[0].header['FLATNORM'] = 100
        result = backsub(hdul, method='flatnorm')
        assert np.allclose(result['FLUX'].data, flux - 100, equal_nan=True)
        assert np.allclose(result['ERROR'].data, err, equal_nan=True)

        # header keywords updated
        assert result[0].header['BGSOURCE'] == 'FLATNORM keyword'
        assert result[0].header['BGVALUE'] == 100

        # missing flatnorm value
        hdul = intermediate_data()
        result = backsub(hdul, method='flatnorm')
        assert np.allclose(result['FLUX'].data, flux, equal_nan=True)
        assert np.allclose(result['ERROR'].data, err, equal_nan=True)
        assert result[0].header['BGSOURCE'] == 'FLATNORM keyword'
        assert result[0].header['BGVALUE'] == 0
        assert 'FLATNORM keyword is missing' in capsys.readouterr().err

    def test_median(self):
        hdul = intermediate_data()
        flux = hdul['FLUX'].data.copy()
        err = hdul['ERROR'].data.copy()
        medval = np.nanmedian(flux)

        # subtract median value; error not affected
        result = backsub(hdul, method='median')
        assert np.allclose(result['FLUX'].data, flux - medval, equal_nan=True)
        assert np.allclose(result['ERROR'].data, err, equal_nan=True)

        # header keywords updated
        assert result[0].header['BGSOURCE'] == 'Image median'
        assert result[0].header['BGVALUE'] == medval

    def test_bgfile(self):
        hdul = intermediate_data()
        flux = hdul['FLUX'].data.copy()
        err = hdul['ERROR'].data.copy()

        sky = intermediate_data(nodbeam='B', dthindex=2)

        # subtract sky, error is propagated
        result = backsub(hdul, sky)
        assert np.allclose(result['FLUX'].data, flux - sky['FLUX'].data,
                           equal_nan=True)
        var = err ** 2 + sky['ERROR'].data ** 2
        assert np.allclose(result['ERROR'].data, np.sqrt(var),
                           equal_nan=True)

        # header keywords updated
        assert result[0].header['BGSOURCE'] == sky[0].header['FILENAME']
        assert result[0].header['BGVALUE'] == np.nanmedian(sky['FLUX'].data)

        # no error present: flux is corrected, error is not
        hdul = intermediate_data()
        del sky['ERROR']
        result = backsub(hdul, sky)
        assert np.allclose(result['FLUX'].data, flux - sky['FLUX'].data,
                           equal_nan=True)
        assert np.allclose(result['ERROR'].data, err, equal_nan=True)
