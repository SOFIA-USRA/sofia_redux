# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.spectroscopy.speccor import speccor
import pytest
import warnings


@pytest.fixture
def spectra():
    spectra = np.zeros(256)[None] + np.arange(5)[:, None]
    spectra[2, 100] = np.nan
    spectra = spectra.astype(float)
    spectra += np.random.random(spectra.shape)
    return spectra


def test_invalid_input():
    assert speccor(np.arange(5)) is None
    assert speccor(np.zeros((5, 5)), err_stack=np.arange(5)) is None


def test_expected_values(spectra):
    info = {}
    result, err = speccor(spectra, err_stack=spectra.copy(), info=info)
    means = np.nanmean(result, axis=1)
    assert np.allclose(means, means[0], atol=0.1)
    assert np.nanstd(result[0]) > np.nanstd(result[4])
    valid = np.isfinite(result)
    assert np.allclose(err[valid], result[valid])
    assert np.sum(info['correction_mask']) == 255


def test_options(spectra):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = speccor(spectra, window=1000, fwidth=0)
    assert np.isnan(result).all()

    result = speccor(spectra, select=[True, False, False, False, False])
    means = np.nanmean(result, axis=1)
    assert np.allclose(means, np.nanmean(spectra[0]), atol=0.1)

    result = speccor(spectra, select=[False, False, False, False, True])
    means = np.nanmean(result, axis=1)
    assert np.allclose(means, np.nanmean(spectra[4]), atol=0.1)

    # test reference spectrum option
    refspec = spectra[1]
    default = speccor(spectra)
    ref1 = speccor(spectra, refspec=refspec)
    assert not np.allclose(default, ref1)
    # sum of sq differences from refspec should be higher for default
    assert np.nansum((default - refspec)**2) > np.nansum((ref1 - refspec)**2)
