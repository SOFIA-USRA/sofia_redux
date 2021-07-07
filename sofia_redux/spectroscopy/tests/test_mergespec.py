# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.spectroscopy.mergespec import mergespec
import pytest


@pytest.fixture
def spectra():
    spec1 = np.zeros((4, 10))
    spec1[0] = np.arange(10)
    spec1[1] = 1
    spec1[2] = 3
    spec1[3] = np.arange(10) % 2
    spec2 = np.zeros((4, 10))
    spec2[0] = np.arange(10) + 5.5
    spec2[1] = 2
    spec2[2] = 4
    spec2[3] = 2
    return spec1, spec2


def test_invalid_input(spectra):
    spec1, spec2 = spectra
    with pytest.raises(ValueError) as err:
        mergespec(spec1[0], spec2[0])
    assert "must have 2 dimensions" in str(err.value)
    with pytest.raises(ValueError) as err:
        mergespec([spec1[0]], [spec2[0]])
    assert "must have 2 or more rows" in str(err.value)
    with pytest.raises(ValueError) as err:
        mergespec(spec1, spec2[:-1])
    assert "must have equal rows" in str(err.value)


def test_no_overlap(spectra):
    spec1, spec2 = spectra
    spec2[0] += 100
    info = {}
    result = mergespec(spec1, spec2, info=info)
    assert np.isnan(info['overlap_range']).all()

    u = np.append(spec1, spec2, axis=1)
    u[1:, spec1.shape[1] - 1: spec1.shape[1] + 1] = np.nan
    assert np.allclose(u, result, equal_nan=True)

    # same result with spectra switched:
    # range is dynamically determined
    result = mergespec(spec2, spec1, info=info)
    assert np.allclose(u, result, equal_nan=True)


def test_edge_overlap(spectra):
    spec1, spec2 = spectra
    info = {}
    result = mergespec(spec1, spec2, info=info)
    assert np.allclose(result[:, :4], spec1[:, :4])
    assert np.allclose(result[:, -4:], spec2[:, -4:])
    assert np.allclose(result[:, 6], [6, 1.27, 2.56, 2], atol=0.01)
    assert np.allclose(result[:, 7], [7, 1.27, 2.56, 3], atol=0.01)
    assert np.allclose(info['overlap_range'], [6, 8])


def test_inside_overlap(spectra):
    spec1, spec2 = spectra
    spec2 = spec2[:, :3]
    info = {}
    result = mergespec(spec1, spec2, info=info)
    assert np.allclose(info['overlap_range'], [6, 7])
    assert np.allclose(result[:, :6], spec1[:, :6])
    assert np.allclose(result[:, 8:], spec1[:, 8:])
    assert np.allclose(result[:, 6], [6, 1.27, 2.56, 2], atol=0.01)
    assert np.allclose(result[:, 7], [7, 1.27, 2.56, 3], atol=0.01)


def test_dimensions_and_sum(spectra):
    ospec1, ospec2 = spectra
    spec1, spec2 = ospec1[:3], ospec2[:3]
    result = mergespec(spec1, spec2)
    assert np.allclose(result[:, 6], [6, 1.27, 2.56], atol=0.01)
    result = mergespec(spec1, spec2, sum_flux=True)
    assert np.allclose(result[:, 6], [6, 3, 5.74], atol=0.01)
    spec1, spec2 = ospec1[:2], ospec2[:2]
    result = mergespec(spec1, spec2)
    assert np.allclose(result[:, 6], [6, 1.5])
    result = mergespec(spec1, spec2, sum_flux=True)
    assert np.allclose(result[:, 6], [6, 3])


def test_nans(spectra):
    ospec1, ospec2 = spectra
    spec1, spec2 = ospec1.copy(), ospec2.copy()
    spec1[0, 0] = np.nan
    result = mergespec(spec1, spec2)
    assert result[0, 0] == 1
    spec1 = ospec1.copy()
    spec1[1, 0] = np.nan
    result = mergespec(spec1, spec2)
    assert result[0, 0] == 1
    spec1 = ospec1.copy()
    spec1[1, 7] = np.nan
    result = mergespec(spec1, spec2)
    assert np.allclose(result[:, 7], [7, np.nan, 0, 3], equal_nan=True)
    result = mergespec(spec1, spec2, sum_flux=True)
    assert np.allclose(result[:, 7], [7, np.nan, 5.74, 3],
                       equal_nan=True, atol=0.01)
