# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.units import imperial
import numpy as np
import pytest

from sofia_redux.scan.custom.sofia.integration.models.atran import AtranModel

imperial.enable()
kft = units.Unit('ft') * 1000
degree = units.Unit('degree')


@pytest.fixture
def atran_options():
    return {
        'amcoeffs': ['0.9995', '-0.1089', '0.02018', '0.008359', '-0.006565'],
        'altcoeffs': ['0.9994', '0.01921', '-0.0001924', '-0.0003502',
                      '-2.141e-05', '1.974e-05'],
        'reference': '0.682'}


@pytest.fixture
def atran(atran_options):
    return AtranModel(atran_options)


def test_class():
    assert AtranModel.kft == kft
    assert np.isclose(AtranModel.reference_airmass, np.sqrt(2))
    assert AtranModel.reference_altitude == 41 * kft


def test_init(atran_options):
    model = AtranModel(atran_options)
    assert np.allclose(model.am_coeffs,
                       [0.9995, -0.1089, 0.02018, 0.008359, -0.006565],
                       atol=1e-6)
    assert np.allclose(model.alt_coeffs,
                       [9.994e-01, 1.921e-02, -1.924e-04, -3.502e-04,
                        -2.141e-05, 1.974e-05], rtol=1e-3)
    assert model.reference_transmission == 0.682
    assert isinstance(model.poly_am, np.poly1d)
    assert np.allclose(model.poly_am,
                       [-0.006565, 0.008359, 0.02018, -0.1089, 0.9995],
                       atol=1e-3)
    assert isinstance(model.poly_alt, np.poly1d)
    assert np.allclose(model.poly_alt,
                       [1.974e-05, -2.141e-05, -3.502e-04, -1.924e-04,
                        1.921e-02, 9.994e-01], rtol=1e-3)


def test_configure_options(atran):
    model = atran
    with pytest.raises(ValueError) as err:
        model.configure_options(None)
    assert "Options must be" in str(err.value)
    options = {}
    with pytest.raises(ValueError) as err:
        model.configure_options(options)
    assert 'atran.amcoeffs' in str(err.value)
    options['amcoeffs'] = [0.1, 0.2, 0.3]
    with pytest.raises(ValueError) as err:
        model.configure_options(options)
    assert 'atran.altcoeffs' in str(err.value)
    options['altcoeffs'] = '0.4, 0.5, 0.6'
    with pytest.raises(ValueError) as err:
        model.configure_options(options)
    assert 'atran.reference' in str(err.value)
    options['reference'] = '0.7'
    model.configure_options(options)
    assert np.allclose(model.am_coeffs, [0.1, 0.2, 0.3])
    assert np.allclose(model.alt_coeffs, [0.4, 0.5, 0.6])
    assert model.reference_transmission == 0.7
    assert isinstance(model.poly_am, np.poly1d)
    assert np.allclose(model.poly_am, [0.3, 0.2, 0.1])
    assert isinstance(model.poly_alt, np.poly1d)
    assert np.allclose(model.poly_alt, [0.6, 0.5, 0.4])


def test_get_relative_transmission(atran):
    model = atran
    assert np.isclose(model.get_relative_transmission(41, 45),
                      0.9989003, atol=1e-6)
    assert np.isclose(model.get_relative_transmission(41 * kft, 45 * degree),
                      0.9989003, atol=1e-6)
    assert np.isclose(model.get_relative_transmission(40, 30),
                      0.9249583, atol=1e-6)


def test_get_zenith_tau(atran):
    assert np.isclose(atran.get_zenith_tau(41, 45), 0.271406, atol=1e-6)
    assert np.isclose(atran.get_zenith_tau(42, 70), 0.306295, atol=1e-6)
    assert np.isclose(atran.get_zenith_tau(40 * kft, 30 * degree),
                      0.230366, atol=1e-6)
