# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
import numpy.testing as npt
import astropy.units as u

import sofia_redux.visualization.utils.unit_conversion as uc


class TestUnitConversion(object):

    @pytest.mark.parametrize('in_unit,out_unit',
                             [('sec', u.s), ('Me', u.Mct),
                              ('erg * cm-2 * sec-1 * sr-1 * (cm-1)-1',
                               u.erg / (u.cm ** 2 * u.k * u.s * u.sr))])
    def test_parse_unit(self, in_unit, out_unit):
        parsed = uc.parse_unit(in_unit)
        assert parsed == out_unit

    @pytest.mark.parametrize('in_unit,out_unit,factor,error',
                             [('Jy', 'W/m2', 1, False),
                              ('Jy', 'erg / (s cm2 Angstrom)', 2., False),
                              ('Jy', 'erg / (s cm2 Hz)', 0., False),
                              ('Jy', 'um', 1., True),
                              ('Jy', 'Jy', 0, False),
                              ('Jy', 'Jansky', 0, False)])
    def test_convert_flux(self, in_unit, out_unit, factor, error):
        wave = np.linspace(1, 20, 30)
        flux = np.linspace(1, 20, 30)

        if error:
            with pytest.raises(ValueError):
                uc.convert_flux(in_flux=flux, start_unit=in_unit,
                                end_unit=out_unit, wavelength=wave,
                                wave_unit='um')
        else:
            converted = uc.convert_flux(in_flux=flux, start_unit=in_unit,
                                        end_unit=out_unit, wavelength=wave,
                                        wave_unit='um')

            comparison = converted / flux * wave ** factor
            npt.assert_allclose(comparison / np.mean(comparison), 1)

    @pytest.mark.parametrize('in_unit,out_unit,factor,error',
                             [('um', 'Angstrom', 10000, False),
                              ('um', 'nm', 1000, False),
                              ('um', 'nm', 1000, False),
                              ('um', 'cm', 0.0001, False),
                              ('um', 'Jy', 0, True),
                              ('um', 'um', 1, False),
                              ('um', 'micron', 1, False)])
    def test_wave(self, in_unit, out_unit, factor, error):
        data = np.arange(5) + 1
        if error:
            with pytest.raises(ValueError):
                uc.convert_wave(wavelength=data, start_unit=in_unit,
                                end_unit=out_unit)
        else:
            converted = uc.convert_wave(wavelength=data, start_unit=in_unit,
                                        end_unit=out_unit)
            npt.assert_allclose(converted / data, factor)

    @pytest.mark.parametrize('in_unit,out_unit,xs,ys,changed,error',
                             [({'x': 'um', 'y': 'Jy'},
                               {'x': u.um, 'y': u.Jy},
                               1, 1, False, False),
                              ({'x': 'Jy', 'y': 'Jy'},
                               {'x': u.um, 'y': u.Jy},
                               1, 1, False, True),
                              ({'x': 'um', 'y': 'um'},
                               {'x': u.um, 'y': u.um},
                               1, 1, False, False),
                              ({'x': 'um', 'y': 'Jy'},
                               {'x': u.nm, 'y': u.Jy},
                               1000, 1, True, False),
                              ({'x': 'um', 'y': 'Jy'},
                               {'x': u.um, 'y': 'mJy'},
                               1, 1000, True, False),
                              ({'x': 'um', 'y': 'Jy'},
                               {'x': u.nm, 'y': u.uJy},
                               1000, 1e6, True, False),
                              ({'x': 'um', 'y': 'Jy'},
                               {'x': u.nm, 'y': 'erg/(s cm2 Hz)'},
                               1000, 1e-23, True, False),
                              ])
    def test_convert_model_fit(self, gauss_fit, in_unit, out_unit,
                               xs, ys, changed, error):
        # move the mean from zero for testing
        gauss_fit.mean_0 += 1.0
        old_fit = gauss_fit.copy()
        if error:
            with pytest.raises(ValueError) as err:
                uc.convert_model_fit(
                    gauss_fit, in_unit, out_unit, wave=np.arange(20))
            assert 'Unable to convert' in str(err)
        else:
            new_fit, status = uc.convert_model_fit(
                gauss_fit, in_unit, out_unit, wave=np.arange(20))
            assert status is changed
            assert np.allclose(new_fit.amplitude_0.value,
                               old_fit.amplitude_0.value * ys)
            assert np.allclose(new_fit.mean_0.value,
                               old_fit.mean_0.value * xs)
            assert np.allclose(new_fit.stddev_0.value,
                               old_fit.stddev_0.value * xs)
            assert np.allclose(new_fit.slope_1.value,
                               old_fit.slope_1.value * ys / xs)
            assert np.allclose(new_fit.intercept_1.value,
                               old_fit.intercept_1 * ys)

    @pytest.mark.parametrize('units,parseable',
                             [({'x': 'um', 'y': 'Jy'}, True),
                              ({'x': u.um, 'y': u.Jy}, True),
                              ({'x': 1 * u.um, 'y': 1 * u.Jy}, True),
                              ({'x': 1, 'y': None}, False)])
    def test_confirm_quantity(self, units, parseable):
        expected = {'x': u.um, 'y': u.Jy}
        unexpected = {'x': None, 'y': None}
        if parseable:
            assert uc._confirm_quantity(units) == expected
        else:
            assert uc._confirm_quantity(units) == unexpected

    def test_convert_slope(self):
        s1 = 1 * u.Jy / (1 * u.um)
        s2 = 1 * u.W / u.m ** 2 / (1 * u.Angstrom)

        equivs = u.spectral_density(1. * u.um)
        s = uc._convert_slope(s1, s2, equivs, {'x': u.um, 'y': u.Jy},
                              {'x': u.Angstrom, 'y': u.Unit('W / m2')})
        assert np.allclose(s, 2.39e-13)
