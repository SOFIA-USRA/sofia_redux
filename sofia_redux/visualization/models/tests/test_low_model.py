#  Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import logging
import numpy as np
import numpy.testing as npt
import astropy.units as u

from sofia_redux.visualization.models import low_model
from sofia_redux.visualization.utils import unit_conversion as uc


class TestLowModel(object):

    def test_init(self, spectrum_hdu):
        filename = 'test.fits'

        model = low_model.LowModel(spectrum_hdu, filename)

        assert model.hdu == spectrum_hdu
        assert model.filename == filename
        npt.assert_array_equal(model.data, spectrum_hdu.data)
        assert model.unit is None
        assert model.unit_key == ''
        assert model.name == spectrum_hdu.name.lower()
        assert model.kind == ''
        assert len(model.kind_names) == 0
        assert model.default_ndims == 0
        assert model.default_field is None
        assert model.id is None
        assert model.enabled

    def test_define_units(self):
        units = low_model.LowModel._define_units()
        assert isinstance(units, dict)
        keys = ['scale', 'unitless', 'position', 'flux', 'wavelength']
        assert all([key in units.keys() for key in keys])

    def test_get_unit(self, spectrum_hdu):
        model = low_model.LowModel(spectrum_hdu, 'test.fits')
        key = 'Jy'
        model.unit_key = key
        value = model.get_unit()
        assert value == key

    def test_equal(self, spectrum_hdu):
        model_1 = low_model.LowModel(spectrum_hdu, 'test.fits')
        model_2 = low_model.LowModel(spectrum_hdu, 'test.fits')
        assert model_1 == model_2

        model_1 = low_model.LowModel(spectrum_hdu, 'test_1.fits')
        model_2 = low_model.LowModel(spectrum_hdu, 'test_2.fits')
        assert model_1 != model_2

    @pytest.mark.parametrize('unit,key,kind,result,out_key',
                             [('Jansky', 'YUNITS', 'flux', u.Jy, 'Jy'),
                              ('erg / s', 'YUNIT', 'flux', u.erg / u.s,
                               'erg / s'),
                              ('micron', 'BUNIT', 'wavelength', u.um,
                               'micron'),
                              ('(cm-1)', 'XUNIT', 'wavelength', u.kayser,
                               '1 / cm')])
    def test_parse_units(self, spectrum_hdu, key, kind, unit, result, out_key):
        spectrum_hdu.header[key] = unit
        model = low_model.LowModel(spectrum_hdu, 'test.fits',
                                   kind=kind)

        assert model.unit is None
        model._parse_units()

        assert model.unit == result
        assert model.unit_key == out_key

    def test_verify_unit_parse(self, spectrum_hdu, caplog):
        caplog.set_level(logging.DEBUG)
        model = low_model.LowModel(spectrum_hdu, 'test.fits',
                                   kind='flux')
        model.unit = 'jansky'
        with pytest.raises(RuntimeError) as msg:
            model._verify_unit_parse()
        assert 'Failure to parse unit' in str(msg)

        model.unit = u.um
        model.unit_key = 'micron'
        assert 'micron' not in model.available_units['flux']
        model._verify_unit_parse()
        assert 'Non-standard unit found' in caplog.text
        assert 'micron' in model.available_units['flux']

    def test_parse_kind(self, spectrum_hdu):
        model = low_model.LowModel(spectrum_hdu, 'test.fits')
        with pytest.raises(RuntimeError) as msg:
            model._parse_kind()
        assert 'Failed to parse kind' in str(msg)

    def test_convert(self, spectrum_hdu):
        model = low_model.LowModel(spectrum_hdu, 'test.fits')
        with pytest.raises(NotImplementedError):
            model.convert('', '', '')

    def test_retrieve(self, spectrum_hdu):
        model = low_model.LowModel(spectrum_hdu, 'test.fits')
        value = model.retrieve()
        assert isinstance(value, np.ndarray)


class TestImage(object):

    def test_init(self, image_hdu):
        model = low_model.Image(image_hdu, 'image.fits')
        assert model.default_ndims == 2
        assert model.kind == 'flux'
        assert model.kind_names['unitless'] == ['badmask', 'spatial_map']

    def test_data_mean(self, image_hdu):
        model = low_model.Image(image_hdu, 'image.fits')
        mean = model.data_mean()
        npt.assert_almost_equal(mean, 0.011, decimal=3)

        model.data = None
        mean = model.data_mean()
        assert np.isnan(mean)

    def test_convert(self, image_hdu):
        model = low_model.Image(image_hdu, 'image.fits')

        with pytest.raises(NotImplementedError):
            model.convert('', '', '')


class TestSpectrum(object):

    def test_init(self, spectrum_hdu):
        filename = 'test.fits'
        model = low_model.Spectrum(spectrum_hdu, filename)
        assert model.id == f'{filename}/{spectrum_hdu.name.lower()}'
        assert model.kind_names['flux'] == ['spectral_flux', 'spectral_error']
        assert model.kind == 'flux'

    @pytest.mark.parametrize('kind,fc_count,wv_count,vv_count,fail',
                             [('flux', 1, 0, 1, False),
                              ('wavelength', 0, 1, 1, False),
                              ('position', 0, 1, 1, False),
                              ('scale', 0, 0, 0, False),
                              ('unitless', 0, 0, 0, False),
                              ('other', 0, 0, 0, True)])
    def test_convert(self, spectrum_hdu, mocker, caplog, kind, fc_count,
                     wv_count, vv_count, fail):
        caplog.set_level(logging.ERROR)
        flux_convert = mocker.patch.object(uc, 'convert_flux',
                                           return_value=None)
        wave_convert = mocker.patch.object(uc, 'convert_wave',
                                           return_value=None)
        verify = mocker.patch.object(low_model.LowModel, '_verify_unit_parse',
                                     return_value=None)

        model = low_model.Spectrum(spectrum_hdu, 'spec.fits')
        model.kind = kind
        if fail:
            with pytest.raises(ValueError) as msg:
                model.convert('Jy', None, 'um')
            assert 'Unknown conversion kind' in str(msg)
        else:
            model.convert('Jy', None, 'um')
            assert flux_convert.call_count == fc_count
            assert wave_convert.call_count == wv_count
            assert verify.call_count == vv_count + 1

    def test_convert_unrecognized(self, spectrum_hdu):
        # set an unrecognized unit in the header
        spectrum_hdu.header['YUNIT'] = 'bad'
        model = low_model.Spectrum(spectrum_hdu, 'spec.fits')
        model.kind = 'flux'
        wave = np.arange(len(spectrum_hdu.data))

        # attempted conversion to the same bad unit should be allowed
        model.convert('bad', wave, 'um')
        assert model.unit == 'bad'

        # attempted conversion to a different bad unit should fail
        with pytest.raises(ValueError):
            model.convert('other', wave, 'um')
