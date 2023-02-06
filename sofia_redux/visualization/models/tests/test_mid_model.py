#  Licensed under a 3-clause BSD style license - see LICENSE.rst

import logging

import pytest
import astropy.io.fits as pf
import numpy as np
import numpy.testing as npt

from sofia_redux.visualization.models import mid_model, low_model


class TestMidModel(object):

    def test_init(self, ):
        number = 5
        model = mid_model.MidModel(hdul=None, filename=None,
                                   number=number)

        assert model.number == number
        assert model.name == ''
        assert isinstance(model.data, dict)
        assert model.enabled
        assert not model.populated()

    def test_not_implemented(self):
        model = mid_model.MidModel(hdul=None, filename=None,
                                   number=1)

        bad_functions = {'load_split': {'f': model.load_split,
                                        'args': {'hdul': None,
                                                 'filename': None}},
                         'load_combined': {'f': model.load_combined,
                                           'args': {'hdu': None,
                                                    'filename': None}},
                         'retrieve': {'f': model.retrieve,
                                      'args': {'field': None,
                                               'level': None}},
                         'describe': {'f': model.describe,
                                      'args': None}}

        for bad in bad_functions.values():
            with pytest.raises(NotImplementedError):
                if bad['args']:
                    bad['f'](**bad['args'])
                else:
                    bad['f']()

    def test_set_visibility(self):
        model = mid_model.MidModel(hdul=None, filename=None,
                                   number=1)
        assert model.enabled
        model.set_visibility(enabled=False)
        assert not model.enabled

    @pytest.mark.parametrize('field,result', [('flux', True),
                                              ('error', True),
                                              ('respnose', False)])
    def test_valid_field(self, field, result):
        model = mid_model.MidModel(hdul=None, filename=None,
                                   number=1)
        model.data = {'flux': 'one', 'error': 'two',
                      'transmission': 'three'}
        value = model.valid_field(field=field)
        assert value is result


class TestBook(object):

    @pytest.mark.parametrize('hdul,sp_count,co_count',
                             [(['one'], 0, 1, ), (['a', 'b'], 1, 0)])
    def test_init(self, caplog, mocker, hdul, sp_count, co_count):
        caplog.set_level(logging.DEBUG)
        split = mocker.patch.object(mid_model.Book, 'load_split')
        combined = mocker.patch.object(mid_model.Book, 'load_combined')
        number = 1
        model = mid_model.Book(hdul, '', number)

        assert split.call_count == sp_count
        assert combined.call_count == co_count
        assert model.name == f'Book_{number+1}'
        assert 'Initializing Book_' in caplog.text

    def test_load_split(self, forcast_hdul_image, caplog):
        caplog.set_level(logging.DEBUG)

        mid_model.Book(forcast_hdul_image,
                       forcast_hdul_image.filename(), 0)

        assert 'Loading split book' in caplog.text

    def test_load_combined(self):
        model = mid_model.MidModel(hdul=None, filename=None,
                                   number=1)
        with pytest.raises(NotImplementedError):
            model.load_combined(None, None)

    def test_retrieve(self):
        model = mid_model.MidModel(hdul=None, filename=None,
                                   number=1)
        with pytest.raises(NotImplementedError):
            model.retrieve(None, None)

    def test_describe(self):
        model = mid_model.MidModel(hdul=None, filename=None,
                                   number=1)
        with pytest.raises(NotImplementedError):
            model.describe()


class TestOrder(object):

    @pytest.mark.parametrize('hdul,sp_count,co_count',
                             [(['one'], 0, 1, ), (['a', 'b'], 1, 0)])
    def test_init(self, hdul, sp_count, co_count, mocker):
        split = mocker.patch.object(mid_model.Order, 'load_split')
        combined = mocker.patch.object(mid_model.Order, 'load_combined')

        filename = 'order.fits'
        number = 1
        model = mid_model.Order(hdul, filename, number)

        assert model.name == f'Order_{number+1}'
        assert split.call_count == sp_count
        assert combined.call_count == co_count

    @pytest.mark.parametrize('hdul,sp_count,co_count',
                             [(['one'], 0, 1, ), (['a', 'b'], 1, 0)])
    def test_init_aperture(self, hdul, sp_count, co_count, mocker):
        split = mocker.patch.object(mid_model.Order, 'load_split')
        combined = mocker.patch.object(mid_model.Order, 'load_combined')

        filename = 'order.fits'
        number = 1
        model = mid_model.Order(hdul, filename, number,
                                aperture=0, num_apertures=1)

        assert model.name == f'Order_{number+1}.1'
        assert split.call_count == sp_count
        assert combined.call_count == co_count

    def test_load_split(self, split_order_hdul, caplog):
        caplog.set_level(logging.DEBUG)

        model = mid_model.Order(hdul=split_order_hdul,
                                filename=split_order_hdul.filename(),
                                number=1)

        model.data.clear()

        model.load_split(split_order_hdul, split_order_hdul.filename())

        assert len(model.data) == len(split_order_hdul)
        assert 'Load split order from' in caplog.text

    def test_load_split_2(self, combined_order_hdul, caplog):
        caplog.set_level(logging.INFO)

        model = mid_model.Order(hdul=combined_order_hdul,
                                filename=combined_order_hdul.filename(),
                                number=2)
        model.data.clear()
        model.load_split(combined_order_hdul, combined_order_hdul.filename())

        assert len(model.data) == len(combined_order_hdul)

    def test_load_split_bad_names(self, split_order_hdul, caplog):
        split_order_hdul['spectral_flux'].name = 'flux_order'
        split_order_hdul['spectral_error'].name = 'error_order'

        model = mid_model.Order(hdul=split_order_hdul,
                                filename=split_order_hdul.filename(),
                                number=1)
        model.data.clear()
        model.load_split(split_order_hdul, split_order_hdul.filename())

        assert 'spectral_flux_order' in model.data.keys()

    def test_load_split_image(self, split_order_hdul):
        hdu = pf.ImageHDU(data=np.zeros((20, 20)),
                          header=split_order_hdul[0].header,
                          name='flux_order_01')
        split_order_hdul.append(hdu)
        model = mid_model.Order(hdul=split_order_hdul,
                                filename=split_order_hdul.filename(),
                                number=1)
        model.data.clear()
        model.load_split(split_order_hdul, split_order_hdul.filename())

        assert 'flux_order' not in model.data.keys()

    def test_load_split_full(self, exes_mrm_a2):
        hdul = pf.open(exes_mrm_a2)
        model = mid_model.Order(hdul, exes_mrm_a2, 1)
        model.data.clear()
        model.load_split(hdul, exes_mrm_a2, 1)
        assert isinstance(model, mid_model.Order)
        keys = ['SPECTRAL_FLUX', 'SPECTRAL_ERROR',
                'TRANSMISSION', 'RESPONSE', 'RESPONSE']
        assert all([key.lower() in model.data.keys() for key in keys])

    def test_load_split_full_error(self, exes_mrm_a2, mocker, caplog):
        caplog.set_level(logging.DEBUG)

        hdul = pf.open(exes_mrm_a2)
        model = mid_model.Order(hdul, exes_mrm_a2, 1)
        model.data.clear()
        model.aperture = None

        mocker.patch.object(low_model, 'Spectrum',
                            side_effect=[None, RuntimeError, RuntimeError,
                                         RuntimeError, RuntimeError])

        model.load_split(hdul, exes_mrm_a2, 1)
        assert 'Loading split order encountered error' in caplog.text

    def test_load_combined(self, combined_order_hdul, caplog,
                           split_order_hdul):
        caplog.set_level(logging.DEBUG)
        model = mid_model.Order(hdul=combined_order_hdul,
                                filename=combined_order_hdul.filename(),
                                number=0)
        model.data.clear()
        model.load_combined(hdu=combined_order_hdul[0],
                            filename=combined_order_hdul.filename())
        assert len(model.data) == len(split_order_hdul)
        assert 'Load combined order from' in caplog.text

        model.data.clear()
        data = np.expand_dims(combined_order_hdul[0].data, axis=0)
        combined_order_hdul[0].data = data
        model.load_combined(hdu=combined_order_hdul[0],
                            filename=combined_order_hdul.filename())
        assert len(model.data) == len(split_order_hdul)
        assert 'Load combined order from' in caplog.text

    @pytest.mark.parametrize('raw_units', [None, 'ct'])
    def test_load_combined_response(self, combined_order_hdul, caplog,
                                    split_order_hdul, raw_units):
        # add transmission and response arrays to combined data
        data = combined_order_hdul[0].data
        data = np.vstack([data, data[0].copy(), data[1].copy()])
        combined_order_hdul[0].data = data

        # add units
        combined_order_hdul[0].header['YUNITS'] = 'Jy'
        if raw_units is not None:
            combined_order_hdul[0].header['RAWUNITS'] = raw_units

        caplog.set_level(logging.DEBUG)
        model = mid_model.Order(hdul=combined_order_hdul,
                                filename=combined_order_hdul.filename(),
                                number=0)
        model.data.clear()
        model.load_combined(hdu=combined_order_hdul[0],
                            filename=combined_order_hdul.filename())
        assert len(model.data) == len(split_order_hdul) + 2
        assert 'Load combined order from' in caplog.text

        model.data.clear()
        data = np.expand_dims(combined_order_hdul[0].data, axis=0)
        combined_order_hdul[0].data = data
        model.load_combined(hdu=combined_order_hdul[0],
                            filename=combined_order_hdul.filename())
        assert len(model.data) == len(split_order_hdul) + 2
        assert 'Load combined order from' in caplog.text

        if raw_units is None:
            assert model.data['response'].unit == 'Me / s / Jy'
        else:
            assert model.data['response'].unit == f'{raw_units} / Jy'

    def test_retrieve(self, combined_order_hdul):
        model = mid_model.Order(hdul=combined_order_hdul,
                                filename=combined_order_hdul.filename(),
                                number=0)
        value = model.retrieve('other')
        assert value is None

        value = model.retrieve('spectral_flux')
        assert value is not None

        value = model.retrieve('wavepos', level='low')
        assert isinstance(value, low_model.Spectrum)
        assert value.name == 'wavepos'

        value = model.retrieve('wavepos', level='raw')
        assert isinstance(value, np.ndarray)
        npt.assert_array_almost_equal(np.diff(np.diff(value)),
                                      np.zeros(len(value) - 2))

    def test_retrieve_fail(self, combined_order_hdul, caplog):
        caplog.set_level(logging.DEBUG)
        model = mid_model.Order(hdul=combined_order_hdul,
                                filename=combined_order_hdul.filename(),
                                number=0)

        result = model.retrieve('bad')
        assert result is None
        assert 'Field bad not found in Order' in caplog.text

        caplog.clear()
        model.data['spectral_error_order_'] = model.data['spectral_error']
        result = model.retrieve('spectral_error')
        assert result is None
        assert 'does not uniquely identify' in caplog.text

    def test_describe(self, combined_order_hdul):
        model = mid_model.Order(hdul=combined_order_hdul,
                                filename=combined_order_hdul.filename(),
                                number=0)
        details = model.describe()
        assert isinstance(details, dict)
        assert details['name'] == model.name
        assert all(details['fields'].values())
