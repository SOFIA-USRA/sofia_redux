#  Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import logging
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

    def test_load_split(self, split_order_hdul, caplog):
        caplog.set_level(logging.INFO)

        model = mid_model.Order(hdul=split_order_hdul,
                                filename=split_order_hdul.filename(),
                                number=1)

        model.data.clear()

        model.load_split(split_order_hdul, split_order_hdul.filename())

        assert len(model.data) == len(split_order_hdul)
        assert 'Load split order from' in caplog.text

    def test_load_combined(self, combined_order_hdul, caplog,
                           split_order_hdul):
        caplog.set_level(logging.INFO)
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

    def test_retrieve(self, combined_order_hdul):
        model = mid_model.Order(hdul=combined_order_hdul,
                                filename=combined_order_hdul.filename(),
                                number=0)
        value = model.retrieve('flux')
        assert value is None

        value = model.retrieve('wavepos', level='low')
        assert isinstance(value, low_model.Spectrum)
        assert value.name == 'wavepos'

        value = model.retrieve('wavepos', level='raw')
        assert isinstance(value, np.ndarray)
        npt.assert_array_almost_equal(np.diff(np.diff(value)),
                                      np.zeros(len(value) - 2))

    def test_describe(self, combined_order_hdul):
        model = mid_model.Order(hdul=combined_order_hdul,
                                filename=combined_order_hdul.filename(),
                                number=0)
        details = model.describe()
        assert isinstance(details, dict)
        assert details['name'] == model.name
        assert all(details['fields'].values())
