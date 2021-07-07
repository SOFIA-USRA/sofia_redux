#  Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import logging
import numpy as np
import astropy.io.fits as pf

from sofia_redux.visualization.models import high_model, mid_model, low_model


class TestHighModel(object):

    def test_init(self, grism_hdul):
        model = high_model.HighModel(grism_hdul)

        assert model.filename == grism_hdul.filename()
        assert model.id == grism_hdul.filename()
        assert model.index == 0
        assert model.enabled

    def test_init_other_filenames(self, grism_hdul):
        hdu = pf.ImageHDU()
        hdul = pf.HDUList(hdu)
        model = high_model.HighModel(hdul)
        assert model.filename == 'UNKNOWN'

        hdul[0].header['FILENAME'] = 'TEST'
        model = high_model.HighModel(hdul)
        assert model.filename == 'TEST'

    def test_not_implemented(self, grism_hdul):
        model = high_model.HighModel(grism_hdul)

        bad_functions = {'load_data': {'f': model.load_data, 'args': None},
                         'retrieve': {'f': model.retrieve, 'args': None},
                         'list_enabled': {'f': model.list_enabled,
                                          'args': None},
                         'valid_field': {'f': model.valid_field,
                                         'args': 'foo'},
                         'enable_orders': {'f': model.enable_orders,
                                           'args': None}}

        for bad in bad_functions.values():
            with pytest.raises(NotImplementedError):
                if bad['args']:
                    bad['f'](bad['args'])
                else:
                    bad['f']()


class TestGrism(object):

    def test_init(self, grism_hdul):
        model = high_model.Grism(grism_hdul)
        assert model.default_ndims == 1
        assert model.default_field == 'spectral_flux'
        assert isinstance(model.books, list)
        assert isinstance(model.orders, list)

    @pytest.mark.parametrize('spec_val,img_val,order_count,book_count',
                             [(True, False, 1, 0), (False, True, 0, 1),
                              (False, False, 1, 1)])
    def test_load_data_combined(self, grism_hdul, mocker, spec_val,
                                img_val, order_count, book_count):
        model = high_model.Grism(grism_hdul)

        order_mock = mocker.patch.object(high_model.Grism, '_load_order')
        book_mock = mocker.patch.object(high_model.Grism, '_load_book')
        mocker.patch.object(high_model.Grism, '_spectra_only',
                            return_value=spec_val)
        mocker.patch.object(high_model.Grism, '_image_only',
                            return_value=img_val)

        model.load_data()

        assert order_mock.call_count == order_count
        assert book_mock.call_count == book_count

    @pytest.mark.parametrize('filename,result',
                             [('CLN', True), ('DRP', True), ('LNZ', True),
                              ('STK', True), ('LOC', True), ('TRC', True),
                              ('APS', True), ('BGS', True), ('CAL', False)])
    def test_image_only(self, grism_hdul, filename, result):
        model = high_model.Grism(grism_hdul)
        model.filename = f'grism_{filename}_100.fits'

        out = model._image_only()

        assert out is result

    @pytest.mark.parametrize('filename,result',
                             [('CMB', True), ('MRG', True), ('SPC', True),
                              ('CAL', True), ('RSP', True), ('IRS', True),
                              ('CLN', False), ('BGS', False), ('COA', False)])
    def test_spectra_only(self, grism_hdul, filename, result):
        model = high_model.Grism(grism_hdul)
        model.filename = f'grism_{filename}_100.fits'

        out = model._spectra_only()

        assert out is result

    def test_load_order(self, grism_hdul):
        model = high_model.Grism(grism_hdul)
        model.orders = list()

        model._load_order()

        assert len(model.orders) == 1

    def test_load_book(self, grism_hdul):
        model = high_model.Grism(grism_hdul)
        model.books = list()

        model._load_book()

        assert len(model.books) == 1

    def test_retrieve(self, grism_hdul):
        model = high_model.Grism(grism_hdul)

        value = model.retrieve(book=True, level='high')
        assert isinstance(value, mid_model.Book)

        # value = model.retrieve(book=True, level='low')
        # assert isinstance(value, mid_model.Book)

        # value = model.retrieve(book=True, level='raw')
        # assert isinstance(value, mid_model.Book)

        value = model.retrieve(order=0, level='high')
        assert isinstance(value, mid_model.Order)

        value = model.retrieve(order=0, level='low', field='spectral_flux')
        assert isinstance(value, low_model.Spectrum)

        value = model.retrieve(order=0, level='raw', field='spectral_flux')
        assert isinstance(value, np.ndarray)

    @pytest.mark.parametrize('order,book', [(True, None), (None, True)])
    def test_retrieve_empty(self, grism_hdul, order, book):
        model = high_model.Grism(grism_hdul)
        model.books.clear()
        model.orders.clear()

        value = model.retrieve(order=order, book=book)
        assert value is None

    def test_retrieve_fail(self, grism_hdul):
        model = high_model.Grism(grism_hdul)

        with pytest.raises(RuntimeError) as msg:
            model.retrieve(book=True, order=True)
        assert 'Invalid identifier' in str(msg)

        with pytest.raises(RuntimeError) as msg:
            model.retrieve(book=True, level='mid')
        assert 'Invalid level' in str(msg)

    @pytest.mark.parametrize('field,result', [('spectral_flux', True),
                                              ('transmission', True),
                                              ('polarization', False)])
    def test_valid_field(self, field, result, grism_hdul, caplog):
        caplog.set_level(logging.DEBUG)
        model = high_model.Grism(grism_hdul)

        check = model.valid_field(field)
        if result:
            assert check
            assert 'is valid' in caplog.text
        else:
            assert not check
            assert 'is not valid' in caplog.text

    def test_enable_orders(self, grism_hdul):
        model = high_model.Grism(grism_hdul)
        for order in model.orders:
            order.enabled = False

        model.enable_orders()

        assert all([o.enabled for o in model.orders])

    def test_list_enabled(self, grism_hdul):
        model = high_model.Grism(grism_hdul)

        enabled = model.list_enabled()

        assert len(enabled['orders']) == 1
        assert len(enabled['books']) == 1

    def test_multi_aperture_split(self, multi_ap_grism_hdul):
        model = high_model.Grism(multi_ap_grism_hdul)
        assert model.num_orders == 2
        assert isinstance(model.orders, list)

        for order in range(2):
            value = model.retrieve(order=order, level='high')
            assert isinstance(value, mid_model.Order)

            value = model.retrieve(order=order, level='low',
                                   field='spectral_flux')
            assert isinstance(value, low_model.Spectrum)

    def test_multi_aperture_combined(self, multiorder_hdul_spec):
        model = high_model.Grism(multiorder_hdul_spec)
        assert model.num_orders == 10
        assert isinstance(model.orders, list)
        for order in range(10):
            value = model.retrieve(order=order, level='high')
            assert isinstance(value, mid_model.Order)

            value = model.retrieve(order=order, level='low',
                                   field='spectral_flux')
            assert isinstance(value, low_model.Spectrum)


class TestMultiOrder(object):

    def test_init_merged(self, multiorder_hdul_merged):
        model = high_model.MultiOrder(multiorder_hdul_merged)
        assert model.default_ndims == 1
        assert isinstance(model.orders, list)

    def test_init_spec(self, multiorder_hdul_spec):
        model = high_model.MultiOrder(multiorder_hdul_spec)
        assert model.default_ndims == 1
        assert isinstance(model.orders, list)

    def test_load_data(self, multiorder_hdul_spec, caplog):
        caplog.set_level(logging.INFO)
        model = high_model.MultiOrder(multiorder_hdul_spec)

        model.orders.clear()
        model.num_orders = 0

        model.load_data()

        norders = multiorder_hdul_spec[0].header['norders']
        assert model.num_orders == norders
        assert f'Loading {norders} orders from ' in caplog.text
        assert len(model.orders) == norders
        assert all([isinstance(o, mid_model.Order) for o in model.orders])

    def test_retrieve(self, multiorder_hdul_spec):
        model = high_model.MultiOrder(multiorder_hdul_spec)

        value = model.retrieve(level='high')
        assert isinstance(value, mid_model.Order)
        assert value.name == 'Order_1'

        value = model.retrieve(level='low', field='spectral_flux')
        assert isinstance(value, low_model.Spectrum)
        assert value.name == 'spectral_flux'

    def test_retrieve_fail(self, multiorder_hdul_spec, caplog):
        model = high_model.MultiOrder(multiorder_hdul_spec)

        with pytest.raises(TypeError) as msg:
            model.retrieve(order='one', level='high')
        assert 'must be an integer' in str(msg)

        caplog.set_level(logging.DEBUG)
        assert 'Need to provide field' not in caplog.text
        result = model.retrieve(level='low')
        assert result is None
        assert 'Need to provide field' in caplog.text

    def test_enable_orders(self, multiorder_hdul_spec, caplog):
        caplog.set_level(logging.DEBUG)
        model = high_model.MultiOrder(multiorder_hdul_spec)

        on_flag = [1, 2, 3]
        off_flag = [4, 5, 6, 7, 8, 9]

        model.enable_orders(on_flag)

        assert all([o.enabled if i in on_flag else not o.enabled
                    for i, o in enumerate(model.orders)])
        for flag in on_flag:
            assert f'Order {flag} to enabled' in caplog.text
        for flag in off_flag:
            assert f'Order {flag} to disabled' in caplog.text

    def test_list_enabled(self, multiorder_hdul_spec, caplog):
        caplog.set_level(logging.DEBUG)
        model = high_model.MultiOrder(multiorder_hdul_spec)

        enabled = model.list_enabled()
        assert len(enabled['books']) == 0
        assert len(enabled['orders']) == 10
        assert 'Current enabled fields' in caplog.text

    @pytest.mark.parametrize('field,result', [('spectral_flux', True),
                                              ('spectral_error', True),
                                              ('flux', False),
                                              ('transmission', True),
                                              ('wavepos', True)])
    def test_valid_field(self, multiorder_hdul_spec, caplog,
                         field, result):
        caplog.set_level(logging.DEBUG)
        model = high_model.MultiOrder(multiorder_hdul_spec)

        value = model.valid_field(field)
        assert value is result


class TestATRAN(object):

    def test_init(self, atran_hdul):
        model = high_model.ATRAN(atran_hdul)
        assert model.default_ndims == 1
        assert model.default_field == 'transmission'
        assert isinstance(model.orders, list)

    def test_load_atran(self, atran_hdul, mocker):
        pass
