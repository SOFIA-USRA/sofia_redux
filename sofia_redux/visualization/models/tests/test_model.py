#  Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import logging
import astropy.io.fits as pf

from sofia_redux.visualization.models.model import Model
from sofia_redux.visualization.models.model import general, is_number
from sofia_redux.visualization.models import high_model
import numpy as np


class TestModel(object):

    def test_add_model_args(self, grism_hdul, multiorder_hdul_spec):
        with pytest.raises(RuntimeError) as msg:
            Model.add_model()
        assert 'Need to provide' in str(msg)

        with pytest.raises(RuntimeError) as msg:
            Model.add_model('test.fits', grism_hdul)
        assert 'not both' in str(msg)

    def test_add_model_hdul(self, grism_hdul, multiorder_hdul_spec):
        obj = Model.add_model(hdul=grism_hdul)
        assert isinstance(obj, high_model.Grism)

        obj = Model.add_model(hdul=multiorder_hdul_spec)
        assert isinstance(obj, high_model.MultiOrder)

    def test_add_model_filename(self, grism_hdul, mocker, caplog):
        caplog.set_level(logging.DEBUG)
        mocker.patch.object(pf, 'open', return_value=grism_hdul)
        filename = 'test.fits'

        obj = Model.add_model(filename)

        assert isinstance(obj, high_model.Grism)
        assert obj.filename == filename
        assert obj.id == filename
        assert 'Created model' in caplog.text

    def test_add_model_fail(self, grism_hdul):
        hdul = pf.HDUList()
        hdu = grism_hdul['FLUX'].copy()
        hdul.append(hdu)
        with pytest.raises(NotImplementedError):
            Model.add_model(hdul=hdul)

        grism_hdul[0].header['instrume'] = 'HAWC'
        with pytest.raises(NotImplementedError):
            Model.add_model(hdul=grism_hdul)

    def test_add_model_misc(self, grism_hdul):
        hdul = pf.HDUList()
        hdu = grism_hdul['FLUX'].copy()
        hdul.append(hdu)
        hdul[0].header['instrume'] = None
        obj = Model.add_model(hdul=hdul)
        assert isinstance(obj, high_model.Grism)

    def test_add_model_general(self, spectrum_file):
        obj = Model.add_model(filename=spectrum_file)
        assert isinstance(obj, high_model.Grism)


def test_general(spectrum_file, spectrum_file_comma, spectrum_file_noheader):
    files = [spectrum_file, spectrum_file_comma, spectrum_file_noheader]
    for file in files:
        general_model = general(file)
        assert general_model[0].header['XUNITS'] == 'um'
        assert general_model[0].header['YUNITS'] == 'Jy'
        assert general_model[0].header['instrume'] == 'General'
        assert isinstance(general_model, pf.HDUList)


def test_general_single(spectrum_file_single):
    general_model = general(spectrum_file_single)
    data = general_model[0].data
    tt = np.arange(data.shape[1])
    assert np.mean(data[0, :]) == np.mean(tt)


def test_general_multi(spectrum_file_multi):
    general_model = general(spectrum_file_multi)
    data = general_model[0].data

    assert data.shape[0] == 5
    assert data[0, 0] == 3.0392023
    assert data[1, 0] == 10
    assert data[2, 0] == 0.1
    assert data[3, 0] == 7
    assert data[4, 0] == 0.2


def test_general_units(spectrum_file_units, spectrum_file_units_paran):
    files = [spectrum_file_units, spectrum_file_units_paran]
    for file in files:
        general_model = general(file)
        assert general_model[0].header['XUNITS'] == 'cm'
        assert general_model[0].header['YUNITS'] == 'ergs/s'


def test_general_parse_error(tmpdir, mocker):
    filename = tmpdir.join('bad_file.txt')

    # bad headers
    filename.write('bad columns\n1 2\n')
    with pytest.raises(RuntimeError) as err:
        general(str(filename))
    assert 'Unexpected columns' in str(err)

    # bad data
    filename.write('wave flux\n1 2 3\n4 5 6 7\n8\n')
    with pytest.raises(RuntimeError) as err:
        general(str(filename))
    assert 'Could not parse' in str(err)


@pytest.mark.parametrize('s,result',
                         [('5.7', True), ('-1.0', True), ('aa', False)])
def test_is_number(s, result):
    out = is_number(s)
    assert out is result
