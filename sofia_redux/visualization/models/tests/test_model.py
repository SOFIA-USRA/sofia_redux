#  Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import logging
import astropy.io.fits as pf

from sofia_redux.visualization.models.model import Model
from sofia_redux.visualization.models import high_model


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
