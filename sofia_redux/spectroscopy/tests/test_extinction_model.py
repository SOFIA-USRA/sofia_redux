# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.spectroscopy.extinction_model import ExtinctionModel
import pytest


def test_extinction_failure():
    with pytest.raises(AttributeError) as err:
        ExtinctionModel('model_does_not_exist')
        assert "model not available" in str(err)


def test_extinction_model():
    model = ExtinctionModel('rieke1989')
    assert np.allclose(model([3650, 4400, 5500]), [1.64, 1, 0])
    assert np.isclose(model(7000), -0.78)
    assert np.isnan(model(3000))
    model = ExtinctionModel('rieke1989', extrapolate=True)
    assert not np.isnan(model(3000))
    model = ExtinctionModel('rieke1989', cval=0)
    assert model(3000) == 0
    model = ExtinctionModel('nishiyama2009')
    assert np.isclose(model(3650), 1.531 / 0.112)
