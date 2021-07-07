import pytest

from sofia_redux.toolkit.resampling.resample_utils import get_object_size


def test_errors():
    with pytest.raises(TypeError) as err:
        get_object_size(lambda x: 1)
    assert "does not take argument of type" in str(err.value).lower()


def test_sizes():
    assert get_object_size(None) == 16
    assert get_object_size(1) == 28
    assert get_object_size(1.0) == 24
    assert get_object_size('1') == 50
