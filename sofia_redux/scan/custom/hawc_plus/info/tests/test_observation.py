# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.custom.hawc_plus.info.observation import \
    HawcPlusObservationInfo


def test_is_aor_valid():
    info = HawcPlusObservationInfo()
    info.aor_id = 'foo'
    assert info.is_aor_valid()
    info.aor_id = '0'
    assert not info.is_aor_valid()
