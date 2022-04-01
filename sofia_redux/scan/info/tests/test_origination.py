# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.info.origination import OriginationInfo


class TestOriginationInfo(object):
    def test_init(self):
        info = OriginationInfo()
        assert info.organization is None
        assert info.observer is None
        assert info.log_id == 'orig'
