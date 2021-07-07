# Licensed under a 3-clause BSD style license - see LICENSE.rst

from pandas import DataFrame

import sofia_redux.instruments.forcast.configuration as dripconfig
from sofia_redux.instruments.forcast.hdrequirements \
    import hdrequirements, parse_condition


class TestHdrequirements(object):

    def test_parse_condition(self):
        ors = ['K1<1 & K2>=2']
        ors += ['K3=3 & K4!=4']
        ors = ' | '.join(ors)
        expect = [[('K1', '<', '1'), ('K2', '>=', '2')],
                  [('K3', '==', '3'), ('K4', '!=', '4')]]
        assert parse_condition(ors) == expect
        assert parse_condition('<G> >> 2a') is None
        assert parse_condition('K1<<<1') is None

    def test_hdrequirements(self):
        default = hdrequirements()
        assert isinstance(default, DataFrame)
        assert len(default) > 0
        empty = hdrequirements('/does/not/exist.txt')
        assert isinstance(empty, DataFrame)
        assert len(empty) == 0

    def test_hdreq_errors(self):
        # test configuration load
        dripconfig.configuration = None
        default = hdrequirements()
        assert len(default) > 0
