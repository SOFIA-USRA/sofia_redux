# Licensed under a 3-clause BSD style license - see LICENSE.rst

import sofia_redux.instruments.forcast.configuration as dripconfig
from sofia_redux.instruments.forcast.read_section import read_section


class TestReadSection(object):

    def test_read_section(self):
        dripconfig.load()
        config = dripconfig.configuration
        section = [int(val) for val in config['nlinsection']]
        default_section = 128, 128, 200, 200
        result = read_section(256, 256)
        for v1, v2 in zip(result, section):
            assert v1 == v2
        result = read_section(0, 0)
        assert result == default_section

    def test_readsec_errors(self, capsys):
        # default for comparison
        dripconfig.load()
        conf_section = read_section(256, 256)
        default_section = 128, 128, 200, 200

        # test config load
        dripconfig.configuration = None
        result = read_section(256, 256)
        assert result == conf_section

        # test missing nlinsection
        dripconfig.load()
        del dripconfig.configuration['nlinsection']
        result = read_section(256, 256)
        assert result == default_section
        capt = capsys.readouterr()
        assert 'section has not been specified' in capt.err

        # bad dimensions
        dripconfig.load()
        result = read_section('a', 256)
        assert result == default_section
        result = read_section(256, 'b')
        assert result == default_section

        # bad nlinsec

        # non-int value
        dripconfig.configuration['nlinsection'] = [128, 128, 190, 'c']
        result = read_section(256, 256)
        assert result == default_section
        capt = capsys.readouterr()
        assert 'wrong format' in capt.err

        # wrong number of values
        dripconfig.configuration['nlinsection'] = [128, 128, 190]
        result = read_section(256, 256)
        assert result == default_section
        capt = capsys.readouterr()
        assert 'wrong format' in capt.err

        # size too small
        dripconfig.configuration['nlinsection'] = [128, 128, 5, 5]
        result = read_section(256, 256)
        assert result == default_section
        capt = capsys.readouterr()
        assert 'wrong size' in capt.err

        # bad x center
        dripconfig.configuration['nlinsection'] = [256, 128, 190, 190]
        result = read_section(256, 256)
        assert result == default_section
        capt = capsys.readouterr()
        assert 'wrong section size along x' in capt.err

        # bad y center
        dripconfig.configuration['nlinsection'] = [128, 256, 190, 190]
        result = read_section(256, 256)
        assert result == default_section
        capt = capsys.readouterr()
        assert 'wrong section size along y' in capt.err
