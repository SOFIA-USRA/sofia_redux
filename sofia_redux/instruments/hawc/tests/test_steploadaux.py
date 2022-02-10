# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

import pytest

from sofia_redux.instruments.hawc.datatext import DataText
from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steploadaux import StepLoadAux
from sofia_redux.instruments.hawc.tests.resources import DRPTestCase


class TestStepLoadAux(DRPTestCase):
    def test_load(self):
        step = StepLoadAux()

        # defaults
        step.loadauxsetup()
        assert step.auxpar == 'aux'
        pkeys = [p[0] for p in step.paramlist]
        assert 'auxfile' in pkeys

        # test key
        step.loadauxsetup(auxpar='test')
        assert step.auxpar == 'test'
        pkeys = [p[0] for p in step.paramlist]
        assert 'testfile' in pkeys

    def test_getdataobj(self):
        step = StepLoadAux()

        # text data
        df = step.getdataobj('test.txt')
        assert isinstance(df, DataText)

        # FITS data
        df = step.getdataobj('test.fits')
        assert isinstance(df, DataFits)

    def test_loadauxname(self, tmpdir, capsys):
        step = StepLoadAux()
        step.loadauxsetup()

        # default, no config -- raises error
        with pytest.raises(ValueError):
            step.loadauxname()
        capt = capsys.readouterr()
        assert 'No aux files' in capt.err

        # config parameters to find a single file in tmpdir
        aux1 = tmpdir.join('aux1.txt')
        aux1.write('# testkey = A\n'
                   '# DATE-OBS = 2019-02-20T02:00:00.000\n'
                   'testval1')
        # auxfile param
        step.paramlist[0][1] = os.path.join(str(tmpdir), 'aux*.txt')
        # fitkey param
        step.paramlist[2][1] = [[]]
        result = step.loadauxname()
        assert result == str(aux1)
        capt = capsys.readouterr()
        assert 'Found unique file' in capt.out

        # find two files, but still no fitkeys
        aux2 = tmpdir.join('aux2.txt')
        aux2.write('# testkey = B\n'
                   '# DATE-OBS = 2019-03-01T12:00:00.000\n'
                   'testval2')
        # multi = False
        result = step.loadauxname()
        assert result == str(aux1)
        capt = capsys.readouterr()
        assert 'No fitkeys: Return first' in capt.out
        # multi = True - return all
        result = step.loadauxname(multi=True)
        assert result == [str(aux1), str(aux2)]
        capt = capsys.readouterr()
        assert 'No fitkeys: return all' in capt.out

        # set fitkeys, datain to match aux2
        df = DataText()
        df.header['testkey'] = 'B'
        step.datain = [df]
        step.paramlist[2][1] = ['testkey']
        result = step.loadauxname()
        assert result == str(aux2)
        capt = capsys.readouterr()
        assert 'Matching aux found' in capt.out

        # good fitkeys, but no match -- returns first, or all if multi
        df.header['testkey'] = 'C'
        result = step.loadauxname()
        assert result == str(aux1)
        result = step.loadauxname(multi=True)
        assert result == [str(aux1), str(aux2)]
        capt = capsys.readouterr()
        assert 'NO MATCH' in capt.err

        # multi with two matches
        df.header['testkey'] = 'B'
        aux3 = tmpdir.join('aux3.txt')
        aux3.write('# testkey = B\n'
                   '# DATE-OBS = 2019-04-01T12:00:00.000\n'
                   'testval3')
        result = step.loadauxname(multi=True)
        assert result == [str(aux2), str(aux3)]
        capt = capsys.readouterr()
        assert 'Matching aux found' in capt.out

        # bad fitkeys value
        step.paramlist[2][1] = [{'a': 1}]
        with pytest.raises(TypeError):
            step.loadauxname()
        capt = capsys.readouterr()
        assert 'incorrect format' in capt.err

        # date-obs case - should also return aux2
        step.paramlist[2][1] = ['DATE-OBS']
        # missing key -- gives error
        with pytest.raises(KeyError):
            step.loadauxname()
        capt = capsys.readouterr()
        assert 'Missing DATE-OBS' in capt.err

        # key present
        df.header['DATE-OBS'] = '2019-03-01T00:00:00.000'
        step.datain = df
        result = step.loadauxname()
        assert result == str(aux2)
        capt = capsys.readouterr()
        assert 'Matching aux found' in capt.out

        # load matching aux files
        df = step.loadauxfile()
        assert isinstance(df, DataText)
        assert df.data == ['testval2']

        result = step.loadauxfile(multi=True)
        assert isinstance(result, list)
        df = result[0]
        assert isinstance(df, DataText)
        assert df.data == ['testval2']
