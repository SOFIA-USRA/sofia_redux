# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

import numpy as np
import pytest

from sofia_redux.instruments.fifi_ls.get_lines import get_lines
from sofia_redux.instruments.fifi_ls.tests.resources import FIFITestCase


class TestGetLines(FIFITestCase):

    def test_errors(self, capsys, tmpdir, mocker):
        # test for missing/bad defaults file
        os.makedirs(tmpdir.join('line_lists'))
        default = tmpdir.join('line_lists', 'primary_lines.txt')
        default.write('test\n')

        # mock the data path
        mock_file = tmpdir.join('test_file')
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.__file__', str(mock_file))

        with pytest.raises(ValueError):
            get_lines()
        capt = capsys.readouterr()
        assert 'Cannot read line list file' in capt.err

    def test_success(self):
        wavelength, name = get_lines()
        assert isinstance(wavelength, list)
        assert isinstance(name, list)
        assert len(wavelength) == len(name)
        assert np.min(wavelength) > 0
        assert np.max(wavelength) < 1000
