# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.instruments.exes import cirrus


class TestCirrus(object):

    def test_cirrus(self, rdc_high_low_hdul, capsys):
        data = rdc_high_low_hdul[0].data
        header = rdc_high_low_hdul[0].header
        abeams = [1, 3]
        bbeams = [0, 2]
        flat = np.ones(data.shape[1:])

        cdata = cirrus.cirrus(data.copy(), header, abeams, bbeams, flat)
        assert cdata.shape == data.shape

        # warns for continuum removal, logs noise parameters
        capt = capsys.readouterr()
        assert capt.out.count('Sky noise parameters') == 2
        assert 'removes average continuum' in capt.err

        # should bring the mean closer to zero in the A data;
        # B data are left alone
        assert not np.allclose(cdata, data)
        # B
        assert np.allclose(cdata[0], data[0])
        assert np.allclose(cdata[2], data[2])
        # A
        assert np.abs(np.mean(cdata[1])) < np.abs(np.mean(data[1]))
        assert np.abs(np.mean(cdata[3])) < np.abs(np.mean(data[3]))

    def test_cirrus_errors(self, rdc_high_low_hdul, capsys, mocker):
        data = rdc_high_low_hdul[0].data
        header = rdc_high_low_hdul[0].header
        abeams = [1, 3]
        bbeams = [0, 2]
        flat = np.ones(data.shape[1:])

        # data is generally just returned on error

        # bad dimensions
        bad_data = data[0, 1]
        cdata = cirrus.cirrus(bad_data, header, abeams, bbeams, flat)
        assert cdata is bad_data
        assert 'wrong dimensions' in capsys.readouterr().err

        # bad beams
        test_data = data.copy()
        cdata = cirrus.cirrus(test_data, header, [1], bbeams, flat)
        assert cdata is test_data
        assert np.allclose(cdata, data)
        assert 'A and B beams' in capsys.readouterr().err

        # bad flat
        cdata = cirrus.cirrus(test_data, header, abeams, bbeams,
                              np.zeros(data.shape[1:]))
        assert cdata is test_data
        assert np.allclose(cdata, data)
        assert 'No good points in flat' in capsys.readouterr().err

        # inversion error
        mocker.patch('numpy.linalg.inv', side_effect=np.linalg.LinAlgError)
        cdata = cirrus.cirrus(test_data, header, abeams, bbeams, flat)
        assert cdata is test_data
        assert np.allclose(cdata, data)
        assert 'Could not find sky parameters' in capsys.readouterr().err
