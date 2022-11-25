# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.instruments.exes import correct_row_gains as crg


class TestCorrectRowGains(object):

    def make_data(self, add_noise=False):
        nx = 10
        ny = 10
        nz = 3

        data = np.full((nz, ny, nx), 1.0)
        data *= np.arange(ny, dtype=float)[None, :, None]

        # add an odd/even effect
        odd = data[:, 1::2, :]
        odd += 3.0

        if add_noise:
            rand = np.random.RandomState(42)
            data += rand.rand(nz, ny, nx) * 1e-5

        return data

    def test_correct_row_gains(self, capsys):
        data = self.make_data()

        # flat data fails fit
        cdata = crg.correct_row_gains(data * 0 + 1)
        assert np.allclose(cdata, 1.0)
        assert 'Fit failed; not correcting' in capsys.readouterr().err

        # data with and without noise should work
        data = self.make_data(add_noise=False)
        cdata = crg.correct_row_gains(data)
        assert not np.allclose(cdata, data)
        assert 'Fit failed' not in capsys.readouterr().err
        assert np.allclose(cdata[:, 1::2, :], cdata[:, :-1:2, :])

        # odd data should be corrected to even data
        assert np.allclose(cdata[:, 1::2, :], cdata[:, :-1:2, :], atol=1e-5)

        data = self.make_data(add_noise=True)
        cdata = crg.correct_row_gains(data)
        assert not np.allclose(cdata, data)
        assert 'Fit failed' not in capsys.readouterr().err

        # odd data should be corrected to even data
        assert np.allclose(cdata[:, 1::2, :], cdata[:, :-1:2, :], atol=1e-5)
