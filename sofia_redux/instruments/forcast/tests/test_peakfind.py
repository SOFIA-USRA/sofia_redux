# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy.modeling.models import Gaussian2D

from sofia_redux.instruments.forcast.peakfind import peakfind, PeakFinder
from astropy.stats import gaussian_fwhm_to_sigma


def fake_data(one_peak=False):
    peaks = np.array([
        [377.34711, 118.49461],
        [309.15344, 158.84050],
        [240.92036, 199.25980],
        [172.72487, 239.61237],
        [104.48902, 280.03351]
    ])
    if one_peak:
        peaks = peaks[2].reshape(1, 2)
    nx = 480
    ny = 418
    fwhm = 4.5
    sigma = fwhm * gaussian_fwhm_to_sigma
    data = np.zeros((ny, nx))
    y, x = np.mgrid[:ny, :nx]
    gp = {'amplitude': 1.0, 'x_stddev': sigma, 'y_stddev': sigma}
    for peak in peaks:
        gp['x_mean'] = peak[0]
        gp['y_mean'] = peak[1]
        g = Gaussian2D(**gp)
        data += g(x, y)
        gp['amplitude'] *= -1
    return {'data': data, 'peaks': peaks}


class TestPeakfind(object):

    def test_defaults(self):
        """Check results are to with 0.1 pixels of IDL result"""
        test = fake_data()
        result = peakfind(test['data'], npeaks=len(test['peaks']))
        dx = result['x'] - test['peaks'][:, 0]
        dy = result['y'] - test['peaks'][:, 1]
        dr = ((dx ** 2) + (dy ** 2)) ** 0.5
        assert (dr < 0.1).all()

    def test_peakfind_errors(self, mocker):
        test = fake_data()

        # check object return
        pf = peakfind(test['data'], npeaks=len(test['peaks']),
                      return_object=True, chopnoddist=[80., 80.])
        assert isinstance(pf, PeakFinder)

        # bad input to search_peaks
        result = pf.search_peaks(None)
        assert len(result) == 0

        # good input
        table = pf.search_peaks(test['data'])
        assert len(table) > 0

        # test maxiter
        pf.maxiter = 1
        one_pk = fake_data(one_peak=True)
        maxtab = pf.search_peaks(one_pk['data'])
        assert len(maxtab) == 1

        # test bad inputs in chopnod_sort --
        # should all just quietly leave table alone

        # bad table
        pf.chopnod_sort(None)

        # bad chop/nod dist
        inp = table.copy()
        pf.chopdist = None
        pf.chopnod_sort(inp)
        assert table is not inp
        assert np.allclose(table['ycentroid'], inp['ycentroid'])
        pf.chopdist = 80.

        # missing centroid
        inp.remove_column('xcentroid')
        assert 'xcentroid' not in inp.columns
        pf.chopnod_sort(inp)
        assert np.allclose(table['ycentroid'], inp['ycentroid'])

        # bad input to findpeaks

        # nan image
        pf.image *= np.nan
        assert pf.findpeaks() is None

    def test_reference(self):
        """Test the image comparison with a reference"""
        test = fake_data()
        offset = (5, 10)
        reference = np.roll(test['data'], offset, axis=(0, 1))
        shifts = peakfind(reference, test['data'])
        for x, y in zip(shifts['x'], shifts['y']):
            assert np.allclose(offset[0], y, atol=1e-3)
            assert np.allclose(offset[1], x, atol=1e-3)

    def test_positive(self):
        """Check the positive keyword works"""
        test = fake_data()
        npeaks = len(test['peaks'])
        result_standard = peakfind(test['data'], npeaks=npeaks)
        # allow negative fits to check they are properly removed
        pos = peakfind(test['data'], npeaks=npeaks, positive=True)
        eps = 1e-3
        npos = 0
        for i1 in range(len(result_standard)):
            row = result_standard.iloc[i1]
            match = pos[(abs(pos['x'] - row['x']) < eps)
                        & (abs(pos['y'] - row['y']) < eps)
                        & (abs(pos['peak'] - row['peak']) < eps)]
            if len(match) == 1:
                npos += 1
        expectpos = (npeaks // 2) + (npeaks % 2)
        assert npos == expectpos

    def test_negative(self):
        """Check that negative peaks can be identified"""
        test = fake_data()
        npeaks = len(test['peaks'])
        negative = peakfind(test['data'], npeaks=npeaks,
                            positive=False)
        positive = peakfind(test['data'], npeaks=npeaks,
                            positive=True)
        # negative found 5 peaks, positive found 3
        assert sum(np.isnan(negative['fit_amplitude'])) == 0
        assert sum(np.isnan(positive['fit_amplitude'])) == 2

        # matching ids: 0, 2, 4 for negative, 0, 1, 2 for positive
        assert np.allclose(negative['peak'][0:5:2], positive['peak'][0:3])
