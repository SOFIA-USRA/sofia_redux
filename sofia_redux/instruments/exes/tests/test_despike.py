# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
from astropy.io import fits
import numpy as np

from sofia_redux.instruments.exes import despike as dsp
from sofia_redux.toolkit.utilities.fits import set_log_level


class TestDespike(object):

    def test_check_dimensions(self, rdc_header, rdc_data):
        data, var = rdc_data
        nz = dsp._check_dimensions(data, rdc_header)
        assert nz == 4

        nz = dsp._check_dimensions(data[0], rdc_header)
        assert nz == 1

        with pytest.raises(ValueError) as err:
            dsp._check_dimensions(np.zeros(5), rdc_header)
        assert 'Data has wrong dimensions' in str(err)

    def test_check_variance(self, rdc_data, rdc_header):
        data, var = rdc_data
        nz = dsp._check_dimensions(data, rdc_header)

        dovar = dsp._check_variance(var, rdc_header, nz)
        assert dovar is True

        dovar = dsp._check_variance(None, rdc_header, nz)
        assert dovar is False

        with pytest.raises(ValueError) as err:
            dsp._check_variance(var[0], rdc_header, nz)
        assert 'Variance has wrong dimensions' in str(err)

    def test_check_good_array(self, rdc_header):
        shape = (rdc_header['NSPEC'], rdc_header['NSPAT'])

        out_data = dsp._check_good_array(None, rdc_header)
        assert out_data.shape == shape
        assert np.all(out_data)

        num_bad = int(0.2 * out_data.shape[0] * out_data.shape[1])
        num_good = out_data.size - num_bad
        indices = np.random.choice(out_data.shape[0] * out_data.shape[1],
                                   replace=False, size=num_bad)
        out_data[np.unravel_index(indices, out_data.shape)] = False

        new_data = dsp._check_good_array(out_data, rdc_header)
        assert new_data.shape == shape
        assert new_data.sum() == num_good
        assert new_data.dtype == bool

        with pytest.raises(ValueError) as err:
            dsp._check_good_array(np.full(shape, False), rdc_header)
        assert 'No good pixels' in str(err)

    def test_apply_beams(self, rdc_data, rdc_beams):
        data, var = rdc_data

        beams = dsp._apply_beams(data, rdc_beams['a']['beam'],
                                 rdc_beams['b']['beam'])

        beam_shape = (len(rdc_beams['a']['beam']),) + data.shape[1:]
        assert all([b['data'].shape == beam_shape
                    for b in beams.values()])

        beams = dsp._apply_beams(data, [], [0, 1, 2, 3])
        assert beams['A']['data'].shape == (0, 0)
        assert beams['B']['data'].shape == (4, data.shape[1], data.shape[2])

        beams = dsp._apply_beams(data, [0, 1, 2, 3], [])
        assert beams['A']['data'].shape == (4, data.shape[1], data.shape[2])
        assert beams['B']['data'].shape == (0, 0)

        beams = dsp._apply_beams(data, [], [])
        assert beams['A']['data'].shape == (4, data.shape[1], data.shape[2])
        assert beams['B']['data'].shape == (0, 0)
        assert np.all(beams['A']['beam'] == np.arange(4))
        assert beams['B']['beam'].size == 0

        beams = dsp._apply_beams(data[0], [], [])
        assert beams['A']['data'].shape == (data.shape[1], data.shape[2])
        assert beams['B']['data'].shape == (0, 0)
        assert np.all(beams['A']['beam'] == np.arange(1))
        assert beams['B']['beam'].size == 0

    def test_read_noise_contribution(self):
        header = fits.Header()
        header['nspat'] = 1024
        header['frametim'] = 1
        header['pagain'] = 2.8
        header['beamtime'] = 16
        header['eperadu'] = 35
        header['readnois'] = 30

        out = dsp._read_noise_contribution(header)
        correct_out = {'frame_gain': 2.8,
                       'varmin': 8.16327,
                       'read_var': 0.00286990}
        for key, correct in correct_out.items():
            assert np.isclose(out[key], correct)

        header['frametim'] = -1
        out = dsp._read_noise_contribution(header)
        correct_out = {'frame_gain': 1,
                       'varmin': 1024,
                       'read_var': 0.7346938}
        for key, correct in correct_out.items():
            assert np.isclose(out[key], correct)

    def test_trash_frame_check(self):
        header = fits.Header()
        nz = 4
        sky = np.arange(nz)
        beam_name = 'A'

        tframe, tsum = dsp._trash_frame_check(sky, nz, header, beam_name)
        assert tsum == 0
        assert tframe.shape == (nz,)
        assert not np.any(tframe)

        header['TRASH'] = True
        tframe, tsum = dsp._trash_frame_check(sky, nz, header, beam_name)
        assert tsum == 2
        assert tframe.shape == (nz,)
        assert np.all(tframe == [True, False, False, True])

    def test_saturated_pixels(self, capsys):
        nz, ny, nx = 5, 5, 4
        header = fits.Header({'SATVAL': 100.0, 'NSPEC': ny, 'NSPAT': nx})
        full_sky = np.ones((nz, ny, nx),
                           dtype=float) * (np.arange(nz)[:, None, None] + 1)
        beam_data = np.arange(nz * ny * nx, dtype=float).reshape((nz, ny, nx))
        good_beam_frames = np.full(nz, True)
        good_beam_frames[1:3] = False
        frame_gain = 2.0
        good_index = np.full((ny, nx), True)

        good, scaled, pix, sky = dsp._saturated_pixels(
            header, full_sky, beam_data, good_beam_frames,
            frame_gain, good_index)

        # mean of input sky in good frames
        assert sky == (1 + 4 + 5) / 3

        # data scaled by sky, without bad frames
        assert scaled.shape == (3, ny, nx)
        assert np.allclose(scaled[0], beam_data[0] * sky)
        assert np.allclose(scaled[-1], beam_data[-1] * sky / nz)

        # mean of scaled data over frames
        assert pix.shape == (ny, nx)
        assert np.allclose(pix.mean(), scaled.mean())

        # saturated pixels (average values > 50) marked in mask
        assert good_index.sum() == ny * nx - 10
        assert '10 saturated pixels found' in capsys.readouterr().out

    def test_calculate_variance(self, capsys):
        nz, ny, nx = 5, 5, 4
        avg_pix = np.ones((ny, nx), dtype=float)
        scaled = np.arange(nz * ny * nx, dtype=float).reshape((nz, ny, nx))
        read_noise = {'eperadu': 11, 'read_var': 15, 'varmin': 1000,
                      'gain': 1.0, 'frame_time': 1.0, 'beam_time': 1.0,
                      'frame_gain': 1.0}

        with set_log_level('DEBUG'):
            var = dsp._calculate_variance(scaled, avg_pix, nz, read_noise)

        # avgvar < varmin
        assert 'Possible inadequate digitization' in capsys.readouterr().out

        assert var['avgvar'] == 800
        assert var['calcvar'] == 1 / 11 + 15
        assert var['var'].shape == (ny, nx)
        assert np.allclose(var['var'], 800)
        assert var['value'].shape == (nz, ny, nx)
        assert np.allclose(var['value'], scaled - avg_pix[None, :, :])

    def test_find_spikes(self):
        nz, ny, nx = 5, 6, 7
        header = fits.Header({'SPIKEFAC': 10})
        var_value = np.ones((nz, ny, nx)) * (np.arange(nz)[:, None, None] + 1)
        above_average = np.full((ny, nx), True)
        frame_value = np.ones((ny, nx))
        frame_value[1, 2] = 100
        frame_value[3, 4] = 100
        var_value[1] = frame_value.copy()

        idx, avg = dsp._find_spikes(header, var_value, nz,
                                    above_average, frame_value)
        assert idx.sum() == 2
        assert np.all(avg['avg1'] == 3.25)
        assert np.all(avg['avgsq1'] == 12.75)

    def test_replace_spikes(self):
        nz, ny, nx = 5, 6, 7

        spike_data = np.ones((nz, ny, nx), dtype=float)
        frame_index = 2
        spike_index = np.full((ny, nx), False)
        averages = {'avg1': np.full((ny, nx), 2.0)}
        average_pixels = np.ones((ny, nx))
        sky = 3
        average_sky = 4
        do_var = True
        variance = np.zeros((nz, ny, nx), dtype=float)

        # no spikes: nothing to do
        d = spike_data.copy()
        v = variance.copy()
        dsp._replace_spikes(d, frame_index, spike_index,
                            averages, average_pixels, sky, average_sky,
                            do_var, v, propagate_nan=False)
        assert np.allclose(d, spike_data)
        assert np.allclose(v, variance)

        # add spikes
        spike_index[1, 2] = True
        spike_index[3, 4] = True
        d = spike_data.copy()
        v = variance.copy()
        dsp._replace_spikes(d, frame_index, spike_index,
                            averages, average_pixels, sky, average_sky,
                            do_var, v, propagate_nan=False)
        # data replaced
        assert np.allclose(d[2, spike_index], (1 + 2) * 3 / 4)
        assert np.allclose(d[:, ~spike_index], 1)
        # variance is left as is
        assert np.allclose(v, variance)

        # propagate NaN
        d = spike_data.copy()
        v = variance.copy()
        dsp._replace_spikes(d, frame_index, spike_index,
                            averages, average_pixels, sky, average_sky,
                            do_var, v, propagate_nan=True)
        # data and variance replaced
        assert np.all(np.isnan(d[2, spike_index]))
        assert np.allclose(d[:, ~spike_index], 1)
        assert np.all(np.isnan(v[2, spike_index]))
        assert np.allclose(v[:, ~spike_index], 0)

    def test_despike_full_data(self, rdc_low_hdul, capsys):
        data = rdc_low_hdul[0].data
        header = rdc_low_hdul[0].header
        variance = rdc_low_hdul[1].data ** 2
        abeams = [1, 3]
        bbeams = [0, 2]
        header['SPIKEFAC'] = 20.0

        # all okay, some spikes found if mark all as A beams
        d, m, g = dsp.despike(data.copy(), header)
        capt = capsys.readouterr()
        assert 'All good frames' in capt.out
        assert len(g) == 4
        assert np.allclose(d[m], data[m])
        assert not np.allclose(d[~m], data[~m])
        assert (~m).sum() == 80

        # all trashed if beams are specified
        d, m, g = dsp.despike(data.copy(), header, variance, abeams, bbeams)
        capt = capsys.readouterr()
        assert '2 frame(s) from A are trashed' in capt.out
        assert '2 frame(s) from B are trashed' in capt.out
        assert len(g) == 0
        assert np.allclose(d, data)

        # turn off trashing, finds spikes okay
        header['TRASH'] = False
        d, m, g = dsp.despike(data.copy(), header, variance, abeams, bbeams)
        capt = capsys.readouterr()
        assert 'All good frames' in capt.out
        assert len(g) == 4
        assert np.allclose(d[m], data[m])
        assert not np.allclose(d[~m], data[~m])
        assert (~m).sum() == 160

        # add too many spikes
        rand = np.random.RandomState(42)
        idx = rand.random_sample(data.size // 2) * data.size
        data.flat[idx.astype(int)] = 10000
        d, m, g = dsp.despike(data.copy(), header, variance, abeams, bbeams)
        capt = capsys.readouterr()
        assert capt.err.count('Too many spikes') == 4
        # Two frames are kept so that not all are trashed
        assert len(g) == 2
        assert np.allclose(d[m], data[m])
        assert not np.allclose(d[~m], data[~m])
        assert (~m).sum() == 2001350
