# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

from astropy.io import fits
import numpy as np

from sofia_redux.instruments.exes import make_template as mt


class TestMakeTemplate(object):

    def test_make_template_single(self, nsb_single_order_hdul):
        data = nsb_single_order_hdul[0].data
        header = nsb_single_order_hdul[0].header
        weighting_frame = nsb_single_order_hdul[1].data ** 2
        nz, ny, nx = data.shape

        template = mt.make_template(data, header, weighting_frame)

        # template matches data frame shape
        assert template.shape == (ny, nx)

        # replication is along spectral direction, so x axis for single order
        avg_template = template[:, 0]
        assert np.allclose(template.T, avg_template)

        # data is nodded, so should add to roughly zero,
        # with a positive and negative peak
        assert np.allclose(avg_template.sum(), 0, atol=1)
        maxi, mini = np.argmax(avg_template), np.argmin(avg_template)
        assert np.allclose(avg_template[maxi], -avg_template[mini],
                           rtol=0.01)
        assert maxi == 472
        assert mini == 552

        # if collapsed, just returns profile rather than image
        collapsed = mt.make_template(data, header, weighting_frame,
                                     collapsed=True)
        assert collapsed.shape == (ny,)
        assert np.allclose(collapsed, avg_template)

    def test_make_template_multi(self, und_cross_dispersed_hdul):
        data = und_cross_dispersed_hdul[0].data
        header = und_cross_dispersed_hdul[0].header
        weighting_frame = und_cross_dispersed_hdul[1].data ** 2
        nz, ny, nx = data.shape

        template = mt.make_template(data, header, weighting_frame)

        # template matches data frame shape
        assert template.shape == (ny, nx)

        # replication is along spectral direction, so x axis
        avg_template = template[:, 0]
        assert np.allclose(template.T, avg_template)

        # data is nodded off slit in 22 orders,
        # so should sum to positive value with 22 peaks
        # of approximately equal area
        one_peak = np.sum(avg_template[:86])
        assert np.allclose(avg_template.sum(), 22 * one_peak, rtol=0.3)

        # if collapsed, just returns profile rather than image
        collapsed = mt.make_template(data, header, weighting_frame,
                                     collapsed=True)
        assert collapsed.shape == (ny,)
        assert np.allclose(collapsed, avg_template)

    def test_input_errors(self, capsys, mocker):
        nz, ny, nx = 4, 5, 6
        data = np.ones((nz, ny, nx), dtype=float)
        header = fits.Header()
        weight = np.ones((ny, nx), dtype=float)

        # dimensions unspecified
        template = mt.make_template(data, header, weight)
        assert template is None
        assert 'Data has wrong dimensions' in capsys.readouterr().err

        # okay if nspat/spec in header
        header['NSPAT'] = nx
        header['NSPEC'] = ny
        template = mt.make_template(data, header, weight)
        assert template.shape == (ny, nx)
        assert np.all(template == 1)

        # okay if only 1 frame
        template = mt.make_template(data[0], header, weight)
        assert template.shape == (ny, nx)
        assert np.all(template == 1)

        # mock error in good data points
        mocker.patch.object(mt, '_good_data_points',
                            side_effect=RuntimeError('test1'))
        template = mt.make_template(data, header, weight)
        assert template is None
        assert 'test1' in capsys.readouterr().err

        # mock error in weight good frames
        mocker.patch.object(mt, '_weight_good_frames',
                            side_effect=RuntimeError('test2'))
        template = mt.make_template(data, header, weight)
        assert template is None
        assert 'test2' in capsys.readouterr().err

    def test_weight_good_frames(self):
        nz = 6
        good_frames = [1, 2, 3]
        weights = None

        # good frames equally weighted
        zsum = mt._weight_good_frames(good_frames, weights, nz)
        assert zsum == len(good_frames)
        zsum = mt._weight_good_frames(None, weights, nz)
        assert zsum == nz
        zsum = mt._weight_good_frames([], weights, nz)
        assert zsum == nz

        # no good frames
        with pytest.raises(RuntimeError) as err:
            mt._weight_good_frames([21, 22, 23], weights, nz)
        assert 'No good frames' in str(err)

        # weights provided badly: uniform weights
        weights = np.arange(10)
        zsum = mt._weight_good_frames(good_frames, weights, nz)
        assert zsum == len(good_frames)

        # weights match nz: set to zero for bad frames
        weights = np.arange(nz)
        zsum = mt._weight_good_frames(good_frames, weights, nz)
        assert zsum == np.sum(weights[good_frames])

    def test_good_data_points(self):
        nz, ny, nx = 4, 5, 6
        illum = np.ones((ny, nx))
        weight_frame = np.ones((nz, ny, nx))
        shape = (ny, nx)
        weights = np.ones(nz)

        # all good
        zg, mask = mt._good_data_points(illum, weight_frame, shape, weights)
        assert zg.shape == shape
        assert np.all(zg)
        assert np.sum(zg) == ny * nx
        assert mask.shape == (nz, ny, nx)
        assert np.all(mask == 1)
        assert np.sum(mask) == nz * ny * nx

        # some bad
        illum[0, :] = 0
        weight_frame[1, 2, 3] = 0
        zg, mask = mt._good_data_points(illum, weight_frame, shape, weights)
        assert zg.shape == shape
        assert np.sum(zg) == (ny - 1) * nx - 1
        assert mask.shape == (nz, ny, nx)
        assert np.sum(mask) == nz * (ny - 1) * nx - 1

        # all bad
        weight_frame *= 0
        with pytest.raises(RuntimeError) as err:
            mt._good_data_points(illum, weight_frame, shape, weights)
        assert 'No good data points' in str(err)

    def test_create_template_image(self, capsys):
        ny, nx = 5, 6
        z_good = np.ones((ny, nx))
        z_avg = np.arange(ny * nx, dtype=float).reshape((ny, nx))
        z_weight_frame = np.ones((ny, nx))

        # collapsed
        template_col = mt._create_template_image(z_good, z_avg, nx, ny,
                                                 z_weight_frame, True)
        assert template_col.shape == (ny,)
        assert np.allclose(template_col, np.sum(z_avg, axis=1) / nx)

        # full image
        template = mt._create_template_image(z_good, z_avg, nx, ny,
                                             z_weight_frame, False)
        assert template.shape == (ny, nx)
        assert np.allclose(template.T, template_col)

        # some bad data: average is the same
        z_good[1, 2] = 0
        z_good[3, 4] = 0
        template_col = mt._create_template_image(z_good, z_avg, nx, ny,
                                                 z_weight_frame, True)
        assert template_col.shape == (ny,)
        assert np.allclose(template_col, np.sum(z_avg, axis=1) / nx)

        # no good data
        z_good *= 0
        template = mt._create_template_image(z_good, z_avg, nx, ny,
                                             z_weight_frame, False)
        assert template is None
        assert 'No good data' in capsys.readouterr().err
