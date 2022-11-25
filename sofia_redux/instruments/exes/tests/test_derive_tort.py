# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
from astropy.io import fits

from sofia_redux.instruments.exes import derive_tort as dt
from sofia_redux.instruments.exes.readhdr import readhdr
from sofia_redux.toolkit.utilities.fits import set_log_level


class TestDeriveTort(object):

    def test_derive_tort_single_order(self, capsys, rdc_low_flat_hdul):
        hdul = rdc_low_flat_hdul
        data = hdul[0].data[0]
        header = hdul[0].header

        td, ti = dt.derive_tort(data, header)

        # should find one central illuminated region and
        # record its edges in the header
        illum = (ti == 1)
        assert np.mean(td[illum]) > np.mean(td[~illum])
        idx = np.argwhere(illum)
        # y
        assert np.min(idx[:, 0]) == 324
        assert np.max(idx[:, 0]) == 673
        assert header['ORDR_B'] == '324'
        assert header['ORDR_T'] == '673'
        # x - has 2 pixel buffer
        assert np.min(idx[:, 1]) == 0
        assert np.max(idx[:, 1]) == 1023
        assert header['ORDR_S'] == '2'
        assert header['ORDR_E'] == '1021'

        # specify explicit start/end values
        td2, ti2 = dt.derive_tort(data, header, top_pixel=500,
                                  bottom_pixel=350,
                                  start_pixel=50, end_pixel=1000)
        # order edges are recorded
        assert header['ORDR_B'] == '350'
        assert header['ORDR_T'] == '500'
        assert header['ORDR_S'] == '50'
        assert header['ORDR_E'] == '1000'
        # data is the same
        assert np.allclose(td2, td)
        assert np.allclose(ti2, ti)

    @pytest.mark.parametrize('method', ['deriv', 'sqderiv', 'sobel'])
    def test_derive_tort_high_low(self, capsys, rdc_high_low_flat_hdul,
                                  method):
        hdul = rdc_high_low_flat_hdul
        data = hdul[0].data[0]
        header = hdul[0].header

        td, ti = dt.derive_tort(data, header, edge_method=method)

        # uses FFT on the edge image + iteration to find krot
        assert 'KROT Iteration 1' in capsys.readouterr().out
        assert np.allclose(header['KROT'], 0.074, atol=.001)

        # for any method, should find 22 illuminated orders and
        # record their edges in the header
        illum = (ti == 1)
        assert np.mean(td[illum]) > np.mean(td[~illum])
        assert header['NORDERS'] == 22
        assert header['ORDERS'] == ','.join([str(f) for f in range(22, 0, -1)])

        # top
        expected = [955, 909, 864, 819, 774, 729, 683, 638, 593, 548, 502, 457,
                    412, 367, 322, 276, 231, 186, 141, 95, 50, 5]
        found = [int(f) for f in header['ORDR_B'].split(',')]
        assert np.allclose(found, expected, atol=2)

        # bottom
        expected = [990, 944, 899, 854, 809, 764, 718, 673, 628, 583, 537,
                    492, 447, 402, 357, 311, 266, 221, 176, 130, 85, 40]
        found = [int(f) for f in header['ORDR_T'].split(',')]
        assert np.allclose(found, expected, atol=1)

        # start
        expected = [46, 42, 38, 34, 30, 26, 22, 17, 13, 9, 5,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        found = [int(f) for f in header['ORDR_S'].split(',')]
        assert np.allclose(found, expected, atol=1)

        # end
        expected = [1021, 1021, 1021, 1021, 1021, 1021, 1021, 1021, 1021,
                    1021, 1015, 1010, 1005, 1000, 995, 990, 985, 980, 975,
                    971, 966, 961]
        found = [int(f) for f in header['ORDR_E'].split(',')]
        assert np.allclose(found, expected, atol=1)

    @pytest.mark.parametrize('method', ['deriv', 'sqderiv', 'sobel'])
    def test_derive_tort_high_med(self, capsys, rdc_high_med_flat_hdul,
                                  method):
        hdul = rdc_high_med_flat_hdul
        data = hdul[0].data[0]
        header = hdul[0].header

        td, ti = dt.derive_tort(data, header, edge_method=method)

        # uses FFT on the edge image + iteration to find krot
        assert 'KROT Iteration 1' in capsys.readouterr().out
        assert np.allclose(header['KROT'], -0.126, atol=.002)

        # for any method, should find 9 illuminated orders and
        # record their edges in the header
        illum = (ti == 1)
        assert np.mean(td[illum]) > np.mean(td[~illum])
        assert header['NORDERS'] == 9
        assert header['ORDERS'] == ','.join([str(f) for f in range(9, 0, -1)])

        # top
        expected = [921, 818, 715, 612, 509, 406, 303, 199, 97]
        found = [int(f) for f in header['ORDR_B'].split(',')]
        assert np.allclose(found, expected, atol=3)

        # bottom
        expected = [1014, 911, 808, 705, 602, 499, 396, 293, 190]
        found = [int(f) for f in header['ORDR_T'].split(',')]
        assert np.allclose(found, expected, atol=2)

        # start
        expected = [297, 42, 32, 23, 14, 5, 2, 2, 2]
        found = [int(f) for f in header['ORDR_S'].split(',')]
        assert np.allclose(found[1:], expected[1:], atol=2)

        # end
        expected = [1021, 1021, 1021, 1021, 1016, 1004, 993, 981, 970]
        found = [int(f) for f in header['ORDR_E'].split(',')]
        assert np.allclose(found, expected, atol=2)

    @pytest.mark.parametrize('explicit_path', [True, False])
    def test_custom_wavemap(self, tmpdir, capsys, rdc_high_low_flat_hdul,
                            explicit_path):
        hdul = rdc_high_low_flat_hdul
        data = hdul[0].data[0]
        header = hdul[0].header

        # raises error for bad file
        with pytest.raises(ValueError) as err:
            dt.derive_tort(data, header, custom_wavemap='badfile.dat')
        assert 'Could not apply modification' in str(err)

        wvm = tmpdir.join('customWVM.dat')
        wvm.write('954 989 40 1020\n'
                  '908 943 40 1020\n'
                  '863 898 40 1020\n')

        # should accept either an explicit path to the file or else
        # the file 'customWVM.dat' in the current directory
        if explicit_path:
            dt.derive_tort(data, header, custom_wavemap=str(wvm))
        else:
            with tmpdir.as_cwd():
                dt.derive_tort(data, header, custom_wavemap=True)

        # either way, should explicitly set the orders as specified
        assert header['NORDERS'] == 3
        assert header['ORDERS'] == ','.join([str(f) for f in range(3, 0, -1)])

        # top
        expected = [954, 908, 863]
        found = [int(f) for f in header['ORDR_B'].split(',')]
        assert np.allclose(found, expected)

        # bottom
        expected = [989, 943, 898]
        found = [int(f) for f in header['ORDR_T'].split(',')]
        assert np.allclose(found, expected)

        # start
        expected = [40, 40, 40]
        found = [int(f) for f in header['ORDR_S'].split(',')]
        assert np.allclose(found, expected)

        # end
        expected = [1020, 1020, 1020]
        found = [int(f) for f in header['ORDR_E'].split(',')]
        assert np.allclose(found, expected)

    def test_start_end_xd(self, tmpdir, capsys, rdc_high_low_flat_hdul):
        hdul = rdc_high_low_flat_hdul
        data = hdul[0].data[0]
        header = hdul[0].header

        dt.derive_tort(data, header, start_pixel=50, end_pixel=1000)

        # should explicitly set start and end as specified
        assert header['NORDERS'] == 22

        # start
        expected = [50] * 22
        found = [int(f) for f in header['ORDR_S'].split(',')]
        assert np.allclose(found, expected)

        # end
        expected = [1000] * 22
        found = [int(f) for f in header['ORDR_E'].split(',')]
        assert np.allclose(found, expected)

    def test_describe_orders_errors(self, capsys, mocker):
        ny, nx = 20, 30
        tortdata = np.arange(ny * nx).reshape((ny, nx))
        header = readhdr(fits.Header({'NSPAT': nx, 'NSPEC': ny,
                                      'WAVENO0': 1210.0}),
                         check_header=False)
        illum = np.ones((ny, nx), dtype=int)

        with set_log_level('DEBUG'):
            spacing, angle = dt._describe_orders(tortdata, header, illum)

        # in flat data, can't find max in fft
        capt = capsys.readouterr()
        assert "Can't find harmonic max" in capt.err
        assert "Error encountered during FFT peak location" in capt.out
        assert np.allclose(spacing, 1.02283)
        assert np.allclose(angle, 0)

        # predicted spacing doesn't match derived
        header['SPACING'] = 10.0
        spacing, angle = dt._describe_orders(tortdata, header, illum)
        capt = capsys.readouterr()
        assert "Order spacing disagrees with prediction" in capt.err
        assert np.allclose(spacing, 8.57142)
        assert np.allclose(angle, 0)

        # zero max power
        tortdata *= 0
        with pytest.raises(ValueError) as err:
            dt._describe_orders(tortdata, header, illum)
        assert "Can't determine order spacing" in str(err)

    def test_get_xd_power_errors(self, capsys):
        ny, nx = 40, 30
        tortdata = np.arange(ny * nx, dtype=float).reshape((ny, nx))
        header = readhdr(fits.Header({'NSPAT': nx, 'NSPEC': ny,
                                      'WAVENO0': 1210.0, 'SPACING': 8.0}),
                         check_header=False)

        power, illumx = dt._get_xd_power(tortdata, header)
        assert power.shape == (9,)
        assert np.allclose(power, np.arange(596, 605, 1))
        assert illumx.shape == (nx,)
        assert np.sum(illumx[illumx == 1]) == 13
        assert header['NORDERS'] == 3
        assert header['XORDER1'] == 3.0

        power, illumx = dt._get_xd_power(-tortdata, header)
        assert power.shape == (9,)
        assert np.allclose(power, -np.arange(596, 605, 1))
        assert illumx.shape == (nx,)
        assert np.sum(illumx[illumx == 1]) == 15
        assert header['NORDERS'] == 3
        assert header['XORDER1'] == 3.0

        with pytest.raises(ValueError) as err:
            dt._get_xd_power(tortdata * np.nan, header)
        assert "No illuminated pixels" in str(err)

    def test_process_cross_dispersed_errors(self, capsys, mocker):
        ny, nx = 40, 30
        data = np.arange(ny * nx, dtype=float).reshape((ny, nx))
        header = readhdr(fits.Header({'NSPAT': nx, 'NSPEC': ny,
                                      'WAVENO0': 1210.0}),
                         check_header=False)

        with pytest.raises(ValueError) as err:
            dt._process_cross_dispersed(data, header, maxiter=2)
        assert 'Illuminated power could not be found' in str(err)
        assert '2 iterations used' in capsys.readouterr().err

        with pytest.raises(ValueError):
            dt._process_cross_dispersed(data, header, maxiter=2, fixed=True)
        capt = capsys.readouterr()
        assert 'Illuminated power could not be found' in str(err)
        assert 'Angle > 0.001' in capt.err
        assert '2 iterations used' not in capt.err

        mocker.patch.object(dt, '_get_xd_power',
                            return_value=(np.ones(ny), np.ones(nx)))
        mocker.patch.object(dt, '_get_top_bottom_pixels',
                            return_value=(None, None, None, None))
        with pytest.raises(ValueError) as err:
            dt._process_cross_dispersed(data, header, maxiter=2, fixed=True)
        assert 'Order edges could not be found' in str(err)

    def test_get_top_bottom_pixels_errors(self, capsys):
        ny, nx = 40, 30
        header = readhdr(fits.Header({'NSPAT': nx, 'NSPEC': ny,
                                      'WAVENO0': 1210.0,
                                      'INSTCFG': 'HIGH_MED'}),
                         check_header=False)
        illum_y = np.ones(ny)

        # flat illum: no orders found
        illum_x = np.ones(nx)
        b1, t1, ss1, ee1 = dt._get_top_bottom_pixels(header, illum_x, illum_y)
        assert b1 is None
        assert t1 is None
        assert ss1 is None
        assert ee1 is None
        assert header['NORDERS'] == 0
        assert 'Not all orders were found' in capsys.readouterr().err

        # some orders indicated
        illum_x[::3] = 0
        b1, t1, ss1, ee1 = dt._get_top_bottom_pixels(header, illum_x, illum_y)
        assert b1.size == 9
        assert t1.size == 9
        assert header['NORDERS'] == 9
        assert capsys.readouterr().err == ''
        illum_x[:] = 1

        # mismatched order
        illum_x[-3:] = 0
        b1, t1, ss1, ee1 = dt._get_top_bottom_pixels(header, illum_x, illum_y)
        assert b1 is None
        assert t1 is None
        assert header['NORDERS'] == 0
        assert 'Not all orders were found' in capsys.readouterr().err

    def test_get_left_right_pixels_errors(self, capsys):
        ny, nx = 40, 30
        header = readhdr(fits.Header({'NSPAT': nx, 'NSPEC': ny,
                                      'NORDERS': 1,
                                      'INSTCFG': 'LOW'}),
                         check_header=False)
        tortillum = np.ones((ny, nx))

        # flat illum: orders set to start/end
        s1, e1 = dt._get_left_right_pixels(header, tortillum, [0], [ny - 1])
        assert s1 == [2]
        assert e1 == [nx - 3]

        # same for empty/bad illum
        tortillum[:] = -1
        s1, e1 = dt._get_left_right_pixels(header, tortillum, [0], [ny - 1])
        assert s1 == [2]
        assert e1 == [nx - 3]
