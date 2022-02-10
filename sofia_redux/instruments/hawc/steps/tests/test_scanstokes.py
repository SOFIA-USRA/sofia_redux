# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepscanmappol import StepScanMapPol
from sofia_redux.instruments.hawc.steps.stepscanstokes import StepScanStokes
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, scan_raw_data, scanpol_crh_data


class TestScanStokes(DRPTestCase):
    def make_data(self, tmpdir, angle=None):
        # make scan pol data
        if angle is None:
            angle = [5.0, 50.0, 27.0, 72.0]
        inp = []
        for i in range(4):
            hdul = scan_raw_data()
            hdul[2].data['hwpCounts'] = angle[i] * 4
            ffile = str(tmpdir.join('test{}.fits'.format(i)))
            hdul.writeto(ffile, overwrite=True)
            inp.append(DataFits(ffile))
        return inp

    def test_siso(self, tmpdir):
        inp = self.make_data(tmpdir)

        # move to tmpdir -- writes temp files
        with tmpdir.as_cwd():
            # raw data fails
            step = StepScanStokes()
            with pytest.raises(ValueError):
                step(inp[0])

            # scanmapped data works
            scanmapped = StepScanMapPol()(inp)
            out = step(scanmapped[0])
            assert isinstance(out, DataFits)

            assert 'STOKES I' in out.imgnames
            assert 'STOKES Q' in out.imgnames
            assert 'STOKES U' in out.imgnames

    def test_read_radius(self, capsys):
        df = DataFits()
        df.setheadval('SPECTEL1', 'HAWE')
        step = StepScanStokes()
        step.datain = df
        step.runstart(df, {})

        expected = step.getarg('zero_level_radius')[-1]

        # test defaults
        result = step.read_radius()
        assert result == expected

        # bad spectel
        df.setheadval('SPECTEL1', 'HAWQ')
        with pytest.raises(ValueError):
            step.read_radius()
        capt = capsys.readouterr()
        assert 'Cannot parse waveband' in capt.err

        df.setheadval('SPECTEL1', '')
        with pytest.raises(ValueError):
            step.read_radius()
        capt = capsys.readouterr()
        assert 'Cannot parse waveband' in capt.err

        # bad arglist
        df.setheadval('SPECTEL1', 'HAWE')
        step.runstart(df, {'zero_level_radius': [1, 2, 3]})
        with pytest.raises(IndexError):
            step.read_radius()
        capt = capsys.readouterr()
        assert 'Missing radius values' in capt.err

    def test_nwhwp(self, capsys):
        df = DataFits()
        df.setheadval('NHWP', 3)
        step = StepScanStokes()

        # fails if nhwp!=4
        with pytest.raises(ValueError):
            step(df)
        capt = capsys.readouterr()
        assert 'Unexpected number of HWP angles' in capt.err

    def test_angles(self, tmpdir, capsys):
        inp = self.make_data(tmpdir, angle=[5.0, 60.0, 27.0, 80.0])
        with tmpdir.as_cwd():
            df = StepScanMapPol()(inp)[0]
            step = StepScanStokes()

            # warns for bad angles
            out = step(df, hwp_tol=5.0)
            capt = capsys.readouterr()
            assert 'Stokes Q: HWP angles differ by ' \
                   '55.0 degrees (should be 45.0)' in capt.err
            assert 'Stokes U: HWP angles differ by ' \
                   '53.0 degrees (should be 45.0)' in capt.err

            # but still completes processing
            assert 'STOKES I' in out.imgnames
            assert 'STOKES Q' in out.imgnames
            assert 'STOKES U' in out.imgnames

    def test_hwpstart(self, tmpdir, capsys):
        hdul = scanpol_crh_data()
        del hdul[0].header['HWPSTART']
        hwpinit = hdul[0].header['HWPINIT']
        assert hwpinit != -9999

        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        hdul.close()
        df = DataFits(ffile)
        step = StepScanStokes()

        # adds HWPINIT as HWPSTART if missing
        out = step(df)
        assert out.getheadval('HWPSTART') == hwpinit

        # same for -9999
        df.setheadval('HWPSTART', -9999)
        out = step(df)
        assert out.getheadval('HWPSTART') == hwpinit

    def test_shift(self, tmpdir, capsys, mocker):
        inp = self.make_data(tmpdir, angle=[5.0, 60.0, 27.0, 80.0])
        with tmpdir.as_cwd():
            df = StepScanMapPol()(inp)[0]
            step = StepScanStokes()

            # test that shifts are performed for T to R registration

            # zero shift
            mocker.patch.object(step, 'wcs_shift', return_value=[0.005, 0.005])
            step(df)
            capt = capsys.readouterr()
            assert 'Shifting R image' not in capt.out
            assert 'Shifting T image' not in capt.out

            # other shift
            mocker.patch.object(step, 'wcs_shift', return_value=[0.1, 0.1])
            step(df)
            capt = capsys.readouterr()
            assert 'Shifting R image 1 by x,y=0.1,0.1' in capt.out
            assert 'Shifting T image 1 by x,y=0.1,0.1' in capt.out

    def test_zero_level_auto(self, tmpdir, capsys):
        hdul = scanpol_crh_data()

        # Add a negative patch to R and T in all HWP
        for i in range(0, len(hdul), 6):
            hdul[i].data[5:10, 5:10] = -100.0 + i
            hdul[i + 1].data[5:10, 5:10] = -200.0 + i
        zl_rad = [15] * 5

        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        hdul.close()
        df = DataFits(ffile)

        step = StepScanStokes()
        result1 = step(df, zero_level_method='none', zero_level_radius=zl_rad,
                       zero_level_region='auto')
        assert 'No zero level correction attempted' in capsys.readouterr().out
        assert np.nanmin(result1.data) < 0

        # auto method mean
        result1 = step(df, zero_level_method='mean', zero_level_radius=zl_rad,
                       zero_level_region='auto')
        capt = capsys.readouterr()
        assert 'Correcting zero level' in capt.out
        for i in range(4):
            assert f'R level: {-100 + i * 6}' in capt.out
            assert f'T level: {-200 + i * 6}' in capt.out
        assert np.nanmin(result1.data) >= 0

        # auto method median
        result2 = step(df, zero_level_method='median',
                       zero_level_radius=zl_rad,
                       zero_level_region='auto')
        capt = capsys.readouterr()
        assert 'Correcting zero level' in capt.out
        for i in range(4):
            assert f'R level: {-100 + i * 6}' in capt.out
            assert f'T level: {-200 + i * 6}' in capt.out
        assert np.allclose(result2.data, result1.data, equal_nan=True)

        # no negative patch found: no correction attempted
        for i in range(0, len(hdul), 6):
            hdul[i].data[5:10, 5:10] = 100.0 + i
            hdul[i + 1].data[5:10, 5:10] = 200.0 + i
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        hdul.close()
        df = DataFits(ffile)
        result3 = step(df, zero_level_method='mean', zero_level_radius=zl_rad,
                       zero_level_region='auto')
        capt = capsys.readouterr()
        assert 'No negative zero level' in capt.out
        assert 'R level' not in capt.out
        assert 'T level' not in capt.out
        assert np.nanmin(result3.data) >= 0

    def test_zero_level_specified(self, tmpdir, capsys):
        hdul = scanpol_crh_data()

        # Add a negative patch to R and T in all HWP
        for i in range(0, len(hdul), 6):
            hdul[i].data[5:10, 5:10] = -100.0 + i
            hdul[i + 1].data[5:10, 5:10] = -200.0 + i

        # region in the middle of this patch in RA, Dec, radius
        region = [341.1646137, -8.8991308, 0.0041764]
        zl_rad = [15] * 5

        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        hdul.close()
        df = DataFits(ffile)
        step = StepScanStokes()

        # no correction
        result0 = step(df, zero_level_method='none',
                       zero_level_region=region)
        assert 'No zero level correction attempted' in capsys.readouterr().out
        assert np.nanmin(result0.data) < 0

        # specified method with mean
        result1 = step(df, zero_level_method='mean',
                       zero_level_region=region, zero_level_sigma=0)
        capt = capsys.readouterr()
        assert 'Correcting zero level' in capt.out
        for i in range(4):
            assert f'R level: {-100 + i * 6}' in capt.out
            assert f'T level: {-200 + i * 6}' in capt.out
        assert np.nanmin(result1.data) >= 0

        # specified method with median
        result2 = step(df, zero_level_method='median',
                       zero_level_region=region, zero_level_sigma=0)
        capt = capsys.readouterr()
        assert 'Correcting zero level' in capt.out
        for i in range(4):
            assert f'R level: {-100 + i * 6}' in capt.out
            assert f'T level: {-200 + i * 6}' in capt.out
        assert np.allclose(result2.data, result1.data, equal_nan=True)

        # attempt to use header region without adding keys
        result3 = step(df, zero_level_method='mean', zero_level_radius=zl_rad,
                       zero_level_region='header')
        capt = capsys.readouterr()
        assert 'Falling back to auto' in capt.err
        assert 'Correcting zero level' in capt.out
        for i in range(4):
            assert f'R level: {-100 + i * 6}' in capt.out
            assert f'T level: {-200 + i * 6}' in capt.out
        assert np.allclose(result3.data, result1.data, equal_nan=True)

        # attempt to specify an invalid region with special value -9999
        df.header['ZERO_RA'] = -9999.0
        df.header['ZERO_DEC'] = -9999.0
        df.header['ZERO_RAD'] = -9999.0
        result3a = step(df, zero_level_method='mean', zero_level_radius=zl_rad,
                        zero_level_region='header')
        capt = capsys.readouterr()
        assert 'Falling back to auto' in capt.err
        assert 'Correcting zero level' in capt.out
        assert 'nan' not in capt.err
        assert np.allclose(result3a.data, result1.data, equal_nan=True)

        # attempt to specify a completely invalid region
        # (same effect, but different error message)
        df.header['ZERO_RA'] = -9998
        df.header['ZERO_DEC'] = -9998
        df.header['ZERO_RAD'] = -9998
        result3b = step(df, zero_level_method='mean', zero_level_radius=zl_rad,
                        zero_level_region='header')
        capt = capsys.readouterr()
        assert 'falling back to auto' in capt.err
        assert 'Correcting zero level' in capt.out
        assert 'nan' in capt.err
        assert np.allclose(result3b.data, result1.data, equal_nan=True)

        # now specify correct region in header instead of arguments
        df.header['ZERO_RA'] = region[0]
        df.header['ZERO_DEC'] = region[1]
        df.header['ZERO_RAD'] = region[2]
        result4 = step(df, zero_level_method='mean',
                       zero_level_region='header',
                       zero_level_sigma=3.0)
        capt = capsys.readouterr()
        assert 'falling back' not in capt.err.lower()
        assert 'Correcting zero level' in capt.out
        for i in range(4):
            assert f'R level: {-100 + i * 6}' in capt.out
            assert f'T level: {-200 + i * 6}' in capt.out
        assert np.allclose(result4.data, result1.data, equal_nan=True)

        # no negative patch: correction still attempted without
        # checking value when region is directly specified
        for i in range(0, len(hdul), 6):
            hdul[i].data[5:10, 5:10] = 100.0 + i
            hdul[i + 1].data[5:10, 5:10] = 200.0 + i
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        hdul.close()
        df = DataFits(ffile)
        result5 = step(df, zero_level_method='median',
                       zero_level_region=region,
                       zero_level_sigma=3.0)
        capt = capsys.readouterr()
        assert 'Correcting zero level' in capt.out
        for i in range(4):
            assert f'R level: {100 + i * 6}' in capt.out
            assert f'T level: {200 + i * 6}' in capt.out
        assert np.nanmin(result5.data) < 0

        # bad patch location: falls back to auto method
        region[0] = 0
        result6 = step(df, zero_level_method='mean', zero_level_radius=zl_rad,
                       zero_level_region=region)
        capt = capsys.readouterr()
        assert 'not on array' in capt.err
        assert 'falling back to auto' in capt.err
        assert 'No negative zero level' in capt.out
        assert np.nanmin(result6.data) >= 0

        # bad patch specification: raises error
        region[0] = 'bad'
        with pytest.raises(ValueError) as err:
            step(df, zero_level_method='mean',
                 zero_level_region=str(region))
        assert 'Badly formatted' in str(err)
