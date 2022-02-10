# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepzerolevel import StepZeroLevel
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, scan_smp_data


class TestZeroLevel(DRPTestCase):
    def make_data(self, tmpdir):
        hdul = scan_smp_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        return DataFits(ffile)

    def test_siso(self, tmpdir):
        df = self.make_data(tmpdir)
        step = StepZeroLevel()
        out = step(df)

        # check expected output format
        assert isinstance(out, DataFits)
        assert 'PRIMARY IMAGE' in out.imgnames
        assert 'NOISE' in out.imgnames
        assert 'EXPOSURE' in out.imgnames
        assert 'S/N' in out.imgnames

    def test_read_radius(self, capsys):
        df = DataFits()
        df.setheadval('SPECTEL1', 'HAWE')
        step = StepZeroLevel()
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

    def test_zero_level_auto(self, tmpdir, capsys):
        hdul = scan_smp_data()

        # Add a negative patch to the image
        hdul[0].data[5:10, 5:10] = -100.0
        zl_rad = [15] * 5

        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        hdul.close()
        df = DataFits(ffile)

        step = StepZeroLevel()
        result1 = step(df, zero_level_method='none', zero_level_radius=zl_rad,
                       zero_level_region='auto')
        assert 'no zero level correction attempted' in capsys.readouterr().out
        assert np.nanmin(result1.data) < 0

        # auto method mean
        result1 = step(df, zero_level_method='mean', zero_level_radius=zl_rad,
                       zero_level_region='auto')
        capt = capsys.readouterr()
        assert 'Correcting zero level' in capt.out
        assert 'Zero level: -100' in capt.out
        assert np.nanmin(result1.data) >= 0

        # auto method median
        result2 = step(df, zero_level_method='median',
                       zero_level_radius=zl_rad,
                       zero_level_region='auto')
        capt = capsys.readouterr()
        assert 'Correcting zero level' in capt.out
        assert 'Zero level: -100' in capt.out
        assert np.allclose(result2.data, result1.data, equal_nan=True)

        # no negative patch found: no correction attempted
        hdul[0].data[5:10, 5:10] = 100.0
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        hdul.close()
        df = DataFits(ffile)
        result3 = step(df, zero_level_method='mean', zero_level_radius=zl_rad,
                       zero_level_region='auto')
        capt = capsys.readouterr()
        assert 'No negative zero level' in capt.out
        assert 'Zero level' not in capt.out
        assert np.nanmin(result3.data) >= 0

    def test_zero_level_specified(self, tmpdir, capsys):
        hdul = scan_smp_data()

        # Add a negative patch
        hdul[0].data[5:10, 5:10] = -100.0

        # region in the middle of this patch in RA, Dec, radius
        region = [341.1646137, -8.8991308, 0.0041764]
        zl_rad = [15] * 5

        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        hdul.close()
        df = DataFits(ffile)
        step = StepZeroLevel()

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
        assert 'Zero level: -100' in capt.out
        assert np.nanmin(result1.data) >= 0

        # specified method with median
        result2 = step(df, zero_level_method='median',
                       zero_level_region=region, zero_level_sigma=0)
        capt = capsys.readouterr()
        assert 'Correcting zero level' in capt.out
        assert 'Zero level: -100' in capt.out
        assert np.allclose(result2.data, result1.data, equal_nan=True)

        # attempt to use header region without adding keys
        result3 = step(df, zero_level_method='mean', zero_level_radius=zl_rad,
                       zero_level_region='header')
        capt = capsys.readouterr()
        assert 'Falling back to auto' in capt.err
        assert 'Correcting zero level' in capt.out
        assert 'Zero level: -100' in capt.out
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
        assert 'Zero level: -100' in capt.out
        assert np.allclose(result4.data, result1.data, equal_nan=True)

        # no negative patch: correction still attempted without
        # checking value when region is directly specified
        hdul[0].data[5:10, 5:10] = 100.0
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        hdul.close()
        df = DataFits(ffile)
        result5 = step(df, zero_level_method='median',
                       zero_level_region=region,
                       zero_level_sigma=3.0)
        capt = capsys.readouterr()
        assert 'Correcting zero level' in capt.out
        assert 'Zero level: 100' in capt.out
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
