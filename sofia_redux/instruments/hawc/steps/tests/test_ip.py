# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepip import StepIP
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_bgs_data


class TestIP(DRPTestCase):
    def test_siso(self, tmpdir):
        hdul = pol_bgs_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)

        step = StepIP()
        out = step(inp)

        assert isinstance(out, DataFits)

    def test_read_ip(self, capsys):
        df = DataFits()
        df.setheadval('SPECTEL1', 'HAWE')
        step = StepIP()
        step.datain = df
        step.runstart(df, {})

        expected = (step.getarg('qinst')[-1],
                    step.getarg('uinst')[-1])

        # test defaults
        result = step.read_ip()
        assert result == expected

        # bad spectel
        df.setheadval('SPECTEL1', 'HAWQ')
        with pytest.raises(ValueError):
            step.read_ip()
        capt = capsys.readouterr()
        assert 'Cannot parse waveband' in capt.err

        df.setheadval('SPECTEL1', '')
        with pytest.raises(ValueError):
            step.read_ip()
        capt = capsys.readouterr()
        assert 'Cannot parse waveband' in capt.err

        # bad arglist
        df.setheadval('SPECTEL1', 'HAWE')
        step.runstart(df, {'qinst': [1, 2, 3]})
        with pytest.raises(IndexError):
            step.read_ip()
        capt = capsys.readouterr()
        assert 'Missing IP values' in capt.err

        # wrong values
        df.setheadval('SPECTEL1', 'HAWE')
        step.runstart(df, {'qinst': [1, 2, 3, 4, 5]})
        with pytest.raises(ValueError):
            step.read_ip()
        capt = capsys.readouterr()
        assert 'Absolute value of IP parameters must be <= 1' in capt.err

    def test_read_file_ip(self, tmpdir, capsys):
        df = DataFits()
        df.setheadval('SPECTEL1', 'HAWE')
        step = StepIP()
        step.datain = df
        step.runstart(df, {})

        # test that a bad file raises an error
        fake_ip = DataFits()
        fake_ip.imageset(np.zeros(10))
        fname = str(tmpdir.join('badfile.fits'))
        fake_ip.save(fname)
        with pytest.raises(ValueError):
            step.read_file_ip(fname)
        capt = capsys.readouterr()
        assert 'Problem with band E HDU in fileip' in capt.err

    def test_ip_options(self, tmpdir, capsys):
        hdul = pol_bgs_data(empty=True)
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)

        step = StepIP()

        # try to run on nhwp=1 (image data)
        df2 = df.copy()
        df2.setheadval('NHWP', 1)
        out = step(df2)
        capt = capsys.readouterr()
        assert 'Only 1 HWP, so skipping step' in capt.out
        assert np.allclose(out.image, df.image, equal_nan=True)

        # call with and without fileip = uniform -- default should
        # be close, on average, for flat image
        out1 = step(df.copy())
        out2 = step(df.copy(), fileip='uniform')
        assert np.allclose(np.nanmedian(out1.imageget('STOKES Q')),
                           np.nanmedian(out2.imageget('STOKES Q')),
                           atol=0.05)
        assert np.allclose(np.nanmedian(out1.imageget('STOKES U')),
                           np.nanmedian(out2.imageget('STOKES U')),
                           atol=0.05)

        # try to call with a bad ip file
        with pytest.raises(ValueError):
            step(df.copy(), fileip='badfile.fits')
        capt = capsys.readouterr()
        assert 'Fileip (badfile.fits) was not found' in capt.err
