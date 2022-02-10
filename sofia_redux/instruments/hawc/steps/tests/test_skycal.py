# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import shutil

import numpy as np
import pandas as pd
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepskycal import StepSkycal
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, flat_data, pixel_data


class TestSkycal(DRPTestCase):
    def make_data(self, tmpdir, rval=1.0, tval=2.0, pixdata=0):
        intcal = flat_data(rval=1.0, tval=1.0, seed=42)
        os.makedirs(tmpdir.join('intcals'))
        intcal_fname = str(tmpdir.join('intcals', 'test_DCL.fits'))
        intcal.writeto(intcal_fname)

        flat1 = flat_data(seed=43, rval=rval, tval=tval)
        flat1_fname = str(tmpdir.join('flat1.fits'))
        flat1.writeto(flat1_fname)

        flat2 = flat_data(seed=44, rval=rval, tval=tval)
        flat2_fname = str(tmpdir.join('flat2.fits'))
        flat2.writeto(flat2_fname)

        if pixdata > 0:
            # make pixeldata files
            pixnames = []
            for i in range(pixdata):
                h, d = pixel_data(scan=f'File-{i+1}', seed=42 + i)
                outfile = tmpdir.join(f'pixel-{i+1}.dat')
                with open(outfile, 'w') as fh:
                    for line in h:
                        fh.write(line)
                d.to_csv(outfile, sep='\t', index=None, header=0, mode='a')
                pixnames.append(outfile)
            return intcal_fname, flat1_fname, flat2_fname, pixnames
        else:
            return intcal_fname, flat1_fname, flat2_fname

    def test_success(self, tmpdir):
        i1, f1, f2 = self.make_data(tmpdir)
        df1 = DataFits(f1)
        df2 = DataFits(f2)

        ttor = 1.3
        step = StepSkycal()
        out = step([df1, df2], ttor=ttor)

        # output is single data fits
        assert isinstance(out, DataFits)

        # check for flat extensions
        ext = ['R ARRAY GAIN', 'T ARRAY GAIN',
               'R BAD PIXEL MASK', 'T BAD PIXEL MASK']
        assert out.imgnames == ext
        rgain = out.imageget('R ARRAY GAIN')
        tgain = out.imageget('T ARRAY GAIN')
        rbad = out.imageget('R BAD PIXEL MASK')
        tbad = out.imageget('T BAD PIXEL MASK')

        # output R flat should have expected median value of 1.0
        assert np.allclose(np.nanmedian(rgain), 1.0, atol=0.01)

        # output T flat should be scaled to desired ratio
        assert np.allclose(np.nanmedian(tgain) / np.nanmedian(rgain), 1.3)

        # nan pixels in flat should match bad pixel mask
        r_nan = np.isnan(rgain)
        t_nan = np.isnan(tgain)
        assert np.allclose(r_nan, rbad == 1)
        assert np.allclose(t_nan, tbad == 2)

        # output flat will have more bad pixels than input, since they
        # "or" over input and intcal
        assert np.sum(r_nan) > np.sum(df1.imageget('R BAD PIXEL MASK') == 1)
        assert np.sum(r_nan) > np.sum(df2.imageget('R BAD PIXEL MASK') == 1)
        assert np.sum(t_nan) > np.sum(df1.imageget('T BAD PIXEL MASK') == 2)
        assert np.sum(t_nan) > np.sum(df2.imageget('T BAD PIXEL MASK') == 2)

    def test_missing_dcl(self, tmpdir, capsys):
        i1, f1, f2 = self.make_data(tmpdir, rval=2.0)
        df1 = DataFits(f1)
        df2 = DataFits(f2)

        # with intcals
        step = StepSkycal()
        out = step([df1, df2], ttor=1)
        rgain = out.imageget('R ARRAY GAIN').copy()
        tgain = out.imageget('T ARRAY GAIN').copy()
        assert 'No dcl files' not in capsys.readouterr().err

        # without intcals: warns but continues
        shutil.rmtree(tmpdir.join('intcals'))
        out = step([df1, df2], ttor=1)
        rgain2 = out.imageget('R ARRAY GAIN')
        tgain2 = out.imageget('T ARRAY GAIN')
        assert 'No dcl files' in capsys.readouterr().err

        # data does not match
        assert not np.allclose(rgain, rgain2, equal_nan=True)
        assert not np.allclose(tgain, tgain2, equal_nan=True)

    def test_options(self, tmpdir, capsys):
        i1, f1, f2 = self.make_data(tmpdir, rval=2.0)
        df1 = DataFits(f1)
        df2 = DataFits(f2)

        # test normalize option: median value of R will
        # be normalized to 1.0, T will be scaled to 3.0
        step = StepSkycal()
        out = step([df1, df2], ttor=3.0, normalize=True)
        assert np.allclose(np.nanmedian(out.imageget('R ARRAY GAIN')), 1.0)
        assert np.allclose(np.nanmedian(out.imageget('T ARRAY GAIN')), 3.0)
        assert 'Normalizing' in capsys.readouterr().out

        # comparison plot is written
        assert len(step.auxout) == 1
        assert os.path.isfile(tmpdir.join('flat1.SCL_comparison.png'))
        os.remove(str(tmpdir.join('flat1.SCL_comparison.png')))

        # set config to not find pipeline flat for comparison
        step = StepSkycal()
        df1.config['mkflat']['scalfile'] = 'bad_directory'
        step([df1, df2])

        # comparison plot is not written
        assert len(step.auxout) == 0
        assert not os.path.isfile(tmpdir.join('flat1.SCL_comparison.png'))
        assert 'No default pipeline flat found' in capsys.readouterr().err

    def test_read_refpix(self, capsys):
        df = DataFits()
        df.setheadval('SPECTEL1', 'HAWE')
        step = StepSkycal()
        step.datain = [df]
        step.runstart([df], {})

        expected = os.path.join(step.getarg('ref_pixpath'),
                                step.getarg('ref_pixfile')[-1])
        expected = os.path.expandvars(expected)

        # test defaults
        result = step.read_refpix()
        assert result == expected

        # bad spectel
        df.setheadval('SPECTEL1', 'HAWQ')
        with pytest.raises(ValueError):
            step.read_refpix()
        capt = capsys.readouterr()
        assert 'Cannot parse waveband' in capt.err

        df.setheadval('SPECTEL1', '')
        with pytest.raises(ValueError):
            step.read_refpix()
        capt = capsys.readouterr()
        assert 'Cannot parse waveband' in capt.err

        # bad arglist: passes with warning
        df.setheadval('SPECTEL1', 'HAWE')
        step.runstart([df], {'ref_pixfile': [1, 2, 3]})
        result = step.read_refpix()
        capt = capsys.readouterr()
        assert 'No reference pixel file' in capt.err
        assert result is None

        # bad file path
        step.runstart([df], {'ref_pixfile': ['1', '2', '3', '4', '5']})
        with pytest.raises(ValueError):
            step.read_refpix()
        capt = capsys.readouterr()
        assert 'not found' in capt.err

        # valid arguments, but deliberately left blank: warns only
        step.runstart([df], {'ref_pixfile': ['1', '2', '3', '4', '']})
        result = step.read_refpix()
        capt = capsys.readouterr()
        assert 'No reference pixel file specified' in capt.err
        assert result is None

    def test_pixeldata(self, tmpdir, capsys):
        i1, f1, f2, pixnames = self.make_data(tmpdir, pixdata=2)
        df1 = DataFits(f1)
        df2 = DataFits(f2)
        step = StepSkycal()
        step([df1, df2])

        # check for expected output files
        assert len(step.auxout) == 4
        assert os.path.isfile(tmpdir.join('flat1.SCL_comparison.png'))
        assert os.path.isfile(tmpdir.join('flat1.SCL_histogram.png'))
        assert os.path.isfile(tmpdir.join('flat1.SCL_pix2pix.png'))
        assert os.path.isfile(tmpdir.join('flat1.SCL.dat'))

        # check output table for appropriate values
        col_names = ('ch', 'gain', 'weight', 'flag', 'eff',
                     'Gmux1', 'Gmux2', 'idx', 'sub', 'row', 'col')
        p1 = pd.read_table(pixnames[0], names=col_names, comment='#')
        p2 = pd.read_table(pixnames[1], names=col_names, comment='#')
        out = pd.read_table(tmpdir.join('flat1.SCL.dat'),
                            names=col_names, comment='#')

        # output floats are mean combined and trimmed to 3 decimal places
        mean_gain = np.mean([p1['gain'], p2['gain']], axis=0)
        assert np.allclose(out['gain'], mean_gain, atol=.001)
        mean_weight = np.mean([p1['weight'], p2['weight']], axis=0)
        assert np.allclose(out['weight'], mean_weight, atol=.001)
        mean_eff = np.mean([p1['eff'], p2['eff']], axis=0)
        assert np.allclose(out['eff'], mean_eff, atol=.001)

        # output flags are or-combined, with '-' indicating no
        # flag in either table
        expected = ['B', 'g', 'n', 'Bg', 'gn', 'Bn', '-']
        assert np.all(out.flag.isin(expected))
        # all combinations should appear in the pseudo-random data set
        for flag in expected:
            assert np.sum(out['flag'] == flag) > 0

    def test_pixeldata_no_default(self, tmpdir, capsys):
        i1, f1, f2, pixnames = self.make_data(tmpdir, pixdata=3)
        df1 = DataFits(f1)
        df2 = DataFits(f2)

        # set config to not find default pixfile for comparison
        step = StepSkycal()
        df1.config['skycal']['ref_pixfile'] = ['', '', '', '', '']
        step([df1, df2])

        # pixel-to-pixel comparison plot is not written
        assert len(step.auxout) == 3
        assert not os.path.isfile(tmpdir.join('flat1.SCL_pix2pix.png'))
        assert 'No reference pixel file' in capsys.readouterr().err

        # check for the other expected output
        assert os.path.isfile(tmpdir.join('flat1.SCL_comparison.png'))
        assert os.path.isfile(tmpdir.join('flat1.SCL_histogram.png'))
        assert os.path.isfile(tmpdir.join('flat1.SCL.dat'))
