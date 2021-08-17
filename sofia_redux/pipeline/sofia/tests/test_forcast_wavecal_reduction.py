# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the FORCAST Wavecal Reduction class."""

import os
import pickle

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.pipeline.reduction import Reduction
from sofia_redux.pipeline.gui.qad_viewer import QADViewer
from sofia_redux.pipeline.gui.matplotlib_viewer import MatplotlibViewer

try:
    from sofia_redux.pipeline.sofia.forcast_wavecal_reduction \
        import FORCASTWavecalReduction
    from sofia_redux.pipeline.sofia.parameters.forcast_wavecal_parameters\
        import FORCASTWavecalParameters
    HAS_DRIP = True
except ImportError:
    HAS_DRIP = False
    FORCASTWavecalReduction = None
    FORCASTWavecalParameters = None


@pytest.mark.skipif('not HAS_DRIP')
class TestFORCASTWavecalReduction(object):
    def make_file(self, tmpdir, fname='bFT001_0001.fits'):
        """Retrieve a basic test FITS file for FORCAST."""
        from sofia_redux.instruments.forcast.tests.resources \
            import raw_specdata
        hdul = raw_specdata()
        hdul[0].header['DATE-OBS'] = '2018-09-01T00:00:00.000'

        # add some lines to fit, within half pixel of where
        # they are expected to be from the last wavecal
        hdul[0].header['SPECTEL1'] = 'FOR_G063'
        self.pixpos = [62, 72, 82, 97, 102, 127, 134,
                       141, 146, 159, 165, 180, 233]
        for pp in self.pixpos:
            hdul[0].data[0, :, pp] += 1000

        tmpfile = tmpdir.join(fname)
        ffile = str(tmpfile)
        hdul.writeto(ffile, overwrite=True)
        return ffile

    def make_linefile(self, tmpdir):
        linefile = tmpdir.join('linelist.txt')
        lines = "#5.5751920\n5.6437855\n5.7618990\n" \
                "#5.8265619\n5.8847108\n#5.9378405\n" \
                "#5.9804268\n6.0553436\n6.1126299\n" \
                "#6.1788969\n6.4164567\n6.4935565\n" \
                "6.5750675\n6.6351962\n6.7877822\n" \
                "6.8587682\n#6.9643941\n7.0436258\n" \
                "#7.1603341\n#7.2826805\n#7.4579773\n" \
                "7.6681852\n#7.8526405\n"
        linefile.write(lines)
        return linefile

    def standard_setup(self, tmpdir, step, nfiles=1):
        red = FORCASTWavecalReduction()
        ffile = []
        for i in range(nfiles):
            fn = 'bFT001_000{}.fits'.format(i + 1)
            ffile.append(self.make_file(tmpdir, fname=fn))
        red.output_directory = tmpdir

        linefile = self.make_linefile(tmpdir)

        red.load(ffile)
        red.load_parameters()
        idx = red.recipe.index(step)

        # process up to current step
        for i in range(idx):
            # set a necessary non-default parameter
            parset = red.parameters.current[i]
            if 'linefile' in parset:
                parset['linefile']['value'] = linefile
            red.step()

        if nfiles == 1:
            ffile = ffile[0]
        return ffile, red, idx

    def check_lines(self, l1, l2, atol=.005):
        # helper to test lines and positions from header keys
        l1 = l1.split(',')
        l2 = l2.split(',')
        if len(l1) != len(l2):
            return False
        match = True
        for i in range(len(l1)):
            match &= np.allclose(float(l1[i]), float(l2[i]), atol=atol)
        return match

    def test_startup(self):
        red = FORCASTWavecalReduction()
        assert isinstance(red, Reduction)

    def test_load_basic(self, tmpdir):
        red = FORCASTWavecalReduction()
        ffile = self.make_file(tmpdir)

        red.load(ffile)
        red.load_fits()
        assert len(red.input) == 1
        assert isinstance(red.input[0], fits.HDUList)
        assert isinstance(red.parameters, FORCASTWavecalParameters)

        # test updated keys
        header = red.input[0][0].header
        assert header['PIPELINE'] == red.pipe_name
        assert header['PIPEVERS'] == red.pipe_version
        assert 'ASSC_AOR' in header
        assert 'ASSC_OBS' in header
        assert 'ASSC_MSN' in header
        assert header['OBS_ID'].startswith('P_')
        assert header['PROCSTAT'] == 'LEVEL_2'

    def test_load_intermediate(self, tmpdir, capsys):
        red = FORCASTWavecalReduction()
        ffile = self.make_file(tmpdir)

        # Set prodtype -- now looks like an old-style
        # intermediate file with variance and expmap
        fits.setval(ffile, 'PRODTYPE', value='stacked')
        red.load(ffile)
        # check the loaded file now has 3 extensions
        assert len(red.input[0]) == 3
        assert 'PRIMARY' in red.input[0]
        assert 'ERROR' in red.input[0]
        assert 'EXPOSURE' in red.input[0]

        # keep only first two planes -- now looks like
        # flux and variance only
        hdul = fits.open(ffile, mode='update')
        hdul[0].data = hdul[0].data[0:2, :, :]
        hdul.flush()
        hdul.close()
        red.load(ffile)
        # check the loaded file now has 2 extensions
        assert len(red.input[0]) == 2
        assert 'PRIMARY' in red.input[0]
        assert 'ERROR' in red.input[0]

        # write reorganized file to disk, load it
        # again and verify it's unchanged
        # (new-style intermediate)
        red.input[0].writeto(ffile, overwrite=True)
        red.load(ffile)
        assert len(red.input[0]) == 2
        assert 'PRIMARY' in red.input[0]
        assert 'ERROR' in red.input[0]

        # test error for no further steps
        fits.setval(ffile, 'PRODTYPE', value='rectified_image')
        with pytest.raises(ValueError):
            red.load(ffile)
        assert 'No steps to run' in capsys.readouterr().err

    def test_register_viewers(self, mocker):
        mocker.patch.object(QADViewer, '__init__', return_value=None)
        mocker.patch.object(MatplotlibViewer, '__init__', return_value=None)

        red = FORCASTWavecalReduction()
        vz = red.register_viewers()
        # 4 viewers -- QAD, profile, spectra, residuals
        assert len(vz) == 4
        assert isinstance(vz[0], QADViewer)
        assert isinstance(vz[1], MatplotlibViewer)
        assert isinstance(vz[2], MatplotlibViewer)
        assert isinstance(vz[3], MatplotlibViewer)

    def test_all_steps(self, tmpdir):
        # exercises nominal behavior for a typical reduction
        red = FORCASTWavecalReduction()
        ffile = self.make_file(tmpdir)
        linefile = self.make_linefile(tmpdir)
        red.load([ffile, ffile])
        red.output_directory = tmpdir
        red.load_parameters()

        # run all steps
        for idx, step in enumerate(red.recipe):
            # test save
            parset = red.parameters.current[idx]
            if 'linefile' in parset:
                parset['linefile']['value'] = linefile
            if 'save' in parset:
                parset['save']['value'] = True
                red.step()

                fn = ''
                for fn in red.out_files:
                    if red.prodnames[red.prodtype_map[step]] in fn:
                        break
                assert os.path.isfile(fn)
            else:
                red.step()

        # check all steps were run, in history of last file
        hdul = red.input[0]
        history = hdul[0].header['HISTORY']
        for step in red.recipe:
            if step == 'checkhead' or step == 'stack_dithers':
                continue
            msg = '-- Pipeline step: {}'.format(red.processing_steps[step])
            assert msg in history

        # test that final fit is appropriate: should return an almost
        # linear fit in x with intercept near 5 um, dw~.012
        found = False
        for fn in red.out_files:
            if 'WCL' in fn:
                found = True
                hdul = fits.open(fn)
                scoeff = hdul[0].header['WCOEFF']
                for i, sc in enumerate(scoeff.split(',')):
                    if i == 0:
                        assert np.allclose(float(sc), 4.917, atol=.01)
                    elif i == 3:
                        assert np.allclose(float(sc), 0.012, atol=.01)
                    else:
                        assert np.allclose(float(sc), 0, atol=.01)
                break
        assert found

    def test_parameter_copy(self, tmpdir):
        # test that parameter copy gets the additional
        # config attributes necessary for forcast
        red = FORCASTWavecalReduction()
        ffile = self.make_file(tmpdir)

        red.load(ffile)

        param = red.parameters
        pcopy = param.copy()

        assert hasattr(pcopy, 'drip_cal_config')
        assert pcopy.drip_cal_config is not None

        assert hasattr(pcopy, 'drip_config')
        assert pcopy.drip_config is not None

        assert not hasattr(pcopy, 'pipecal_config')

    def test_parameter_to_cfg(self):
        # parameter config has to include spatcal and wavecal flags
        param = FORCASTWavecalParameters()
        cfg = param.to_config()
        assert cfg['wavecal'] is True
        assert cfg['spatcal'] is False
        assert cfg['slitcorr'] is False

    def test_extract_summed_spectrum(self, tmpdir, capsys):
        ffile, red, idx = self.standard_setup(tmpdir,
                                              'extract_summed_spectrum')
        red_copy = pickle.dumps(red)

        # test aperture identification -- all should succeed
        # for simulated data
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'fix to center'
        red.step()
        assert 'Fixing aperture to slit center' in capsys.readouterr().out
        r1 = red.input[0]['SPECTRAL_FLUX'].data

        # fixing to input fails without input position
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'fix to input'
        with pytest.raises(ValueError):
            red.step()
        assert 'Could not read' in capsys.readouterr().err

        # passes with it
        parset['appos']['value'] = '128'
        red.step()
        assert 'Fixing aperture to input position' in capsys.readouterr().out
        # extracted flux is same as center
        assert np.allclose(r1, red.input[0]['SPECTRAL_FLUX'].data,
                           equal_nan=True)

        # auto finds similar position
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'auto'
        red.step()
        assert 'Finding aperture position' in capsys.readouterr().out

        # extracted flux is close but not same as center because radius
        # is auto-set
        r2 = red.input[0]['SPECTRAL_FLUX'].data
        nn = ~np.isnan(r1) & ~np.isnan(r2)
        assert np.allclose(np.mean(r2[nn]), np.mean(r2[nn]), rtol=0.2)

        # setting guess value gives same answer
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'auto'
        parset['appos']['value'] = '128'
        parset['detrend_order']['value'] = -1
        red.step()
        assert np.allclose(r2, red.input[0]['SPECTRAL_FLUX'].data,
                           equal_nan=True)
        assert 'Detrending' not in capsys.readouterr().out

        # test flatten spectrum
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'fix to center'
        parset['detrend_order']['value'] = 2
        red.step()
        # should be nearly the same for already flat spec
        r3 = red.input[0]['SPECTRAL_FLUX'].data
        assert np.allclose(r1, r3, equal_nan=True, rtol=0.05)
        assert 'Detrending' in capsys.readouterr().out

        # apply a gradient to input image-- should be taken out
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'fix to center'
        parset['detrend_order']['value'] = 1
        grad = np.arange(1, 257, dtype=float)
        red.input[0]['FLUX'].data += grad - np.mean(grad)
        red.step()

        # result should be same as original except for a constant
        r4 = red.input[0]['SPECTRAL_FLUX'].data
        assert np.allclose(r1 - np.nanmedian(r1),
                           r4 - np.nanmedian(r4),
                           atol=.5, equal_nan=True)
        assert 'Detrending' in capsys.readouterr().out

    def test_identify_lines(self, tmpdir, capsys, mocker):
        ffile, red, idx = self.standard_setup(tmpdir,
                                              'identify_lines')
        linefile = self.make_linefile(tmpdir)
        parset = red.parameters.current[idx]
        parset['linefile']['value'] = linefile
        red_copy = pickle.dumps(red)

        # default
        red.step()
        dw = red.input[0][0].header['LINEWAV']
        dp = red.input[0][0].header['LINEPOS']

        # test guess lines with accurate guesses
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['guess_lines']['value'] = '5.64,6.056,6.11,6.49,6.79,7.67'
        parset['guess_positions']['value'] = '62,97,102,134,159,233'
        red.step()
        assert self.check_lines(red.input[0][0].header['LINEWAV'], dw)
        assert self.check_lines(red.input[0][0].header['LINEPOS'], dp)

        # mismatched lines and positions
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['guess_lines']['value'] = '5.64,6.056,6.11,6.49,6.79,7.67'
        parset['guess_positions']['value'] = '49,83,88,120,145'
        with pytest.raises(ValueError) as err:
            red.step()
        assert 'do not match' in str(err)

        # too few lines
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['guess_lines']['value'] = '5.64'
        parset['guess_positions']['value'] = '49'
        with pytest.raises(ValueError) as err:
            red.step()
        assert 'Must have at least 2' in str(err)

        # missing line file
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['linefile']['value'] = ''
        with pytest.raises(ValueError) as err:
            red.step()
        assert 'No line list file' in str(err)

        # missing wave file with no positions
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['wavefile']['value'] = ''
        with pytest.raises(ValueError) as err:
            red.step()
        assert 'No wavecal file' in str(err)

        # line type 'either' should work for input with emission lines
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['line_type']['value'] = 'either'
        red.step()
        assert self.check_lines(red.input[0][0].header['LINEWAV'], dw)
        assert self.check_lines(red.input[0][0].header['LINEPOS'], dp)

        # line type 'absorption' should not work
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['line_type']['value'] = 'absorption'
        with pytest.raises(ValueError) as err:
            red.step()
        assert 'No lines found' in str(err)

        # bad line list
        linelist = tmpdir.join('badlist.txt')
        linelist.write('# okay comment\n; bad comment\n')
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['linefile']['value'] = str(linelist)
        with pytest.raises(ValueError) as err:
            red.step()
        assert 'Badly formatted line list' in str(err)

        # mock failure in fitpeaks - also raises error
        def bad_fit(*args, **kwargs):
            raise ValueError('bad')
        mocker.patch('sofia_redux.pipeline.sofia.'
                     'forcast_wavecal_reduction.fitpeaks1d',
                     bad_fit)
        red = pickle.loads(red_copy)
        with pytest.raises(ValueError) as err:
            red.step()
        assert 'No lines found' in str(err)

    def test_reidentify(self, tmpdir, capsys, mocker):
        ffile, red, idx = self.standard_setup(tmpdir, 'reidentify_lines')
        red_copy = pickle.dumps(red)

        # auto aperture should find 3 spectra
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'auto'
        parset['radius']['value'] = '8.929,10.042,8.935'
        red.step()
        assert 'Finding aperture positions' in capsys.readouterr().out
        d1 = red.input[0]['SPECTRAL_FLUX'].data
        a1 = red.input[0][0].header['APPOSO01']
        assert len(d1) == 3

        # same with guess positions
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'auto'
        parset['appos']['value'] = '60,128,196'
        parset['radius']['value'] = ''
        red.step()
        assert 'Finding aperture positions' in capsys.readouterr().out
        assert self.check_lines(a1, red.input[0][0].header['APPOSO01'])
        assert np.allclose(d1, red.input[0]['SPECTRAL_FLUX'].data, rtol=0.01)

        # and fixed positions, with and without radius
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'fix to input'
        parset['appos']['value'] = a1
        parset['radius']['value'] = '8.929,10.042,8.935'
        red.step()
        assert 'Fixing aperture' in capsys.readouterr().out
        assert self.check_lines(a1, red.input[0][0].header['APPOSO01'])
        assert np.allclose(d1, red.input[0]['SPECTRAL_FLUX'].data, rtol=0.01)

        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'fix to input'
        parset['appos']['value'] = a1
        parset['radius']['value'] = ''
        red.step()
        assert 'Fixing aperture' in capsys.readouterr().out
        assert self.check_lines(a1, red.input[0][0].header['APPOSO01'])
        assert np.allclose(d1, red.input[0]['SPECTRAL_FLUX'].data, rtol=0.01)

        # test additional options: line type = either, detrend
        # spectrum - still same
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'auto'
        parset['detrend_order']['value'] = 1
        red.input[0][0].header['LINETYPE'] = 'either'
        red.step()
        assert self.check_lines(a1, red.input[0][0].header['APPOSO01'])
        assert np.allclose(d1, red.input[0]['SPECTRAL_FLUX'].data, rtol=0.2)

        # invalid detrend is quietly ignored
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'auto'
        parset['radius']['value'] = '8.929,10.042,8.935'
        parset['detrend_order']['value'] = -100
        red.step()
        assert self.check_lines(a1, red.input[0][0].header['APPOSO01'])
        assert np.allclose(d1, red.input[0]['SPECTRAL_FLUX'].data, rtol=0.01)

        # but absorption type throws fit off -- all fit positions are NaN
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'auto'
        red.input[0][0].header['LINETYPE'] = 'absorption'
        red.step()
        assert np.all(np.isnan(red.input[0]['LINE_TABLE'].data))

        # error in fitpeaks does the same
        def bad_fit(*args, **kwargs):
            raise ValueError('bad')
        mocker.patch('sofia_redux.pipeline.sofia.'
                     'forcast_wavecal_reduction.fitpeaks1d',
                     bad_fit)
        red = pickle.loads(red_copy)
        red.step()
        assert np.all(np.isnan(red.input[0]['LINE_TABLE'].data))

    def test_fit_lines(self, tmpdir, capsys):
        ffile, red, idx = self.standard_setup(tmpdir, 'fit_lines')
        red_copy = pickle.dumps(red)

        # test weighted/unweighted fit - should be same for simulated data -
        # and existing/simulated wavecal
        parset = red.parameters.current[idx]
        parset['weighted']['value'] = False
        red.step()
        # expected fit coefficient
        capt = capsys.readouterr()
        assert '(0, 0) : 4.9' in capt.out
        assert 'Fit is unweighted' in capt.out
        # default wavecal file
        assert '.fits for spatial calibration' in capt.out
        assert np.min(red.input[0]['SPATCAL'].data) > -1
        assert np.max(red.input[0]['SPATCAL'].data) < 200

        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['weighted']['value'] = True
        parset['spatfile']['value'] = ''
        red.step()
        capt = capsys.readouterr()
        assert '(0, 0) : 4.9' in capt.out
        assert 'Weighting fit' in capt.out
        # simulated spatcal data: pixel positions
        assert 'simulated calibration' in capt.out
        spatcal = red.input[0]['SPATCAL'].data

        # range should match slit height
        assert np.allclose(np.nanmin(spatcal), 0, atol=1)
        assert np.allclose(np.nanmax(spatcal), 194, atol=1)

        # min is at bottom edge, max is at top edge
        assert np.min(np.nanargmin(spatcal, axis=0)) == 0
        assert np.max(np.nanargmax(spatcal, axis=0)) == 255

    def test_sim_spatcal_badflat(self):
        red = FORCASTWavecalReduction()

        # missing calres
        with pytest.raises(AttributeError):
            red._sim_spatcal((10, 10))

        # missing flat file
        red.calres = {}
        with pytest.raises(ValueError) as err:
            red._sim_spatcal((10, 10))
        assert 'Missing order mask' in str(err)

        # bad flat file
        red.calres = {'maskfile': 'badfile.fits'}
        with pytest.raises(ValueError) as err:
            red._sim_spatcal((10, 10))
        assert 'Missing order mask' in str(err)
