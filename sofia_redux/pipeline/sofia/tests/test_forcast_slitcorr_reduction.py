# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the FORCAST Slitcorr Reduction class."""

import os
import pickle

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.pipeline.gui.qad_viewer import QADViewer
from sofia_redux.pipeline.gui.matplotlib_viewer import MatplotlibViewer

try:
    from sofia_redux.pipeline.sofia.forcast_slitcorr_reduction \
        import FORCASTSlitcorrReduction
    from sofia_redux.pipeline.sofia.parameters.forcast_slitcorr_parameters\
        import FORCASTSlitcorrParameters
    HAS_DRIP = True
except ImportError:
    HAS_DRIP = False
    FORCASTSlitcorrReduction = None
    FORCASTSlitcorrParameters = None


@pytest.mark.skipif('not HAS_DRIP')
class TestFORCASTSlitcorrReduction(object):
    def make_file(self, tmpdir, fname='bFT001_0001.fits'):
        """Retrieve a basic test FITS file for FORCAST."""
        from sofia_redux.instruments.forcast.tests.resources \
            import raw_specdata
        hdul = raw_specdata()
        hdul[0].header['DATE-OBS'] = '2018-09-01T00:00:00.000'

        # add a gradient to the flux data to fit
        hdul[0].data[:] = np.repeat(np.arange(256), 256).reshape(256, 256)

        tmpfile = tmpdir.join(fname)
        ffile = str(tmpfile)
        hdul.writeto(ffile, overwrite=True)

        return ffile

    def standard_setup(self, tmpdir, step, nfiles=1):
        red = FORCASTSlitcorrReduction()
        ffile = []
        for i in range(nfiles):
            fn = 'bFT001_000{}.fits'.format(i + 1)
            ffile.append(self.make_file(tmpdir, fname=fn))
        red.output_directory = tmpdir

        red.load(ffile)
        red.load_parameters()
        idx = red.recipe.index(step)

        # process up to current step
        for i in range(idx):
            red.step()

        if nfiles == 1:
            ffile = ffile[0]
        return ffile, red, idx

    def test_all_steps(self, tmpdir):
        # exercises nominal behavior for a typical reduction
        red = FORCASTSlitcorrReduction()
        ffile = self.make_file(tmpdir)
        red.load([ffile, ffile])
        red.output_directory = tmpdir
        red.load_parameters()

        # run all steps
        for idx, step in enumerate(red.recipe):
            # test save
            parset = red.parameters.current[idx]
            # set rectification to simulated
            if 'simwavecal' in parset:
                parset['simwavecal']['value'] = True
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
            if step == 'checkhead':
                continue
            msg = '-- Pipeline step: {}'.format(red.processing_steps[step])
            assert msg in history

        # test that final fit is appropriate: should return a linear
        # fit in y with intercept 0 and slope 1 / 128, give or take a pixel
        found = False
        for fn in red.out_files:
            if 'SCR' in fn:
                found = True
                hdul = fits.open(fn)
                sccoeff = hdul[0].header['SCCOEFF']
                for i, sc in enumerate(sccoeff.split(',')):
                    if i == 0:
                        assert np.allclose(float(sc), 0, atol=0.01)
                    elif i == 1:
                        assert np.allclose(float(sc), 1 / 128, atol=0.001)
                    else:
                        assert np.allclose(float(sc), 0, atol=1e-7)
                break
        assert found

    def test_parameter_copy(self, tmpdir):
        # test that parameter copy gets the additional
        # config attributes necessary for forcast
        red = FORCASTSlitcorrReduction()
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
        # parameter config has to include slitcorr and wavecal flags
        param = FORCASTSlitcorrParameters()
        cfg = param.to_config()
        assert cfg['wavecal'] is False
        assert cfg['spatcal'] is False
        assert cfg['slitcorr'] is True

    def test_register_viewers(self, mocker):
        mocker.patch.object(QADViewer, '__init__', return_value=None)
        mocker.patch.object(MatplotlibViewer, '__init__', return_value=None)

        red = FORCASTSlitcorrReduction()
        vz = red.register_viewers()
        # 3 viewers -- QAD, profile, spectra
        assert len(vz) == 3
        assert isinstance(vz[0], QADViewer)
        assert isinstance(vz[1], MatplotlibViewer)
        assert isinstance(vz[2], MatplotlibViewer)

    def test_extract_median_spectra(self, tmpdir, capsys):
        ffile, red, idx = self.standard_setup(tmpdir,
                                              'extract_median_spectra')
        flux_image = red.input[0]['FLUX'].data
        red_copy = pickle.dumps(red)

        # check error for too many apertures
        red.input[0][0].header['APPOSO01'] \
            = ','.join([str(f) for f in range(200)])
        with pytest.raises(ValueError) as err:
            red.step()
        assert 'Too many apertures' in str(err)

        # test save_1d
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['save_1d']['value'] = True
        red.step()
        fname = red.input[0][0].header['FILENAME']
        assert os.path.exists(tmpdir.join(fname.replace('MSM', 'MSP')))

        medspec = red.input[0]['SPECTRAL_FLUX'].data

        # step again to normalize
        parset = red.parameters.current[idx + 1]
        parset['save_1d']['value'] = True
        red.step()
        fname = red.input[0][0].header['FILENAME']
        assert os.path.exists(tmpdir.join(fname.replace('NIM', 'NMS')))

        normspec = red.input[0]['SPECTRAL_FLUX'].data

        # check ranges of median and normed spectra: median should
        # be near the range of the input data, normalize should go 0-2,
        # with 1 near center row
        assert np.allclose(medspec[0], np.nanmin(flux_image), atol=0.1)
        assert np.allclose(medspec[-1], np.nanmax(flux_image), atol=0.1)
        ns = normspec.shape[0]
        assert np.allclose(normspec[0], 0, atol=0.1)
        assert np.allclose(normspec[ns // 2], 1, atol=0.1)
        assert np.allclose(normspec[-1], 2, atol=0.1)

    def test_make_slitcorr(self, tmpdir, capsys):
        ffile, red, idx = self.standard_setup(tmpdir,
                                              'make_slitcorr')
        red_copy = pickle.dumps(red)

        # test alternate fit methods - should be same for simulated data
        parset = red.parameters.current[idx]
        parset['method']['value'] = '1D'
        red.step()
        fit1d = red.input[0][0].data
        assert 'Mean reduced chi^2' in capsys.readouterr().out

        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['method']['value'] = '2D'
        red.step()
        fit2d = red.input[0][0].data
        assert 'Reduced Chi-Squared' in capsys.readouterr().out

        assert np.allclose(fit1d, fit2d, atol=1e-3)
