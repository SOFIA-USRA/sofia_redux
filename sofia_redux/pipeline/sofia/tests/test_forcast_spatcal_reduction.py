# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the FORCAST Spatcal Reduction class."""

import os
import pickle

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.pipeline.reduction import Reduction
from sofia_redux.pipeline.gui.qad_viewer import QADViewer
from sofia_redux.pipeline.gui.matplotlib_viewer import MatplotlibViewer
try:
    from sofia_redux.pipeline.sofia.forcast_spatcal_reduction \
        import FORCASTSpatcalReduction
    from sofia_redux.pipeline.sofia.parameters.forcast_spatcal_parameters\
        import FORCASTSpatcalParameters
    HAS_DRIP = True
except ImportError:
    HAS_DRIP = False
    FORCASTSpatcalReduction = None
    FORCASTSpatcalParameters = None


@pytest.mark.skipif('not HAS_DRIP')
class TestFORCASTSpatcalReduction(object):
    def make_file(self, tmpdir, fname='bFT001_0001.fits'):
        """Retrieve a basic test FITS file for FORCAST."""
        from sofia_redux.instruments.forcast.tests.resources \
            import raw_specdata
        hdul = raw_specdata()
        hdul[0].header['DATE-OBS'] = '2018-09-01T00:00:00.000'

        tmpfile = tmpdir.join(fname)
        ffile = str(tmpfile)
        hdul.writeto(ffile, overwrite=True)

        return ffile

    def standard_setup(self, tmpdir, step, nfiles=1):
        red = FORCASTSpatcalReduction()
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
            # set a necessary non-default parameter
            parset = red.parameters.current[i]
            if 'num_aps' in parset:
                parset['num_aps']['value'] = 3
            red.step()

        if nfiles == 1:
            ffile = ffile[0]
        return ffile, red, idx

    def test_startup(self):
        red = FORCASTSpatcalReduction()
        assert isinstance(red, Reduction)

    def test_load_basic(self, tmpdir):
        red = FORCASTSpatcalReduction()
        ffile = self.make_file(tmpdir)

        red.load(ffile)
        red.load_fits()
        assert len(red.input) == 1
        assert isinstance(red.input[0], fits.HDUList)
        assert isinstance(red.parameters, FORCASTSpatcalParameters)

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
        red = FORCASTSpatcalReduction()
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

        red = FORCASTSpatcalReduction()
        vz = red.register_viewers()
        # 4 viewers -- QAD, profile, spectra, residuals
        assert len(vz) == 4
        assert isinstance(vz[0], QADViewer)
        assert isinstance(vz[1], MatplotlibViewer)
        assert isinstance(vz[2], MatplotlibViewer)
        assert isinstance(vz[3], MatplotlibViewer)

    def test_all_steps(self, tmpdir):
        # exercises nominal behavior for a typical reduction
        red = FORCASTSpatcalReduction()
        ffile = self.make_file(tmpdir)
        red.load([ffile, ffile])
        red.output_directory = tmpdir
        red.load_parameters()

        # run all steps
        for idx, step in enumerate(red.recipe):
            # test save
            parset = red.parameters.current[idx]
            # set a necessary non-default parameter
            if 'num_aps' in parset:
                parset['num_aps']['value'] = 3
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

        # test that final fit is appropriate: should return a linear
        # fit in y with slope = pixel scale (.768)
        found = False
        for fn in red.out_files:
            if 'SCL' in fn:
                found = True
                hdul = fits.open(fn)
                scoeff = hdul[0].header['SCOEFF']
                for i, sc in enumerate(scoeff.split(',')):
                    if i == 0:
                        continue
                    elif i == 1:
                        assert np.allclose(float(sc), 0.77, atol=.02)
                    else:
                        assert np.allclose(float(sc), 0, atol=1e-3)
                break
        assert found

    def test_parameter_copy(self, tmpdir):
        # test that parameter copy gets the additional
        # config attributes necessary for forcast
        red = FORCASTSpatcalReduction()
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
        param = FORCASTSpatcalParameters()
        cfg = param.to_config()
        assert cfg['wavecal'] is False
        assert cfg['spatcal'] is True
        assert cfg['slitcorr'] is False

    def test_fit_traces(self, tmpdir, capsys):
        ffile, red, idx = self.standard_setup(tmpdir, 'fit_traces')
        red_copy = pickle.dumps(red)

        # test weighted/unweighted fit - should be same for simulated data -
        # and existing/simulated wavecal
        parset = red.parameters.current[idx]
        parset['weighted']['value'] = False
        red.step()
        # expected fit coefficient
        capt = capsys.readouterr()
        assert '(1, 0) : 0.7' in capt.out
        assert 'Fit is unweighted' in capt.out
        # default wavecal file for G111
        assert '.fits for wavelength calibration' in capt.out
        assert np.min(red.input[0]['WAVECAL'].data) > 8
        assert np.max(red.input[0]['WAVECAL'].data) < 14

        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['weighted']['value'] = True
        parset['wavefile']['value'] = ''
        red.step()
        capt = capsys.readouterr()
        assert '(1, 0) : 0.7' in capt.out
        assert 'Weighting fit' in capt.out
        # simulated wavecal data: pixel positions
        assert 'pixel positions for wavelength calibration' in capt.out
        assert np.nanmin(red.input[0]['WAVECAL'].data) == 0
        assert np.nanmax(red.input[0]['WAVECAL'].data) == 255
