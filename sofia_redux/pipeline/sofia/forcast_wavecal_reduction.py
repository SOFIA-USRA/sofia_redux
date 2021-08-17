# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FORCAST Grism Wavecal Reduction pipeline steps"""

import os
import warnings

from astropy import log
from astropy.io import fits
import numpy as np
import pandas

from sofia_redux.pipeline.sofia.sofia_exception import SOFIAImportError
try:
    import sofia_redux.instruments.forcast
    assert sofia_redux.instruments.forcast
except ImportError:
    raise SOFIAImportError('FORCAST modules not installed')

from sofia_redux.instruments.forcast.hdmerge import hdmerge

from sofia_redux.pipeline.gui.matplotlib_viewer import MatplotlibViewer
from sofia_redux.pipeline.gui.qad_viewer import QADViewer
from sofia_redux.pipeline.sofia.forcast_reduction import FORCASTReduction
from sofia_redux.pipeline.sofia.forcast_spectroscopy_reduction \
    import FORCASTSpectroscopyReduction
from sofia_redux.pipeline.sofia.parameters.forcast_wavecal_parameters \
    import FORCASTWavecalParameters

from sofia_redux.spectroscopy.findapertures import find_apertures
from sofia_redux.spectroscopy.getapertures import get_apertures
from sofia_redux.spectroscopy.mkapmask import mkapmask
from sofia_redux.spectroscopy.extspec import extspec
from sofia_redux.spectroscopy.readflat import readflat
from sofia_redux.spectroscopy.readwavecal import readwavecal
from sofia_redux.spectroscopy.simwavecal2d import simwavecal2d

from sofia_redux.toolkit.utilities.fits \
    import hdinsert, getheader
from sofia_redux.toolkit.fitting.fitpeaks1d import fitpeaks1d, medabs_baseline
from sofia_redux.toolkit.fitting.polynomial import polyfitnd
from sofia_redux.toolkit.image.adjust import unrotate90
from sofia_redux.toolkit.interpolate import tabinv


def _min_func(_, y):
    """Minimum function to use for baseline in spectral stamp"""
    baseline = np.full(y.shape, float(np.nanmin(y)))
    return y - baseline, baseline


def _max_func(_, y):
    """Maximum function to use for baseline in spectral stamp"""
    baseline = np.full(y.shape, float(np.nanmax(y)))
    return y - baseline, baseline


class FORCASTWavecalReduction(FORCASTSpectroscopyReduction):
    r"""
    FORCAST wavelength calibration reduction steps.

    This reduction object defines specialized reduction steps
    for generating wavelength calibration data from spectroscopic
    input files.  It is selected by the SOFIA chooser only if a
    top-level configuration flag is supplied (wavecal=True).  The
    final output product from this reduction is a FITS file (\*WCL\*.fits)
    with PRODTYPE = 'wavecal'.  This file can be supplied to the
    standard spectroscopic pipeline, at the make_profiles step,
    to specify a new wavelength calibration.
    """
    def __init__(self):
        """Initialize the reduction object."""
        super().__init__()

        # descriptive attributes specific to calibration
        self.name = 'Wavecal'

        # product type definitions for spectral steps
        self.prodtype_map.update(
            {'make_profiles': 'spatial_profile',
             'extract_summed_spectrum': 'summed_spectrum',
             'identify_lines': 'lines_identified',
             'reidentify_lines': 'lines_reidentified',
             'fit_lines': 'lines_fit',
             'rectify': 'rectified_image'})
        self.prodnames.update(
            {'spatial_profile': 'PRF',
             'summed_spectrum': 'SSM',
             'lines_identified': 'LID',
             'lines_reidentified': 'LRD',
             'lines_fit': 'LFT',
             'rectified_image': 'RIM'})

        # invert the map for quick lookup of step from type
        self.step_map = {v: k for k, v in self.prodtype_map.items()}

        # default recipe and step names
        self.recipe = ['checkhead', 'clean', 'droop', 'nonlin',
                       'stack', 'stack_dithers', 'make_profiles',
                       'extract_summed_spectrum',
                       'identify_lines',
                       'reidentify_lines',
                       'fit_lines',
                       'rectify']
        self.processing_steps.update(
            {'extract_summed_spectrum': 'Extract First Spectrum',
             'identify_lines': 'Identify Lines',
             'reidentify_lines': 'Reidentify Lines',
             'fit_lines': 'Fit Lines',
             'rectify': 'Verify Rectification'})

    def load(self, data, param_class=None):
        """Call parent load, with spatcal parameters."""
        FORCASTReduction.load(self, data,
                              param_class=FORCASTWavecalParameters)

    def register_viewers(self):
        """Return new viewers."""
        prof = MatplotlibViewer()
        prof.name = 'ProfileViewer'
        prof.title = 'Spatial Profiles'
        prof.layout = 'rows'

        # using matplotlib viewer instead of Eye,
        # for line overlay purposes
        spec = MatplotlibViewer()
        spec.name = 'SpectralViewer'
        spec.title = 'Spectra'
        spec.layout = 'rows'

        resid = MatplotlibViewer()
        resid.name = 'ResidualViewer'
        resid.title = 'Fit Residuals'
        resid.layout = 'rows'
        resid.share_axes = None

        viewers = [QADViewer(), prof, spec, resid]

        return viewers

    def set_display_data(self, raw=False, filenames=None, regions=None,
                         residuals=None):
        """
        Store display data for custom viewers.

        Parameters
        ----------
        raw : bool
            If True, display data is taken from self.rawfiles.
            If False, display data is taken from self.input
        filenames : list of str, optional
            If provided and `raw` is False, file names will be
            passed to QADViewer instead of self.input.
        regions : list of str, optional
            File names of DS9 region files to pass to QADViewer.
        residuals : list of array-like, optional
            Fit residual data to pass to ResidualViewer, nplot x 3.
            The arrays should be x value, y value, residuals.
        """
        super().set_display_data(raw=raw, filenames=filenames,
                                 regions=regions, specviewer='matplotlib')

        # set residual plot if necessary
        disp_resid = []
        if residuals is not None:
            for residual in residuals:
                disp = {'args': [residual[0], residual[2]],
                        'kwargs': {'title': 'Fit Residuals vs. X',
                                   'xlabel': 'X (pixel)',
                                   'ylabel': 'Line position - '
                                             'model (pixel)'},
                        'plot_kwargs': {'linestyle': ' ',
                                        'marker': '.',
                                        'markersize': 8,
                                        'alpha': 0.7},
                        'overplot': [{'plot_type': 'hline',
                                      'args': [0],
                                      'kwargs': {'color': 'gray',
                                                 'linestyle': ':'}}],
                        }
                disp_resid.append(disp)
                disp = {'args': [residual[2], residual[1]],
                        'kwargs': {'title': 'Fit Residuals vs. Y',
                                   'xlabel': 'Line position - '
                                             'model (pixel)',
                                   'ylabel': 'Y (pixel)'},
                        'plot_kwargs': {'linestyle': ' ',
                                        'marker': '.',
                                        'markersize': 8,
                                        'alpha': 0.7},
                        'overplot': [{'plot_type': 'vline',
                                      'args': [0],
                                      'kwargs': {'color': 'gray',
                                                 'linestyle': ':'}}]
                        }
                disp_resid.append(disp)

        self.display_data['ResidualViewer'] = disp_resid

    def extract_summed_spectrum(self):
        """Extract high S/N spectrum from unrectified image."""
        # get parameters
        param = self.get_parameter_set()
        method = param.get_value('method')
        detrend_order = param.get_value('detrend_order')
        appos = param.get_value('appos')
        aprad = param.get_value('aprad')

        if str(method).strip().lower() == 'fix to center':
            log.info('Fixing aperture to slit center.')
            positions = None
            radii = self._parse_apertures(aprad, len(self.input))
            fix_ap = True
        elif str(method).strip().lower() == 'fix to input':
            log.info('Fixing aperture to input position.')
            positions = self._parse_apertures(appos, len(self.input))
            radii = self._parse_apertures(aprad, len(self.input))
            fix_ap = True
        else:
            log.info('Finding aperture position from Gaussian fits.')
            if str(appos).strip() == '':
                positions = None
            else:
                positions = self._parse_apertures(appos, len(self.input))
            radii = None
            fix_ap = False

        try:
            detrend_order = int(detrend_order)
            if detrend_order < 0:
                detrend_order = None
            else:
                log.info(f'Detrending spectrum with order {detrend_order} '
                         f'polynomial.')
        except (ValueError, TypeError):
            detrend_order = None

        results = []
        for i, hdul in enumerate(self.input):
            log.info('')
            log.info(hdul[0].header['FILENAME'])

            # retrieve data from input
            header = hdul[0].header
            image = hdul['FLUX'].data
            var = hdul['ERROR'].data ** 2
            mask = (hdul['BADMASK'].data < 1)
            wave = hdul['WAVEPOS'].data
            space = hdul['SLITPOS'].data
            spatmap = {1: hdul['SPATIAL_MAP'].data}
            profile = {1: [hdul['SLITPOS'].data,
                           hdul['SPATIAL_PROFILE'].data]}
            if positions is not None:
                guess = {1: positions[i]}
            else:
                guess = None

            ap = find_apertures(profile, npeaks=1, positions=guess,
                                fwhm=3.0, fix=fix_ap)
            ap[1][0]['sign'] = 1
            if fix_ap:
                ap[1][0]['psf_radius'] = radii[i][0]
            else:
                updated_ap = get_apertures(profile, ap, get_bg=False,
                                           refit_fwhm=False)[1]['apertures'][0]
                ap[1][0]['psf_radius'] = updated_ap['psf_radius']

            # set aperture trace to a row at the aperture position
            ap[1][0]['trace'] = np.full(len(wave), ap[1][0]['position'])

            # make aperture mask from defined aperture, no background
            apmask = mkapmask(space, wave, ap[1])

            rectimg = {1: {'image': image, 'variance': var, 'mask': mask,
                           'wave': wave, 'spatial': space, 'header': header,
                           'apmask': apmask, 'apsign': None}}

            spectra = extspec(rectimg, spatial_map=spatmap,
                              optimal=False, fix_bad=True,
                              sub_background=False,)[1]

            # flatten spectrum if desired
            if detrend_order is not None:
                specflux = spectra[0, 1, :].ravel()
                medval = float(np.nanmedian(specflux))

                pixpos = np.arange(specflux.size, dtype=float)
                trend = polyfitnd(pixpos, specflux, detrend_order, model=True,
                                  robust=2)
                flatspec = specflux - trend(pixpos) + medval
                spectra[0, 1, :] = flatspec

            # record initial aperture
            hdinsert(hdul[0].header, 'APPOSO01',
                     '{:.3f}'.format(ap[1][0]['position']),
                     'Aperture positions [pixel]')
            hdinsert(hdul[0].header, 'PSFRAD01',
                     '{:.3f}'.format(ap[1][0]['psf_radius']),
                     'Aperture PSF radii [pixel]')

            # attach spectral flux and error to output file:
            # shape is n_ap x n_wave
            exthead = fits.Header()
            hdinsert(exthead, 'BUNIT',
                     header.get('BUNIT', 'Me/s'), 'Data units')
            hdul.append(fits.ImageHDU(spectra[:, 1, :], exthead,
                                      name='SPECTRAL_FLUX'))
            hdinsert(exthead, 'BUNIT',
                     header.get('BUNIT', 'Me/s'), 'Data units')
            hdul.append(fits.ImageHDU(spectra[:, 2, :], exthead,
                                      name='SPECTRAL_ERROR'))

            # update output name
            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                self.write_output(hdul, outname)

            if param.get_value('save_1d'):
                log.info('')
                log.info('Saving 1D spectra:')
                spec = self._make_1d(hdul, wavecal=False)
                specname = outname.replace('SSM', 'SSP')
                spec[0].header['FILENAME'] = os.path.basename(specname)
                spec[0].header['PRODTYPE'] = 'spectra_1d'
                self.write_output(spec, specname)

            results.append(hdul)

        log.info('')
        self.input = results
        self.set_display_data()

    def identify_lines(self):
        """Initial identification of lines."""

        # get parameters
        param = self.get_parameter_set()
        wavefile = param.get_value('wavefile')
        linefile = param.get_value('linefile')
        line_type = param.get_value('line_type')
        window = param.get_value('window')
        sigma = param.get_value('sigma')
        guess_lines_input = param.get_value('guess_lines')
        guess_pos_input = param.get_value('guess_positions')

        # check for input guess positions
        guess_lines = []
        guess_pos = []
        if str(guess_lines_input).strip() != '':
            guess_lines = self._parse_apertures(guess_lines_input, 1)[0]
        if str(guess_pos_input).strip() != '':
            guess_pos = self._parse_apertures(guess_pos_input, 1)[0]
        if len(guess_lines) != len(guess_pos):
            raise ValueError('Input guess lines do not match '
                             'input guess positions.')
        if 0 < len(guess_lines) < 2:
            raise ValueError('Must have at least 2 line guesses.')

        mode = '{} {} {}'.format(self.calres['name'], self.calres['slit'],
                                 self.calres['dateobs'])
        if not os.path.isfile(linefile):
            msg = 'No line list file for {}'.format(mode)
            log.error(msg)
            raise ValueError(msg)

        # read wave and spatial cal file
        if len(guess_lines) > 0:
            log.info('Using input guesses for approximate wavecal.')
            order = 2 if len(guess_lines) > 2 else 1
            wfit_model = polyfitnd(guess_pos, guess_lines, order, model=True)
            wave = wfit_model(np.arange(256, dtype=float))
        else:
            log.info('Using existing wavecal as starting point.')
            if not os.path.isfile(wavefile):
                msg = 'No wavecal file for {}'.format(mode)
                log.error(msg)
                raise ValueError(msg)

            # read wavecal
            try:
                rotation = fits.getval(wavefile, 'ROTATION')
            except KeyError:
                rotation = 0
            wavecal, spatcal = readwavecal(wavefile, rotate=rotation)
            ctr = wavecal.shape[0] // 2

            # approximate wavecal
            wave = wavecal[ctr, :]

        # read linefile
        linelist = pandas.read_csv(linefile, names=['line'])['line']
        lines = []
        display_lines = []
        for line in linelist:
            if str(line).startswith('#'):
                try:
                    display_lines.append(float(line[1:]))
                except (ValueError, TypeError, IndexError):
                    continue
            else:
                try:
                    line = float(line)
                except ValueError:
                    raise ValueError('Badly formatted line list') from None
                lines.append(line)
                display_lines.append(line)
        log.debug('Display lines: {}'.format(display_lines))
        log.debug('Fitting lines: {}'.format(lines))

        if line_type == 'emission':
            baseline = _min_func
        elif line_type == 'absorption':
            baseline = _max_func
        else:
            baseline = medabs_baseline

        results = []
        for i, hdul in enumerate(self.input):
            log.info('')
            log.info(hdul[0].header['FILENAME'])

            # spectral flux
            pixelpos = hdul['WAVEPOS'].data
            spectrum = hdul['SPECTRAL_FLUX'].data[0]

            # guess position of each line, then fit it
            fitpos = []
            fitline = []
            for line in lines:
                guess = tabinv(wave, line)
                log.info(f'Line {line}, near pixel {guess}')
                start = int(np.round(guess - window / 2))
                start = 0 if start < 0 else start
                end = int(np.round(guess + window / 2))
                end = len(pixelpos) if end > len(pixelpos) else end
                try:
                    fit_peak = fitpeaks1d(
                        pixelpos[start:end], spectrum[start:end], npeaks=1,
                        guess=guess, stddev=sigma, box_width=('stddev', 3),
                        baseline_func=baseline)
                except ValueError:
                    log.info('Not found')
                    log.info('')
                else:
                    mval = fit_peak[0].mean.value
                    failure = (fit_peak.fit_info['ierr'] not in [1, 2, 3, 4])
                    if (failure or np.allclose(mval, pixelpos[start])
                            or np.allclose(mval, pixelpos[end - 1])):
                        log.info('Not found')
                        log.info('')
                    else:
                        log.info(f'Found at {mval}')
                        log.info('')
                        fitpos.append('{:.3f}'.format(mval))
                        fitline.append(line)

            if not fitpos:
                msg = 'No lines found.'
                log.error(msg)
                raise ValueError(msg)

            # record fit lines
            hdinsert(hdul[0].header, 'LINEWAV',
                     ','.join(str(ln) for ln in fitline),
                     comment='Line wavelengths [um]')
            hdinsert(hdul[0].header, 'LINEPOS', ','.join(fitpos),
                     comment='Line positions [pixels]')
            hdinsert(hdul[0].header, 'LINETYPE', line_type,
                     comment='Line type')
            hdinsert(hdul[0].header, 'LINEWID', sigma,
                     comment='Line width')

            # fit a 1D wavelength solution
            order = 2 if len(fitline) > 2 else 1
            wfit_model = polyfitnd(fitpos, fitline, order, model=True)
            wfit = wfit_model(pixelpos)
            log.info(f'1D order {order} fit to wavelengths:')
            log.info('')
            log.info(wfit_model)

            # store as WAVEPOS
            hdul['WAVEPOS'].data = wfit
            hdul['WAVEPOS'].header['BUNIT'] = 'um'

            # record display lines too
            disppos = tabinv(wfit, display_lines)
            log.debug(display_lines)
            log.debug(disppos)
            hdinsert(hdul[0].header, 'DISPWAV',
                     ','.join(str(ln) for ln in display_lines),
                     comment='Display line wavelengths [um]')
            hdinsert(hdul[0].header, 'DISPPOS',
                     ','.join(str(ln) for ln in disppos),
                     comment='Display line wavelengths [um]')

            # update output name
            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                self.write_output(hdul, outname)

            results.append(hdul)

        log.info('')
        self.input = results
        self.set_display_data()

    def reidentify_lines(self):
        """Reidentification of lines at various slit positions."""
        param = self.get_parameter_set()
        method = str(param.get_value('method')).lower()
        num_aps = param.get_value('num_aps')
        step = param.get_value('step')
        appos_input = param.get_value('appos')
        radius = param.get_value('radius')
        detrend_order = param.get_value('detrend_order')
        window = param.get_value('window')
        s2n_req = param.get_value('s2n')

        try:
            detrend_order = int(detrend_order)
            if detrend_order < 0:
                detrend_order = None
        except (ValueError, TypeError):
            detrend_order = None

        if 'step' in method:
            log.info('Stepping apertures up slit.')
            positions = None
            fix_ap = True
            radii = self._parse_apertures(radius, len(self.input))
        elif str(method).strip().lower() == 'fix to input':
            log.info('Fixing aperture to input position.')
            positions = self._parse_apertures(appos_input, len(self.input))
            fix_ap = True
            if str(radius).strip() == '':
                radii = None
            else:
                radii = self._parse_apertures(radius, len(self.input))
        else:
            log.info('Finding aperture positions from Gaussian fits.')
            if str(appos_input).strip() == '':
                positions = None
            else:
                positions = self._parse_apertures(appos_input, len(self.input))
            if str(radius).strip() == '':
                radii = None
            else:
                radii = self._parse_apertures(radius, len(self.input))
            fix_ap = False

        results = []
        regions = []
        for i, hdul in enumerate(self.input):
            log.info('')
            log.info(hdul[0].header['FILENAME'])

            # flux to extract
            header = hdul[0].header
            image = hdul['FLUX'].data
            var = hdul['ERROR'].data ** 2
            mask = np.full(image.shape, True)
            wave = np.arange(image.shape[1], dtype=float)
            space = np.arange(image.shape[0], dtype=float)
            spatmap = {1: hdul['SPATIAL_MAP'].data}

            # guess line position from initial line ID
            lines = self._parse_apertures(
                hdul[0].header['LINEWAV'], 1)[0]
            guesses = self._parse_apertures(
                hdul[0].header['LINEPOS'], 1)[0]
            line_type = header.get('LINETYPE', 'emission')
            sigma = header.get('LINEWID', 5.0)

            if line_type == 'emission':
                baseline = _min_func
            elif line_type == 'absorption':
                baseline = _max_func
            else:
                baseline = medabs_baseline

            appos = []
            aprad = []
            if 'step' not in method.lower():
                profile = {1: [hdul['SLITPOS'].data,
                               hdul['SPATIAL_PROFILE'].data]}
                if positions is not None:
                    guess = {1: positions[i]}
                else:
                    guess = None

                ap = find_apertures(profile, npeaks=num_aps,
                                    positions=guess, fwhm=3.0,
                                    fix=fix_ap)

                # update radii from input, accounting for overlap
                refit = fix_ap
                if radii is not None:
                    for j, aperture in enumerate(ap[1]):
                        k = j if j < len(radii[i]) else len(radii[i]) - 1
                        aperture['aperture_radius'] = radii[i][k]
                        aperture['psf_radius'] = radii[i][k]
                    refit = False
                updated_ap = get_apertures(profile, ap, get_bg=False,
                                           refit_fwhm=refit)[1]['apertures']

                log.info('')
                log.info('Found apertures:')
                for j, aperture in enumerate(ap[1]):
                    aperture['psf_radius'] = updated_ap[j]['psf_radius']
                    aperture['trace'] = np.full(len(wave),
                                                aperture['position'])
                    appos.append(aperture['position'])
                    aprad.append(aperture['psf_radius'])
                    log.info('  position: {}, '
                             'radius {}'.format(aperture['position'],
                                                aperture['psf_radius']))
                log.info('')

                # make aperture mask from defined apertures, no background
                apmask = mkapmask(space, wave, ap[1])

                # extract all spectra at once
                rectimg = {1: {'image': image, 'variance': var, 'mask': mask,
                               'wave': wave, 'spatial': space,
                               'header': header,
                               'apmask': apmask, 'apsign': None}}
                spectra = extspec(rectimg, spatial_map=spatmap,
                                  optimal=False, fix_bad=True,
                                  sub_background=False)[1]
            else:
                # step up slit, setting apertures, extracting one at a time
                # This allows overlapping radii, which may be useful
                # for higher s/n spectra
                ny = image.shape[0]
                apctr = step // 2
                apstart = apctr - radius
                apend = apctr + radius
                spectra = []

                log.info('')
                log.info('Extracting apertures:')

                while apstart < ny:
                    apmask = np.full(image.shape, 0.0)
                    apstart = 0 if apstart < 0 else apstart
                    apend = ny if apend > ny else apend
                    apmask[apstart:apend, :] = 1.0

                    # check for mostly nans in aperture: skip aperture if so
                    apdata = image[apmask == 1]
                    if np.sum(np.isnan(apdata)) < 0.1 * apdata.size:

                        # effective aperture center
                        eff = (apend - apstart) / 2
                        appos.append(apstart + eff)
                        aprad.append(eff)
                        log.info('  position: {}, '
                                 'radius {}'.format(apstart + eff, eff))

                        rectimg = {1: {'image': image.copy(),
                                       'variance': var.copy(),
                                       'mask': mask.copy(), 'wave': wave,
                                       'spatial': space, 'header': header,
                                       'apmask': apmask, 'apsign': None}}

                        one_spec = extspec(rectimg, spatial_map=spatmap,
                                           optimal=False, fix_bad=True,
                                           sub_background=False)[1]
                        spectra.append(one_spec[0])

                    apctr += step
                    apstart = apctr - radius
                    apend = apctr + radius

                spectra = np.array(spectra)
                log.info('')

            # flatten spectra if desired
            if detrend_order is not None:
                for j, spec in enumerate(spectra):
                    specflux = spec[1, :].ravel()
                    medval = float(np.nanmedian(specflux))
                    pixpos = np.arange(specflux.size, dtype=float)
                    trend = polyfitnd(pixpos, specflux, detrend_order,
                                      model=True, robust=2)
                    flatspec = specflux - trend(pixpos) + medval
                    spectra[j, 1, :] = flatspec

            # guess position of each line in each spectrum
            allpos = []
            allheight = []
            for spec in spectra:
                fitpos = []
                fitheight = []
                pixelpos = spec[0]
                spectrum = spec[1]
                specerr = spec[2]

                for line, guess in zip(lines, guesses):
                    start = int(np.round(guess - window / 2))
                    start = 0 if start < 0 else start
                    end = int(np.round(guess + window / 2))
                    end = len(pixelpos) if end > len(pixelpos) else end
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        s2n = np.nanmean(spectrum[start:end]
                                         / specerr[start:end])
                    if s2n_req > 0 and s2n < s2n_req:
                        fitpos.append(np.nan)
                        fitheight.append(np.nan)
                        continue

                    try:
                        fit_peak = fitpeaks1d(
                            pixelpos[start:end], spectrum[start:end], npeaks=1,
                            guess=guess, stddev=sigma, box_width=('stddev', 3),
                            baseline_func=baseline)
                    except ValueError:
                        fitpos.append(np.nan)
                        fitheight.append(np.nan)
                    else:
                        mval = fit_peak[0].mean.value
                        failure = (fit_peak.fit_info['ierr']
                                   not in [1, 2, 3, 4])
                        if (failure or np.allclose(mval, pixelpos[start])
                                or np.allclose(mval, pixelpos[end - 1])):
                            # set failures or boundary-pegged values to NaN
                            fitpos.append(np.nan)
                            fitheight.append(np.nan)
                        else:
                            # otherwise record fit value
                            fitpos.append(mval)
                            if line_type == 'emission':
                                height = fit_peak[0].amplitude.value
                            else:
                                height = np.abs(fit_peak[0].amplitude.value)
                            fitheight.append(height)

                allpos.append(fitpos)
                allheight.append(fitheight)

            # make position table and do preliminary fit
            allpos = np.array(allpos)
            allheight = np.array(allheight)
            trace_x = allpos.T
            trace_y = np.tile(np.expand_dims(appos, 1),
                              (1, len(lines))).T
            trace_fit = []
            for j, line in enumerate(lines):
                lfit_model = polyfitnd(trace_y[j], trace_x[j], 2,
                                       robust=5.0, model=True)
                lfit = lfit_model(space)
                trace_fit.append(lfit)

            # make a region to display
            log.info('')
            log.info('Region file shows 2nd order 1D fits '
                     'to wavelength positions for reference.')
            prodname = self.prodnames[self.prodtypes[self.step_index]]
            region = self._trace_region(
                header, self.filenum[i], prodname,
                trace_x.ravel(), trace_y.ravel(), space, trace_fit,
                fit_direction='y')
            regions.append(region)

            # record data
            hdul.append(fits.ImageHDU(allpos, name='LINE_TABLE'))
            hdul.append(fits.ImageHDU(allheight, name='LINE_HEIGHT'))
            hdinsert(hdul[0].header, 'APPOSO01',
                     ','.join(['{:.3f}'.format(a) for a in appos]),
                     'Aperture positions [pixel]')
            hdinsert(hdul[0].header, 'PSFRAD01',
                     ','.join(['{:.3f}'.format(a) for a in aprad]),
                     'Aperture PSF radii [pixel]')

            # update spectral flux and error in output file:
            # shape is n_ap x n_wave
            hdul['SPECTRAL_FLUX'].data = spectra[:, 1, :]
            hdul['SPECTRAL_ERROR'].data = spectra[:, 2, :]

            # update output name
            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                self.write_output(hdul, outname)

            if param.get_value('save_1d'):
                log.info('')
                log.info('Saving 1D spectra:')
                spec = self._make_1d(hdul)
                specname = outname.replace('LRD', 'LRS')
                spec[0].header['FILENAME'] = os.path.basename(specname)
                spec[0].header['PRODTYPE'] = 'spectra_1d'
                self.write_output(spec, specname)

            results.append(hdul)

        log.info('')
        self.input = results
        self.set_display_data(regions=regions)

    def _save_residual_plot(self, filenames):
        """Save diagnostic plots for wavecal fit."""

        # thread-safe imports to use in place of pyplot
        from matplotlib.backends.backend_agg \
            import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure

        for i, filename in enumerate(filenames):
            fig = Figure()
            FigureCanvas(fig)
            for j in range(2):
                disp = self.display_data['ResidualViewer'][2 * i + j]
                ax = fig.add_subplot(2, 1, j + 1)
                ax.plot(*disp['args'], **disp['plot_kwargs'])
                ax.set(**disp['kwargs'])
                for oplot in disp['overplot']:
                    if oplot['plot_type'] == 'vline':
                        ax.axvline(*oplot['args'], **oplot['kwargs'])
                    elif oplot['plot_type'] == 'hline':
                        ax.axhline(*oplot['args'], **oplot['kwargs'])

            fig.tight_layout()
            fig.savefig(filename, dpi=300)
            log.info('Wrote residual plot {}'.format(filename))

        return

    def _sim_spatcal(self, data_shape):
        flatfile = self.calres.get('maskfile', 'UNKNOWN')
        log.debug(f'Using order mask {flatfile}')
        if not os.path.isfile(flatfile):
            msg = 'Missing order mask'
            log.error(msg)
            raise ValueError(msg)

        flat = readflat(flatfile)
        _, spatcal, _ = simwavecal2d(data_shape, flat['edgecoeffs'],
                                     flat['xranges'], flat['slith_arc'],
                                     flat['ds'])
        return spatcal

    def fit_lines(self):
        """Fit a 2D surface to line IDs."""

        param = self.get_parameter_set()
        x_order = param.get_value('x_fit_order')
        y_order = param.get_value('y_fit_order')
        weighted = param.get_value('weighted')
        spatfile = param.get_value('spatfile')
        rotation = param.get_value('rotation')

        xpos, ypos, expected, height = [], [], [], []
        data_shape = None
        all_lines = []
        hdr_list = []
        for i, hdul in enumerate(self.input):
            if data_shape is None:
                data_shape = hdul['FLUX'].data.shape

            # lines from previous step
            header = hdul[0].header
            hdr_list.append(header)
            lines = self._parse_apertures(header['LINEWAV'], 1)[0]
            appos = self._parse_apertures(header['APPOSO01'], 1)[0]
            all_lines.extend(lines)

            line_table = hdul['LINE_TABLE'].data
            line_height = hdul['LINE_HEIGHT'].data
            pos_table = np.tile(np.expand_dims(appos, 1), (1, len(lines)))
            wave_expected = np.tile(np.expand_dims(lines, 0), (len(appos), 1))

            xpos.extend(line_table.ravel())
            ypos.extend(pos_table.ravel())
            expected.extend(wave_expected.ravel())
            height.extend(line_height.ravel())

        # 2D surface fit to lines
        if weighted:
            log.info('Weighting fit by line height.')
            error = 1 / np.array(height)
        else:
            log.info('Fit is unweighted.')
            error = None
        lfit_model = polyfitnd(ypos, xpos, expected,
                               [y_order, x_order], error=error,
                               robust=5.0, model=True)
        log.info(lfit_model)

        idx = np.arange(data_shape[0], dtype=float)
        space = np.tile(np.expand_dims(idx, 1), (1, data_shape[1]))
        idx = np.arange(data_shape[1], dtype=float)
        wave = np.tile(np.expand_dims(idx, 0), (data_shape[0], 1))
        lfit = lfit_model(space, wave)

        # spatial calibration from input or pixel positions
        if os.path.isfile(spatfile):
            log.info(f'Using {spatfile} for spatial calibration.')
            _, spatcal = readwavecal(spatfile, rotate=rotation)
            spat_hdr = getheader(spatfile)
        else:
            log.info('Using simulated calibration from slit height.')
            spatcal = self._sim_spatcal(data_shape)
            spat_hdr = fits.Header()

        # match wavecal nans to spatcal
        lfit[np.isnan(spatcal)] = np.nan

        results = []
        regions = []
        residuals = []
        save_names = []
        for i, hdul in enumerate(self.input):

            # record data
            del hdul[0].header['APPOSO01']
            hdul.append(fits.ImageHDU(lfit, name='WAVECAL'))
            hdul.append(fits.ImageHDU(spatcal, name='SPATCAL'))

            # update output name
            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                self.write_output(hdul, outname)

            results.append(hdul)

        # also write final wavecal file
        header = hdmerge(hdr_list, hdr_list[0])

        # rotate wavecal if needed before saving
        hdinsert(header, 'ROTATION', rotation, 'Rotate 90deg value')
        rw = unrotate90(lfit.copy(), rotation)
        rs = unrotate90(spatcal.copy(), rotation)

        calfile = fits.HDUList(
            fits.PrimaryHDU(header=header,
                            data=np.array([rw, rs])))
        hdinsert(calfile[0].header, 'WCTYPE', '2D',
                 comment='Wavelength calibration type')
        hdinsert(calfile[0].header, 'WXDEG', x_order,
                 comment='X polynomial degree for 2D wavecal')
        hdinsert(calfile[0].header, 'WYDEG', y_order,
                 comment='Y polynomial degree for 2D wavecal')
        hdinsert(calfile[0].header, 'WCOEFF',
                 ','.join(str(c) for c in lfit_model.coefficients),
                 comment='Wavelength fit coefficients')
        hdinsert(calfile[0].header, 'SXDEG', spat_hdr.get('SXDEG', 0),
                 comment='X polynomial degree for 2D spatcal')
        hdinsert(calfile[0].header, 'SYDEG', spat_hdr.get('SYDEG', 1),
                 comment='Y polynomial degree for 2D spatcal')
        hdinsert(calfile[0].header, 'SCOEFF', spat_hdr.get('SCOEFF', ''),
                 comment='Spatial fit coefficients')
        hdinsert(calfile[0].header, 'NORDERS', 1,
                 comment='Number of orders')
        hdinsert(calfile[0].header, 'ORDERS', '1',
                 comment='Order numbers')
        outname = self.getfilename(header, update=True,
                                   prodtype='WCL', filenum=self.filenum)
        calfile[0].header['FILENAME'] = os.path.basename(outname)
        calfile[0].header['PRODTYPE'] = 'wavecal'
        self.write_output(calfile, outname)

        # make a region file to display
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            min_line = np.max(np.nanmin(lfit, axis=1))
            max_line = np.min(np.nanmax(lfit, axis=1))
        tlines = [min_line] + sorted(list(set(all_lines))) + [max_line]
        trace_fit = []
        for ap in idx:
            aptab = []
            for wline in tlines:
                aptab.append(tabinv(lfit[int(ap), :], wline,
                                    missing=np.nan))
            trace_fit.append(aptab)
        trace_fit = np.array(trace_fit).T

        region = self._trace_region(header, self.filenum, 'LFT',
                                    xpos, ypos, idx, trace_fit,
                                    fit_direction='y')
        regions.append(region)

        # keep residuals for plotting
        dw = np.nanmean(lfit[:, 1:] - lfit[:, :-1])
        residuals_data = [xpos, ypos,
                          lfit_model.stats.residuals / dw]
        residuals.append(residuals_data)
        pngname = outname.replace('WCL', 'RSD')
        pngname = os.path.join(self.output_directory,
                               os.path.splitext(pngname)[0] + '.png')
        save_names.append(pngname)

        self.input = results
        self.set_display_data(regions=regions, residuals=residuals)

        # save residual plot to disk, after assembled in display data
        self._save_residual_plot(save_names)

        log.info('')

    def rectify(self):
        # modify input to expected extensions only -
        # FLUX, ERROR, WAVECAL, SPATCAL
        expected = ['FLUX', 'ERROR', 'WAVECAL', 'SPATCAL']
        for i, hdul in enumerate(self.input):
            new_hdul = fits.HDUList()
            for extname in expected:
                new_hdul.append(hdul[extname].copy())

            # remove old extraction information
            for key in ['APPOSO01', 'APRADO01', 'PSFRAD01']:
                try:
                    del new_hdul[0].header[key]
                except KeyError:
                    pass

            self.input[i] = new_hdul

        # call the standard make_profiles step, with the testwavecal
        # hidden parameter set
        self.make_profiles()
