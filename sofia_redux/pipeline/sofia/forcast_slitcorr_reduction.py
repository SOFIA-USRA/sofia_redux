# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FORCAST Grism Calibration Reduction pipeline steps"""

import os

from astropy import log
from astropy.io import fits
import numpy as np
from scipy.ndimage import uniform_filter1d

from sofia_redux.pipeline.sofia.sofia_exception import SOFIAImportError
try:
    import sofia_redux.instruments.forcast
    assert sofia_redux.instruments.forcast
except ImportError:
    raise SOFIAImportError('FORCAST modules not installed')

from sofia_redux.pipeline.gui.matplotlib_viewer import MatplotlibViewer
from sofia_redux.pipeline.gui.qad_viewer import QADViewer
from sofia_redux.pipeline.sofia.forcast_reduction import FORCASTReduction
from sofia_redux.pipeline.sofia.forcast_spectroscopy_reduction \
    import FORCASTSpectroscopyReduction
from sofia_redux.pipeline.sofia.parameters.forcast_slitcorr_parameters \
    import FORCASTSlitcorrParameters
from sofia_redux.spectroscopy.extspec import extspec
from sofia_redux.toolkit.utilities.fits import hdinsert
from sofia_redux.toolkit.fitting.polynomial import polyfitnd
from sofia_redux.toolkit.interpolate import tabinv


def _err_func(var, weights):
    var[weights < 1] = np.nan
    count = np.sum(~np.isnan(var))
    return np.sqrt(np.nansum(var)) / count


def _med_func(flux, weights):
    flux[weights < 1] = np.nan
    return np.nanmedian(flux)


class FORCASTSlitcorrReduction(FORCASTSpectroscopyReduction):
    r"""
    FORCAST spesctroscopic slit correction reduction steps.

    This reduction object defines specialized reduction steps
    for generating slit correction calibration data from spectroscopic
    input files.  It is selected by the SOFIA chooser only if a top-level
    configuration flag is supplied (slitcorr=True).  The final
    output product from this reduction is a FITS file (\*SCR\*.fits)
    with PRODTYPE = 'slit_correction'.  This file can be supplied to the
    standard spectroscopic pipeline, at the make_profiles step,
    to specify a new slit response correction.
    """
    def __init__(self):
        """Initialize the reduction object."""
        super().__init__()

        # descriptive attributes specific to calibration
        self.name = 'Slit correction'

        # product type definitions for spectral steps
        self.prodtype_map.update(
            {'rectify': 'test_rectified_image',
             'make_profiles': 'rectified_image',
             'extract_median_spectra': 'median_spectra',
             'normalize': 'normalized_image',
             'make_slitcorr': 'slit_correction'})
        self.prodnames.update(
            {'rectified_image': 'RIM',
             'median_spectra': 'MSM',
             'normalized_image': 'NIM',
             'slit_correction': 'SCR'})

        # invert the map for quick lookup of step from type
        self.step_map = {v: k for k, v in self.prodtype_map.items()}

        # default recipe and step names
        self.recipe = ['checkhead', 'clean', 'droop', 'nonlin',
                       'stack', 'stack_dithers', 'make_profiles',
                       'locate_apertures', 'extract_median_spectra',
                       'normalize', 'make_slitcorr']
        self.processing_steps.update(
            {'make_profiles': 'Make Profiles',
             'extract_median_spectra': 'Extract Median Spectra',
             'normalize': 'Normalize Response',
             'make_slitcorr': 'Make Slit Correction'})

    def load(self, data, param_class=None):
        """Call parent load, with slitcorr parameters."""
        FORCASTReduction.load(self, data,
                              param_class=FORCASTSlitcorrParameters)

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

        viewers = [QADViewer(), prof, spec]

        return viewers

    def set_display_data(self, raw=False, filenames=None, regions=None,
                         specviewer='matplotlib'):
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
                                 regions=regions, specviewer=specviewer)

    def extract_median_spectra(self):
        """Extract median spectra at regular positions."""
        param = self.get_parameter_set()

        results = []
        for i, hdul in enumerate(self.input):
            log.info('')
            log.info(hdul[0].header['FILENAME'])

            # flux to extract
            header = hdul[0].header
            image = hdul['FLUX'].data
            var = hdul['ERROR'].data ** 2
            mask = (hdul['BADMASK'].data < 1)
            wave = hdul['WAVEPOS'].data
            space = hdul['SLITPOS'].data
            profile = hdul['SPATIAL_PROFILE'].data
            spatmap = {1: hdul['SPATIAL_MAP'].data}

            appos = np.array(self._parse_apertures(header['APPOSO01'], 1)[0])
            num_aps = appos.size
            aprad = image.shape[0] // num_aps // 2

            # make aperture mask with no overlap
            apmask = np.zeros_like(image)
            test = aprad * 2
            for j in range(num_aps):
                # get the aperture position in pixels
                appix = int(np.round(tabinv(space, appos[j])))

                # check that we are not overwriting the last aperture
                if not np.all(apmask[appix - aprad:appix + aprad, :]) == 0:
                    raise ValueError('Too many apertures')

                # set the aperture
                apmask[appix - aprad:appix + aprad, :] = j + 1

                # check that the new aperture has the right size
                if (not np.sum(apmask == j + 1)
                        == wave.size * test):  # pragma: no cover
                    raise ValueError('Too many apertures')

            rectimg = {1: {'image': image.copy(), 'variance': var.copy(),
                           'mask': mask.copy(), 'wave': wave,
                           'spatial': space, 'header': header,
                           'apmask': apmask, 'apsign': None}}

            # extract all spectra at once
            spectra = extspec(rectimg,
                              profile=profile, spatial_map=spatmap,
                              optimal=False, fix_bad=True,
                              sub_background=False,
                              sum_function=_med_func,
                              error_function=_err_func)[1]

            # update flux and error planes -- they may have had
            # bad pixels corrected in the extraction process
            hdul['FLUX'].data = rectimg[1]['image']
            hdul['ERROR'].data = np.sqrt(rectimg[1]['variance'])

            # attach spectral flux and error to output file:
            # shape is n_ap x n_wave
            exthead = fits.Header()
            for key in self.wcs_keys:
                if key in header and not key.endswith('A') and '2' not in key:
                    hdinsert(exthead, key, header[key], header.comments[key])
            hdul.append(fits.ImageHDU(spectra[:, 1, :], exthead,
                                      name='SPECTRAL_FLUX'))
            hdul.append(fits.ImageHDU(spectra[:, 2, :], exthead,
                                      name='SPECTRAL_ERROR'))
            hdinsert(hdul['SPECTRAL_FLUX'].header, 'BUNIT',
                     header.get('BUNIT', 'UNKNOWN'), 'Data units')
            hdinsert(hdul['SPECTRAL_ERROR'].header, 'BUNIT',
                     header.get('BUNIT', 'UNKNOWN'), 'Data units')

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
                specname = outname.replace('MSM', 'MSP')
                spec[0].header['FILENAME'] = os.path.basename(specname)
                spec[0].header['PRODTYPE'] = 'spectra_1d'
                self.write_output(spec, specname)

            results.append(hdul)

        log.info('')
        self.input = results
        self.set_display_data()

    def normalize(self):
        """Normalize by central median spectrum."""
        param = self.get_parameter_set()

        results = []
        for i, hdul in enumerate(self.input):
            log.info('')
            log.info(hdul[0].header['FILENAME'])

            # retrieve data from input
            flux = hdul['FLUX'].data
            err = hdul['ERROR'].data
            spec_flux = hdul['SPECTRAL_FLUX'].data
            spec_err = hdul['SPECTRAL_ERROR'].data

            # normalize to center spectrum
            nspec = spec_flux.shape[0]
            reference = spec_flux[nspec // 2]
            hdul['FLUX'].data = flux / reference
            hdul['ERROR'].data = err / reference
            hdul['SPECTRAL_FLUX'].data = spec_flux / reference
            hdul['SPECTRAL_ERROR'].data = spec_err / reference

            # update bunit
            hdul['FLUX'].header['BUNIT'] = ''
            hdul['ERROR'].header['BUNIT'] = ''
            hdul['SPECTRAL_FLUX'].header['BUNIT'] = ''
            hdul['SPECTRAL_ERROR'].header['BUNIT'] = ''

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
                specname = outname.replace('NIM', 'NMS')
                spec[0].header['FILENAME'] = os.path.basename(specname)
                spec[0].header['PRODTYPE'] = 'spectra_1d'
                self.write_output(spec, specname)

            results.append(hdul)

        log.info('')
        self.input = results
        self.set_display_data()

    def make_slitcorr(self):
        """Fit and smooth normalized data."""
        param = self.get_parameter_set()
        method = param.get_value('method')
        weighted = param.get_value('weighted')
        x_order = param.get_value('x_fit_order')
        y_order = param.get_value('y_fit_order')
        y_order_1d = param.get_value('y_fit_order_1d')
        x_width = param.get_value('x_width')

        results = []
        for i, hdul in enumerate(self.input):
            log.info('')
            log.info(hdul[0].header['FILENAME'])

            # retrieve data from input
            header = hdul[0].header
            flux = hdul['FLUX'].data
            spec_flux = hdul['SPECTRAL_FLUX'].data
            spec_err = hdul['SPECTRAL_ERROR'].data
            appos = np.array(self._parse_apertures(header['APPOSO01'], 1)[0])
            wave = hdul['WAVEPOS'].data
            space = hdul['SLITPOS'].data

            # fit to normalized median spectra
            error = None
            if str(method).lower() == '1d':
                sfit = np.full_like(flux, np.nan)
                chisq = []
                for j, xval in enumerate(wave):
                    if weighted:
                        error = spec_err[:, j]
                    sfit_model = polyfitnd(appos, spec_flux[:, j],
                                           y_order_1d, error=error,
                                           robust=5.0, model=True)
                    sfit_col = sfit_model(space)
                    sfit[:, j] = sfit_col
                    chisq.append(sfit_model.stats.rchi2)

                # boxcar smooth in the x direction
                sfit = uniform_filter1d(sfit, x_width, axis=1)

                log.info(f'Mean reduced chi^2: {np.mean(chisq)}')
                log.info('')

            else:
                if weighted:
                    error = spec_err

                ypos = np.tile(np.expand_dims(appos, 1), (1, wave.size))
                xpos = np.tile(np.expand_dims(wave, 0), (appos.size, 1))
                sfit_model = polyfitnd(ypos, xpos, spec_flux,
                                       [y_order, x_order], error=error,
                                       robust=5.0, model=True)
                log.info(sfit_model)

                # expand to full array
                ypos = np.tile(np.expand_dims(space, 1), (1, wave.size))
                xpos = np.tile(np.expand_dims(wave, 0), (space.size, 1))
                sfit = sfit_model(ypos, xpos)

            # delete aperture info from header
            del header['APPOSO01']

            # save fit as slit correction image
            hdul = fits.HDUList(fits.PrimaryHDU(header=header, data=sfit))
            hdinsert(hdul[0].header, 'SCXDEG', x_order,
                     comment='X polynomial degree for slit correction')
            hdinsert(hdul[0].header, 'SCYDEG', y_order,
                     comment='Y polynomial degree for slit correction')
            hdinsert(hdul[0].header, 'SCCOEFF',
                     ','.join(str(c) for c in sfit_model.coefficients),
                     comment='Slit correction fit coeff')

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
