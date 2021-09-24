# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FLITECAM Spectroscopy Reduction pipeline steps"""

import os

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.pipeline.sofia.sofia_exception import SOFIAImportError
try:
    import sofia_redux.instruments.flitecam
    assert sofia_redux.instruments.flitecam
except ImportError:
    raise SOFIAImportError('FLITECAM modules not installed')

from sofia_redux.pipeline.reduction import Reduction
from sofia_redux.pipeline.sofia.flitecam_reduction \
    import FLITECAMReduction
from sofia_redux.pipeline.sofia.parameters.flitecam_spectroscopy_parameters \
    import FLITECAMSpectroscopyParameters
from sofia_redux.pipeline.sofia.forcast_spectroscopy_reduction \
    import FORCASTSpectroscopyReduction
from sofia_redux.toolkit.utilities.fits import hdinsert, set_log_level


class FLITECAMSpectroscopyReduction(FLITECAMReduction,
                                    FORCASTSpectroscopyReduction):
    """
    FLITECAM spectroscopy reduction steps.

    Primary image reduction algorithms are defined in the flitecam
    package (`sofia_redux.instruments.flitecam`).  Spectroscopy-related
    algorithms are pulled from the `sofia_redux.spectroscopy` package, and
    some utilities come from the `sofia_redux.toolkit` package.  This
    reduction object requires that all three packages be installed.

    This reduction object defines a method for each pipeline
    step, that calls the appropriate algorithm from its source
    packages.
    """
    def __init__(self):
        """Initialize the reduction object."""
        FLITECAMReduction.__init__(self)

        # descriptive attributes
        self.mode = 'spectroscopy'

        # product type definitions for FLITECAM spectroscopy steps
        self.prodtype_map.update(
            {'make_image': 'spectral_image',
             'stack_dithers': 'dithers_stacked',
             'make_profiles': 'rectified_image',
             'locate_apertures': 'apertures_located',
             'trace_continuum': 'continuum_traced',
             'set_apertures': 'apertures_set',
             'subtract_background': 'background_subtracted',
             'extract_spectra': 'spectra',
             'flux_calibrate': 'calibrated_spectrum',
             'combine_spectra': 'coadded_spectrum',
             'make_response': 'response_spectrum',
             'combine_response': 'instrument_response',
             'spectral_cube': 'spectral_cube',
             'combined_spectrum': 'combined_spectrum',
             'specmap': 'specmap'})
        self.prodnames.update(
            {'spectral_image': 'IMG',
             'dithers_stacked': 'SKD',
             'rectified_image': 'RIM',
             'apertures_located': 'LOC',
             'continuum_traced': 'TRC',
             'apertures_set': 'APS',
             'background_subtracted': 'BGS',
             'spectra': 'SPM',
             'calibrated_spectrum': 'CRM',
             'coadded_spectrum': 'COA',
             'response_spectrum': 'RSP',
             'instrument_response': 'IRS',
             'spectral_cube': 'SCB',
             'combined_spectrum': 'CMB',
             'specmap': 'SMP'})

        # invert the map for quick lookup of step from type
        self.step_map = {v: k for k, v in self.prodtype_map.items()}

        # default recipe and step names
        self.recipe = ['check_header', 'correct_linearity', 'make_image',
                       'stack_dithers', 'make_profiles', 'locate_apertures',
                       'trace_continuum', 'set_apertures',
                       'subtract_background', 'extract_spectra',
                       'flux_calibrate', 'combine_spectra', 'specmap']
        self.processing_steps.update(
            {'make_image': 'Make Spectral Image',
             'stack_dithers': 'Stack Dithers',
             'make_profiles': 'Make Profiles',
             'locate_apertures': 'Locate Apertures',
             'trace_continuum': 'Trace Continuum',
             'set_apertures': 'Set Apertures',
             'subtract_background': 'Subtract Background',
             'extract_spectra': 'Extract Spectra',
             'flux_calibrate': 'Calibrate Flux',
             'combine_spectra': 'Combine Spectra',
             'make_response': 'Make Response',
             'combine_response': 'Combine Response',
             'specmap': 'Make Spectral Map'})

        # also define an alternate recipe for flux standards
        self.default_recipe = self.recipe.copy()
        self.response_recipe = self.recipe[:-1].copy()
        self.response_recipe[-1] = 'make_response'
        self.response_recipe += ['combine_response']
        # and for 1D spectra (specmap only)
        self.cmb_recipe = ['combined_spectrum', 'specmap']

    def load(self, data, param_class=None):
        """
        Load input data to make it available to reduction steps.

        The process is:

        - Call the parent load method to initialize data
          reduction variables.
        - Use the first loaded FITS header to determine and load
          the configuration.
        - Use the loaded configuration and the product type in the
          base header to determine the data processing recipe.
        - Load parameters for all steps.
        - Load the data immediately if starting from an intermediate
          step; otherwise, just load the raw headers and defer loading
          the data from the FITS files.

        After this step, the input attribute is populated as required
        for the first pipeline step in the recipe.

        Parameters
        ----------
        data : list of str or str
            Input file paths to load.
        param_class : class, optional
            Parameters to instantiate, if not FLITECAMParameters.
        """
        # imports for this step
        from sofia_redux.instruments.flitecam.getcalpath import getcalpath
        from sofia_redux.toolkit.utilities.fits import getheader

        # call the parent method to initialize
        # reduction variables
        Reduction.load(self, data)

        # read the first FITS header and use to configure reduction
        self.basehead = getheader(self.raw_files[0])
        self.calres = getcalpath(self.basehead)

        log.debug('Full configuration:')
        for key, value in self.calres.items():
            log.debug('  {}: {}'.format(key, value))

        # get product type to determine recipe
        intermediate = False
        prodtype = self.basehead.get('PRODTYPE', default='UNKNOWN')

        # check for an off-nominal recipe first
        if str(self.calres['obstype']).lower() == 'standard_telluric':
            # flux standard -- make response
            log.info('Standard detected; using alternate recipe')
            self.recipe = self.response_recipe
        elif prodtype.lower() in self.spec1d_prodtype:
            self.recipe = self.cmb_recipe
            prodtype = 'combined_spectrum'
        else:
            self.recipe = self.default_recipe

        # get remaining recipe for input prodtype
        if prodtype in self.prodtypes:
            pidx = self.prodtypes.index(prodtype)
            self.recipe = self.recipe[pidx + 1:]
            if len(self.recipe) == 0:
                msg = f"No steps to run for prodtype '{prodtype}'."
                log.error(msg)
                raise ValueError(msg)

        if str(prodtype).strip().upper() != 'UNKNOWN':
            intermediate = True

        if param_class is None:
            self.parameters = FLITECAMSpectroscopyParameters(
                config=self.calres,
                pipecal_config=self.cal_conf)
        else:  # pragma: no cover
            # this option is not currently used
            self.parameters = param_class(
                config=self.calres,
                pipecal_config=self.cal_conf)

        # if not starting from raw data, load the files in
        # immediately
        if intermediate:
            self.load_fits(intermediate=True)
        else:
            # just load headers
            self.input = []
            for datafile in self.raw_files:
                self.input.append(getheader(datafile))

    def make_image(self):
        """Pair subtract spectral images"""
        from sofia_redux.instruments.flitecam.mkspecimg import mkspecimg

        # get parameters
        param = self.get_parameter_set()
        flatfile = param.get_value('flatfile')
        pair_sub = param.get_value('pair_sub')

        if not os.path.isfile(flatfile):
            if flatfile.strip() == '':
                flatfile = None
            else:
                msg = f'Cannot find flat file: {flatfile}.'
                log.error(msg)
                raise ValueError(msg)

        # pair-subtract input
        outimg, outnum = mkspecimg(self.input, flatfile=flatfile,
                                   pair_subtract=pair_sub,
                                   filenum=self.filenum)
        self.filenum = outnum

        results = []
        for i, hdul in enumerate(outimg):
            outname = self.update_output(
                hdul, self.filenum[i], self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                self.write_output(hdul, outname)
            results.append(hdul)

        self.input = results
        self.set_display_data()

    def flux_calibrate(self):
        """Calibrate spectral flux."""
        from sofia_redux.instruments.forcast.getatran import get_atran
        from sofia_redux.spectroscopy.fluxcal import fluxcal
        from sofia_redux.spectroscopy.radvel import radvel

        # get parameters
        param = self.get_parameter_set()
        skipcal = param.get_value('skip_cal')
        respfile = param.get_value('respfile')
        resolution = param.get_value('resolution')
        atrandir = param.get_value('atrandir')
        snthresh = param.get_value('sn_threshold')
        atranfile = param.get_value('atranfile')
        auto_shift = param.get_value('auto_shift')
        shift_limit = param.get_value('auto_shift_limit')
        waveshift = param.get_value('waveshift')
        model_order = param.get_value('model_order')

        if skipcal:
            log.info('No flux calibration performed.')
            return

        # get atran parameters
        if str(atranfile).strip() == '':
            atranfile = None
        if str(atrandir).strip() == '':
            atrandir = None
        else:
            # expand environment variables in path
            atrandir = os.path.expandvars(atrandir)
            if not os.path.exists(atrandir):
                atrandir = None
        default_atrandir = os.path.join(self.calres['pathcal'],
                                        'grism', 'atran')
        if atrandir is None:
            atrandir = default_atrandir

        # set waveshift to None if not provided
        if str(waveshift).strip() == '' or np.allclose(waveshift, 0):
            waveshift = None
        else:
            if auto_shift:
                log.info('Disabling auto-shift since manual shift '
                         'was specified.')
                auto_shift = False

        # get response data
        syserr = None
        resname = 'UNKNOWN'
        response = None
        n_ord = 1
        try:
            with set_log_level('ERROR'):
                resp_hdul = fits.open(respfile)
            rhead = resp_hdul[0].header
            response_data = resp_hdul[0].data
            resp_hdul.close()
        except (OSError, ValueError, IndexError, TypeError):
            if not param.get_value('making_response'):
                msg = 'Bad response file: {}'.format(respfile)
                log.error(msg)
                raise ValueError(msg) from None
        else:
            syserr = rhead.get('SYSERR', 0.0)
            resname = respfile.split(self.calres['pathcal'])[-1]

            # handle multiple apertures in the response file as
            # orders, so they can be applied to apertures separately
            if response_data.ndim > 2:
                n_ord = response_data.shape[0]
            response = {}
            for ord in range(n_ord):
                response[ord + 1] = {'wave': response_data[ord, 0, :],
                                     'response': response_data[ord, 1, :],
                                     'error': response_data[ord, 2, :]}

            log.info('Using response file: {}'.format(respfile))

        valid = True
        try:
            resolution = float(resolution)
        except ValueError:
            valid = False
        if not valid or np.allclose(resolution, -9999):
            msg = 'Missing spectral resolution ' \
                  'for {}'.format(self.calres['name'])
            log.error(msg)
            raise ValueError(msg)
        log.info('Using resolution: {}'.format(resolution))

        results = []
        for i, hdul in enumerate(self.input):
            log.info('')
            log.info(hdul[0].header['FILENAME'])

            # retrieve data from input
            header = hdul[0].header
            image = hdul['FLUX'].data
            err = hdul['ERROR'].data
            wave = hdul['WAVEPOS'].data
            spec_flux = hdul['SPECTRAL_FLUX'].data
            spec_err = hdul['SPECTRAL_ERROR'].data
            apmask = np.abs(hdul['APERTURE_MASK'].data)

            # check s/n for auto shift
            test_auto = auto_shift
            if test_auto:
                s2n = np.nanmedian(spec_flux / spec_err)
                if s2n < snthresh:
                    log.warning(f'S/N {s2n:.1f} too low to auto-shift '
                                f'wavelengths. Disabling auto-shift.')
                    test_auto = False

            base_atran = get_atran(header, resolution, filename=atranfile,
                                   wmin=1, wmax=6, atran_dir=atrandir)
            if base_atran is None:
                if atrandir == default_atrandir:
                    msg = 'No matching ATRAN files.'
                    raise ValueError(msg)
                else:
                    # if not found, try the default directory
                    base_atran = get_atran(header, resolution,
                                           filename=atranfile,
                                           wmin=1, wmax=6,
                                           atran_dir=default_atrandir)
                    if base_atran is None:
                        msg = 'No matching ATRAN files.'
                        raise ValueError(msg)

            atran = [base_atran]

            n_ap = 1
            if spec_flux.ndim > 1:
                n_ap = spec_flux.shape[0]

            spectra = {}
            if n_ord == 1:
                spectra[1] = []
            for ap in range(n_ap):
                spec = {'wave': wave, 'flux': image.copy(),
                        'error': err.copy(), 'spectral_flux': spec_flux[ap],
                        'spectral_error': spec_err[ap],
                        'wave_shift': waveshift}
                if n_ord > 1:
                    # handle apertures as orders if the response file
                    # had multiple apertures

                    # get calibration regions from the aperture mask
                    zap = (apmask > ap) & (apmask <= (ap + 1))
                    if ap == 0:
                        # calibrate unspecified and background regions
                        # with first aperture response
                        zap |= np.isnan(apmask)
                        zap |= (apmask == 0)
                    spec['flux'][~zap] = np.nan
                    spec['error'][~zap] = np.nan

                    spectra[ap + 1] = [spec]
                else:
                    # otherwise handle apertures as spectra with
                    # a common response function
                    spectra[1].append(spec)

            result = fluxcal(spectra, atran, response,
                             auto_shift=test_auto,
                             shift_limit=shift_limit,
                             model_order=model_order)
            if result is None:
                msg = 'Problem in flux calibration.'
                log.error(msg)
                raise ValueError(msg)

            wave_shift = []
            calflux = np.full_like(image, np.nan)
            calerror = np.full_like(image, np.nan)
            specflux = []
            specerr = []
            spectrans = []
            specresp = []
            specresperr = []
            for ord in result:
                spectra = result[ord]
                for j, spec in enumerate(spectra):
                    # waveshift information
                    wave_shift.append(str(spec['wave_shift']))
                    log.info(f"Wavelength shift applied to "
                             f"spectrum {ord + j}: "
                             f"{spec['wave_shift']:.2f} pixels")

                    # compose calibrated image
                    calflux = np.nansum([calflux, spec['flux']], axis=0)
                    calerror = np.nansum([calerror, spec['error']], axis=0)

                    # gather calibrated spectra
                    specflux.append(spec['spectral_flux'])
                    specerr.append(spec['spectral_error'])
                    spectrans.append(spec['transmission'])
                    specresp.append(spec['response'])
                    specresperr.append(spec['response_error'])

            # reduce dimensions if only one spectrum
            if len(specflux) == 1:
                specflux = specflux[0]
                specerr = specerr[0]
                spectrans = spectrans[0]
                specresp = specresp[0]
                specresperr = specresperr[0]

            # update average calibration error in header
            if syserr is not None:
                calerr = syserr / np.nanmean(specresp)
                hdinsert(header, 'CALERR', calerr,
                         comment='Fractional flux calibration error')

            # log ATRAN file and record wave shift as used
            log.info('ATRAN file used: {}'.format(header['ATRNFILE']))
            hdinsert(header, 'WAVSHIFT', ','.join(wave_shift),
                     'Wavelength shift (pix)')

            # log response file and resolution
            hdinsert(header, 'RSPNFILE', resname, 'Response file')
            hdinsert(header, 'RP', resolution, 'Resolving power')

            # update data
            hdul[0].header = header
            hdul['FLUX'].data = calflux
            hdul['ERROR'].data = calerror
            hdul['WAVEPOS'].data = wave
            hdul['SPECTRAL_FLUX'].data = np.array(specflux)
            hdul['SPECTRAL_ERROR'].data = np.array(specerr)

            # also append response and transmission curves
            exthead = hdul['SPECTRAL_FLUX'].header.copy()
            if 'TRANSMISSION' in hdul:
                del hdul['TRANSMISSION']
            hdul.append(fits.ImageHDU(spectrans, exthead,
                                      name='TRANSMISSION'))
            hdul.append(fits.ImageHDU(specresp, exthead,
                                      name='RESPONSE'))
            hdul.append(fits.ImageHDU(specresperr,
                                      name='RESPONSE_ERROR'))

            # update BUNIT for all extensions
            if response is not None:
                hdul[0].header['PROCSTAT'] = 'LEVEL_3'
            hdinsert(hdul[0].header, 'RAWUNITS', 'ct/s',
                     'Data units before calibration')
            for ext in ['FLUX', 'ERROR']:
                hdinsert(hdul[ext].header, 'BUNIT', 'Jy/pixel', 'Data units')
            for ext in ['SPECTRAL_FLUX', 'SPECTRAL_ERROR']:
                hdinsert(hdul[ext].header, 'BUNIT', 'Jy', 'Data units')
            hdinsert(hdul['TRANSMISSION'].header, 'BUNIT', '', 'Data units')
            hdinsert(hdul['RESPONSE'].header, 'BUNIT', 'ct/s/Jy',
                     'Data units')
            hdinsert(hdul['RESPONSE_ERROR'].header, 'BUNIT', 'ct/s/Jy',
                     'Data units')

            # add a barycentric and LSR velocity shift to the header
            dw_bary, dw_lsr = radvel(hdul[0].header)
            hdinsert(hdul[0].header, 'BARYSHFT', dw_bary,
                     comment='Barycentric motion dl/l shift (unapplied)')
            hdinsert(hdul[0].header, 'LSRSHFT', dw_lsr,
                     comment='Additional dl/l shift to LSR (unapplied)')

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
                specname = outname.replace('CRM', 'CAL')
                spec[0].header['FILENAME'] = os.path.basename(specname)
                spec[0].header['PRODTYPE'] = 'calibrated_spectrum_1d'
                self.write_output(spec, specname)

            results.append(hdul)

        log.info('')
        self.input = results
        self.set_display_data()
