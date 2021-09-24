# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FORCAST Spectroscopy Reduction pipeline steps"""

import os
import re
import warnings

from astropy import log
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.io import fits
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.wcs import WCS
from matplotlib.backends.backend_agg \
    import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from sofia_redux.pipeline.sofia.sofia_exception import SOFIAImportError
try:
    import sofia_redux.instruments.forcast
    assert sofia_redux.instruments.forcast
except ImportError:
    raise SOFIAImportError('FORCAST modules not installed')

from sofia_redux.instruments.forcast.getpar import getpar
import sofia_redux.instruments.forcast.configuration as dripconfig
from sofia_redux.pipeline.gui.qad_viewer import QADViewer
from sofia_redux.pipeline.gui.matplotlib_viewer import MatplotlibViewer
from sofia_redux.pipeline.reduction import Reduction
from sofia_redux.pipeline.sofia.forcast_reduction import FORCASTReduction
from sofia_redux.pipeline.sofia.parameters.forcast_spectroscopy_parameters \
    import FORCASTSpectroscopyParameters
from sofia_redux.toolkit.utilities.fits \
    import hdinsert, getheader, gethdul, set_log_level
from sofia_redux.visualization.redux_viewer import EyeViewer

# these imports are not used here, but are needed to avoid
# a numba error on linux systems
from sofia_redux.toolkit.interpolate import interpolate
from sofia_redux.toolkit.image.combine import combine_images
from sofia_redux.toolkit.resampling import tree
assert interpolate
assert combine_images
assert tree


class FORCASTSpectroscopyReduction(FORCASTReduction):
    """
    FORCAST spectroscopy reduction steps.

    Primary image reduction algorithms are defined in the DRIP
    package (`sofia_redux.instruments.forcast`).  Spectroscopy-related
    algorithms are pulled from the `sofia_redux.spectroscopy` package,
    and some utilities come from the `sofia_redux.toolkit` package.
    This reduction object requires that all three packages be installed.

    This reduction object defines a method for each pipeline
    step that calls the appropriate algorithm from its source
    packages.

    Attributes
    ----------
    prodtype_map : dict
        Maps the pipeline step to a product type, to assign
        to the PRODTYPE key. Keys are pipeline step function names.
    prodnames : dict
        3-letter file type code, to assign to the output of a pipeline
        step. Keys are the product types (as defined in prodtype_map).
    step_map : dict
        Inverse of the prodtype_map, for looking up pipeline
        step names from product types.  Keys are the product types.
    prodtypes : list
        List of product types, corresponding to the currently
        loaded recipe.  This list is populated whenever the recipe
        attribute is set.
    default_recipe : list
        Processing recipe used for most input data.
    response_recipe : list
        Alternate processing recipe to use when input data is
        marked as a flux standard (OBSTYPE=STANDARD_TELLURIC).
        Instrumental response spectra are generated instead of a final
        calibrated, combined spectrum.
    basehead : `astropy.io.fits.header.Header`
        Header for the first raw input file loaded, used for calibration
        configuration.
    calres : dict-like
        Reduction mode and auxiliary file configuration mapping,
        as returned from the sofia_redux.instruments.forcast `getcalpath`
        function.
    wcs_keys : list
        List of header keywords used for tracking and propagating the
        spectral world coordinate system.
    """
    def __init__(self):
        """Initialize the reduction object."""
        super().__init__()

        # descriptive attributes specific to spectroscopy
        self.mode = 'Spectroscopy'

        # product type definitions for spectral steps
        self.prodtype_map.update(
            {'stack_dithers': 'dithers_stacked',
             'make_profiles': 'rectified_image',
             'locate_apertures': 'apertures_located',
             'trace_continuum': 'continuum_traced',
             'set_apertures': 'apertures_set',
             'subtract_background': 'background_subtracted',
             'extract_spectra': 'spectra',
             'merge_apertures': 'merged_spectrum',
             'flux_calibrate': 'calibrated_spectrum',
             'combine_spectra': 'coadded_spectrum',
             'make_response': 'response_spectrum',
             'combine_response': 'instrument_response',
             'spectral_cube': 'spectral_cube',
             'combined_spectrum': 'combined_spectrum',
             'specmap': 'specmap'})
        self.prodnames.update(
            {'dithers_stacked': 'SKD',
             'rectified_image': 'RIM',
             'apertures_located': 'LOC',
             'continuum_traced': 'TRC',
             'apertures_set': 'APS',
             'background_subtracted': 'BGS',
             'spectra': 'SPM',
             'merged_spectrum': 'MGM',
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
        self.recipe = ['checkhead', 'clean', 'droop', 'nonlin',
                       'stack', 'stack_dithers', 'make_profiles',
                       'locate_apertures', 'trace_continuum',
                       'set_apertures', 'subtract_background',
                       'extract_spectra', 'merge_apertures',
                       'flux_calibrate', 'combine_spectra', 'specmap']
        self.processing_steps.update(
            {'stack_dithers': 'Stack Dithers',
             'make_profiles': 'Make Profiles',
             'locate_apertures': 'Locate Apertures',
             'trace_continuum': 'Trace Continuum',
             'set_apertures': 'Set Apertures',
             'subtract_background': 'Subtract Background',
             'extract_spectra': 'Extract Spectra',
             'merge_apertures': 'Merge Apertures',
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

        # and for making a map from a spectral cube or combined spectrum
        self.cube_recipe = ['spectral_cube', 'specmap']
        self.cmb_recipe = ['combined_spectrum', 'specmap']

        # store some WCS keys used for tracking and propagating
        # the 2D spectral WCS
        self.wcs_keys = ['CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2',
                         'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2',
                         'CDELT1', 'CDELT2', 'CROTA2', 'SPECSYS',
                         'CTYPE1A', 'CTYPE2A', 'CTYPE3A',
                         'CUNIT1A', 'CUNIT2A', 'CUNIT3A',
                         'CRPIX1A', 'CRPIX2A', 'CRPIX3A',
                         'CRVAL1A', 'CRVAL2A', 'CRVAL3A',
                         'CDELT1A', 'CDELT2A', 'CDELT3A',
                         'PC2_2A', 'PC2_3A', 'PC3_2A', 'PC3_3A',
                         'RADESYSA', 'EQUINOXA', 'SPECSYSA']

    def load(self, data, param_class=None):
        """
        Load input data to make it available to reduction steps.

        The process is:

        - Call the parent load method to initialize data
          reduction variables.
        - Use the first loaded FITS header to determine and load
          the DRIP configuration (`sofia_redux.instruments.forcast.getcalpath`,
          `sofia_redux.instruments.forcast.configuration`).
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
            Parameters to instantiate, if not FORCASTImagingParameters.
            Initialization arguments must match.
        """
        # call the parent method to initialize
        # reduction variables
        Reduction.load(self, data)

        # import from sofia_redux.instruments.forcast, sofia_redux.calibration
        from sofia_redux.instruments.forcast.getcalpath import getcalpath

        # read and save the first FITS header
        self.basehead = getheader(self.raw_files[0])
        self.calres = getcalpath(self.basehead)

        log.debug('Full DRIP cal configuration:')
        for key, value in self.calres.items():
            log.debug('  {}: {}'.format(key, value))

        # load sofia_redux.instruments.forcast config
        dripconfig.load(self.calres['conffile'])

        # get product type to determine recipe
        intermediate = False
        prodtype = getpar(self.basehead, 'PRODTYPE', default='UNKNOWN')

        # check for an off-nominal recipe first
        if prodtype.lower() in ['spectral_cube', 'speccube']:
            self.recipe = self.cube_recipe
            prodtype = 'spectral_cube'
        elif prodtype.lower() in self.spec1d_prodtype:
            self.recipe = self.cmb_recipe
            prodtype = 'combined_spectrum'
        elif 'standard' in str(self.calres['obstype']).lower():
            # flux standard -- make response
            log.info('Standard detected; using alternate recipe')
            self.recipe = self.response_recipe
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
            intermediate = True
        elif prodtype.upper() != 'UNKNOWN':
            msg = f"Unrecognized prodtype '{prodtype}'."
            log.error(msg)
            raise ValueError(msg)

        log.debug("Processing steps: {}".format(self.recipe))

        if param_class is None:
            self.parameters = FORCASTSpectroscopyParameters(
                drip_cal_config=self.calres,
                drip_config=dripconfig.configuration)
        else:  # pragma: no cover
            # this option is not currently used
            self.parameters = param_class(
                drip_cal_config=self.calres,
                drip_config=dripconfig.configuration)

        # if not starting from raw data, load the files in
        # immediately
        if intermediate:
            self.load_fits(intermediate=True)
        else:
            # just load headers
            self.input = []
            for datafile in self.raw_files:
                self.input.append(fits.getheader(datafile))

    def register_viewers(self):
        """Return a new QADViewer, ProfileViewer, and SpectralViewer."""
        prof = MatplotlibViewer()
        prof.name = 'ProfileViewer'
        prof.title = 'Spatial Profiles'
        prof.layout = 'rows'

        spec = EyeViewer()
        spec.name = 'SpectralViewer'
        spec.title = 'Spectra'
        spec.layout = 'rows'

        viewers = [QADViewer(), prof, spec]

        return viewers

    def set_display_data(self, raw=False, filenames=None, regions=None,
                         specviewer='eye'):
        """
        Store display data for QAD and Matplotlib viewers.

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
        """
        self.display_data = {}
        if raw:
            data_list = self.raw_files
        elif filenames:
            data_list = filenames
        else:
            data_list = self.input.copy()
        self.display_data['QADViewer'] = data_list

        # set profile plot if necessary
        if not raw:
            disp_plot = []
            disp_spec = []
            for hdul in data_list:
                # get data from disk if necessary
                hdul = gethdul(hdul)
                if hdul is None:
                    continue
                title = hdul[0].header['FILENAME']
                if 'SPATIAL_PROFILE' in hdul:
                    slitpos = hdul['SLITPOS'].data
                    profile = hdul['SPATIAL_PROFILE'].data
                    yunit = hdul['SLITPOS'].header.get('BUNIT', 'arcsec')
                    disp = {'args': [slitpos, profile],
                            'kwargs': {'title': title,
                                       'xlabel': f'Slit position ({yunit})',
                                       'ylabel': 'Normalized median flux'},
                            'plot_kwargs': {'color': 'gray'}}

                    overplots = []
                    if 'APPOSO01' in hdul[0].header:
                        aps = self._parse_apertures(
                            hdul[0].header['APPOSO01'], 1)[0]
                        if 'APRADO01' in hdul[0].header:
                            rads = self._parse_apertures(
                                hdul[0].header['APRADO01'], 1)[0]
                        else:
                            rads = [None] * len(aps)
                        if 'PSFRAD01' in hdul[0].header:
                            psfs = self._parse_apertures(
                                hdul[0].header['PSFRAD01'], 1)[0]
                        else:
                            psfs = [None] * len(aps)
                        for a, r, p in zip(aps, rads, psfs):
                            overplots.append(
                                {'plot_type': 'vline',
                                 'args': [a],
                                 'kwargs': {'color': '#17becf',
                                            'linestyle': '--'}})
                            if r is not None:
                                overplots.append(
                                    {'plot_type': 'vline',
                                     'args': [a - r],
                                     'kwargs': {'color': '#2ca02c',
                                                'linestyle': ':'}})
                                overplots.append(
                                    {'plot_type': 'vline',
                                     'args': [a + r],
                                     'kwargs': {'color': '#2ca02c',
                                                'linestyle': ':'}})
                            if p is not None:
                                overplots.append(
                                    {'plot_type': 'vline',
                                     'args': [a - p],
                                     'kwargs': {'color': '#1f77b4',
                                                'linestyle': '-.'}})
                                overplots.append(
                                    {'plot_type': 'vline',
                                     'args': [a + p],
                                     'kwargs': {'color': '#1f77b4',
                                                'linestyle': '-.'}})
                        if 'BGR' in hdul[0].header:
                            bgrs = self._parse_bg(hdul[0].header['BGR'], 1)[0]
                            for reg in bgrs:
                                if len(reg) == 2:
                                    idx = (slitpos >= reg[0]) \
                                        & (slitpos <= reg[1])
                                    overplots.append(
                                        {'args': [slitpos[idx],
                                                  profile[idx]],
                                         'kwargs': {'color': '#d62728'}})
                    if overplots:
                        disp['overplot'] = overplots
                    disp_plot.append(disp)

                if 'SPECTRAL_FLUX' in hdul:
                    wv = hdul['WAVEPOS'].data
                    wunit = hdul['WAVEPOS'].header.get('BUNIT', 'um')
                    sf = hdul['SPECTRAL_FLUX'].data
                    bunit = hdul['SPECTRAL_FLUX'].header.get('BUNIT',
                                                             'UNKNOWN')
                    disp = {'args': [wv, sf.T],
                            'kwargs': {'title': title,
                                       'xlabel': f'Wavelength ({wunit})',
                                       'ylabel': f'Flux ({bunit})'},
                            'plot_kwargs': {}}

                    overplots = []
                    if 'DISPWAV' in hdul[0].header:
                        lps = self._parse_apertures(
                            hdul[0].header['DISPWAV'], 1)[0]
                        overplots = []
                        for lp in lps:
                            overplots.append(
                                {'plot_type': 'vline',
                                 'args': [lp],
                                 'kwargs': {'color': 'lightgray',
                                            'linestyle': ':'}})
                    if 'LINEWAV' in hdul[0].header:
                        lps = self._parse_apertures(
                            hdul[0].header['LINEWAV'], 1)[0]
                        for lp in lps:
                            overplots.append(
                                {'plot_type': 'vline',
                                 'args': [lp],
                                 'kwargs': {'color': '#17becf',
                                            'linestyle': '--'}})
                    if overplots:
                        disp['overplot'] = overplots

                    disp_spec.append(disp)

            self.display_data['ProfileViewer'] = disp_plot
            if disp_spec:
                if specviewer == 'eye':
                    self.display_data['SpectralViewer'] = data_list
                else:
                    self.display_data['SpectralViewer'] = disp_spec

        # set regions if necessary
        if regions is not None:
            self.display_data['QADViewer'].extend(regions)

    def stack_dithers(self):
        """
        Combine images at common dither positions.

        Calls `sofia_redux.toolkit.image.combine.combine_images` for
        image coaddition.

        The combination method may be configured in parameters,
        or skipped entirely.
        """
        from sofia_redux.instruments.forcast.hdmerge import hdmerge
        from sofia_redux.toolkit.image.combine import combine_images

        # get parameters
        param = self.get_parameter_set()
        do_stack = not param.get_value('skip_stack')
        ignore_dither = param.get_value('ignore_dither')

        # check if stacking should be performed
        if not do_stack or len(self.input) == 1:
            log.info('No stacking performed.')
            return

        method = param.get_value('method')
        weighted = param.get_value('weighted')
        robust = param.get_value('robust')
        sigma = param.get_value('threshold')
        maxiters = param.get_value('maxiters')

        hdr_list = []
        data_list = []
        var_list = []
        dithers = []
        for hdul in self.input:
            hdr_list.append(hdul[0].header)
            data_list.append(hdul[0].data)
            var_list.append(hdul[1].data ** 2)
            dthidx = getpar(hdul[0].header, 'DTHINDEX', default=0)
            dithers.append(dthidx)
        data_list = np.array(data_list)
        var_list = np.array(var_list)
        dithers = np.array(dithers)

        if ignore_dither:
            log.info('Ignoring dither information; stacking all input.')
            dithers[:] = 0

        results = []
        filenums = []
        combined = False
        for dthval in sorted(np.unique(dithers)):
            idx = np.where(dithers == dthval)[0]
            n_img = len(idx)
            if n_img == 1:
                continue

            log.info('Stacking {} dithers at index {}'.format(n_img, dthval))
            combined = True

            outdata, outvar = combine_images(data_list[idx],
                                             variance=var_list[idx],
                                             method=method, weighted=weighted,
                                             robust=robust, sigma=sigma,
                                             maxiters=maxiters)

            headers_to_merge = [hdr_list[i] for i in idx]
            outhdr = hdmerge(headers_to_merge,
                             reference_header=headers_to_merge[0])

            # add some history messages for combination parameters
            hdinsert(outhdr, 'HISTORY',
                     'Stack Dithers: Method: {}'.format(method))
            if method == 'mean':
                hdinsert(outhdr, 'HISTORY',
                         'Stack Dithers: Weighted: {}'.format(weighted))
                hdinsert(outhdr, 'HISTORY',
                         'Stack Dithers: Robust: {}'.format(robust))
            if robust:
                hdinsert(outhdr, 'HISTORY',
                         'Stack Dithers: Threshold: {}'.format(sigma))
                hdinsert(outhdr, 'HISTORY',
                         'Stack Dithers: Max. Iters: {}'.format(maxiters))

            # store output data
            hdul = self.input[idx[0]]
            hdul[0].header = outhdr
            hdul[0].data = outdata
            hdul[1].data = np.sqrt(outvar)

            fnum = [self.filenum[i] for i in idx]
            outname = self.update_output(hdul, fnum,
                                         self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                self.write_output(hdul, outname)

            results.append(hdul)
            filenums.append(fnum)

        if not combined:
            log.info('No repeated dithers; no stacking performed.')
            return

        self.input = results
        self.filenum = filenums
        self.set_display_data()

    def make_profiles(self):
        """
        Rectify spectral images and produce spatial profile fits.

        The rectified images and profiles are stored in self.input
        for continued processing.

        If the `atmosthresh` parameters is set to a
        number > 0, then an atran file associated with the data will
        be retrieved from the FORCAST data default, and used to determine
        spatial profile regions that should be ignored for the profile
        fits.
        """
        from sofia_redux.instruments.forcast.getatran import get_atran
        from sofia_redux.spectroscopy.mkspatprof import mkspatprof
        from sofia_redux.spectroscopy.readflat import readflat
        from sofia_redux.spectroscopy.readwavecal import readwavecal
        from sofia_redux.spectroscopy.rectify import rectify
        from sofia_redux.spectroscopy.rectifyorder import update_wcs

        # get parameters
        param = self.get_parameter_set()
        fit_order = param.get_value('fit_order')
        bg_sub = param.get_value('bg_sub')
        wavefile = param.get_value('wavefile')
        slitfile = param.get_value('slitfile')
        atmosthresh = param.get_value('atmosthresh')
        simwavecal = param.get_value('simwavecal')
        testwavecal = param.get_value('testwavecal')

        # grism mode for output messages
        mode = '{} {} {}'.format(self.calres['name'], self.calres['slit'],
                                 self.calres['dateobs'])
        cnmode = str(self.calres['cnmode']).lower()

        # read order mask/flat -- unlikely to change, so leave at default
        flatfile = self.calres['maskfile']
        log.debug(f'Using order mask {flatfile}')
        if not os.path.isfile(flatfile):
            msg = 'Missing order mask for {}'.format(mode)
            log.error(msg)
            raise ValueError(msg)
        flat = readflat(flatfile)

        # read wave and spatial cal file if desired
        ybuffer = 3
        ds = flat['ds']
        omask = flat['omask']
        wavecal = None
        spatcal = None
        if not (simwavecal or testwavecal):
            if not os.path.isfile(wavefile):
                msg = f'Missing wavecal file for {mode}.'
                log.error(msg)
                raise ValueError(msg)
            log.debug(f'Using wavecal file {wavefile}.')
            wavecal, spatcal = readwavecal(wavefile, rotate=flat['rotation'])

            # if a waveshift is specified in configuration, apply
            # it directly
            if 'waveshift' in self.calres and self.calres['waveshift'] != 0:
                waveshift = self.calres['waveshift']
                log.info(f'Applying default waveshift of {waveshift} '
                         f'for {mode}')
                wavecal += waveshift
        elif simwavecal:
            # make mock calibration data if desired
            log.debug('Simulating calibration data.')
            idx = np.arange(flat['nrows'], dtype=float)
            spatcal = np.tile(np.expand_dims(idx, 1), (1, flat['ncols']))
            wavecal = spatcal.copy().transpose()
            ybuffer = 0
            ds = 1.0

        # read slit correction function
        # warn if not present, but carry on
        if not os.path.isfile(slitfile):
            if not simwavecal:
                log.warning('Missing slit correction '
                            'file for {}.'.format(mode))
                log.warning('Slit response will not be corrected.')
            slitfile = 'NONE'
            slit_fn = None
        else:
            # slit function is an image in the primary HDU
            with set_log_level('ERROR'):
                slit_hdul = fits.open(slitfile)
            log.debug(f'Using slit correction file {slitfile}.')
            slit_fn = slit_hdul[0].data

        # divide by 2 for NMC and SLITSCAN
        if 'nmc' in cnmode or 'scan' in cnmode:
            log.info('Dividing by 2 for NMC chop-nod mode.')
            cn_factor = 2.0
        else:
            cn_factor = 1.0

        # file names for storage in headers
        flatname = flatfile.split(self.calres['pathcal'])[-1]
        wavename = wavefile.split(self.calres['pathcal'])[-1]
        slitname = slitfile.split(self.calres['pathcal'])[-1]

        # loop through input, rectifying images and making profiles
        results = []
        for i, hdul in enumerate(self.input):
            header = hdul[0].header
            data = hdul[0].data
            var = hdul[1].data ** 2

            # update header with flat and wave file
            hdinsert(header, 'ORDRFILE', flatname,
                     comment='Spectral order definition')
            hdinsert(header, 'WAVEFILE', wavename,
                     comment='Spectral/spatial calibration')
            hdinsert(header, 'SLITFILE', slitname,
                     comment='Slit correction function')

            # add slit height and width
            hdinsert(header, 'SLTH_ARC', flat['slith_arc'],
                     comment='Slit height [arcsec]')
            hdinsert(header, 'SLTH_PIX', flat['slith_pix'],
                     comment='Slit height [pix]')
            hdinsert(header, 'SLTW_ARC', flat['slitw_arc'],
                     comment='Slit width [arcsec]')
            hdinsert(header, 'SLTW_PIX', flat['slitw_pix'],
                     comment='Slit width [pix]')

            wcshdr = fits.Header({'CRPIX1': header['CRPIX1'],
                                  'CRPIX2': header['CRPIX2'],
                                  'CRVAL1': header['CRVAL1'],
                                  'CRVAL2': header['CRVAL2'],
                                  'CROTA2': header['CROTA2']})
            if simwavecal:
                # skip actually rectifying,
                # but do trim to order mask
                data[omask != 1] = np.nan
                var[omask != 1] = np.nan
                mask = np.full(data.shape, True)
                mask[omask != 1] = False

                wave = np.arange(data.shape[1], dtype=float)
                space = np.arange(data.shape[0], dtype=float)
                result = {'image': data, 'variance': var, 'mask': mask,
                          'wave': wave, 'spatial': space,
                          'header': wcshdr}
                update_wcs(result, spatcal)
                rectimg = {1: result}
            else:
                if testwavecal:
                    log.info('Using wave/space calibration data in HDUList')
                    wavecal = hdul['WAVECAL'].data
                    spatcal = hdul['SPATCAL'].data

                rectimg = rectify(data, omask, wavecal,
                                  spatcal, header=wcshdr, variance=var,
                                  ybuffer=ybuffer, ds=ds, badfrac=0.2)
                if rectimg[1] is None:
                    raise ValueError('Problem in rectification.')

            # rectimg is indexed by order -- we support only one
            # order for FORCAST
            new_image = rectimg[1]['image'].copy()
            new_error = np.sqrt(rectimg[1]['variance'])

            # divide by slit function if available
            if slit_fn is not None:
                if slit_fn.shape != new_image.shape:
                    msg = f'Slit function image shape {slit_fn.shape} does ' \
                          f'not match rectified image shape {new_image.shape}.'
                    log.error(msg)
                    raise ValueError(msg)
                new_image /= slit_fn
                new_error /= slit_fn

            # divide by 2 for doubled NMC source
            new_image /= cn_factor
            new_error /= cn_factor

            # store new image
            hdul[0].data = new_image

            # update header -- delete old WCS, add new
            oldwcs = WCS(header).to_header(relax=True)
            for key in oldwcs:
                if key in header and 'date' not in key.lower() \
                        and 'equinox' not in key.lower():
                    del header[key]
            for key in wcshdr:
                if key in header and 'date' not in key.lower() \
                        and 'equinox' not in key.lower():
                    del header[key]

            for card in rectimg[1]['header'].cards:
                hdinsert(header, card.keyword, card.value, card.comment)

            hdinsert(header, 'EXTNAME', 'FLUX', 'extension name')
            hdul[0].header = header

            # update the error HDU
            extwcs = rectimg[1]['header']
            hdinsert(extwcs, 'BUNIT',
                     hdul[0].header.get('BUNIT', 'UNKNOWN'),
                     'Data units')
            hdul[1] = fits.ImageHDU(data=new_error,
                                    header=extwcs, name='ERROR')

            # append extra information from rectification
            boolmask = rectimg[1]['mask']
            mask = np.zeros(boolmask.shape, dtype=int)
            mask[~boolmask] = 1
            hdinsert(extwcs, 'BUNIT', '', 'Data units')
            hdul.append(fits.ImageHDU(data=mask,
                                      header=extwcs, name='BADMASK'))

            exthead = fits.Header()
            xunit = 'pixel' if simwavecal else 'um'
            hdinsert(exthead, 'BUNIT', xunit, 'Data units')
            hdul.append(fits.ImageHDU(data=rectimg[1]['wave'],
                                      header=exthead, name='WAVEPOS'))

            yunit = 'pixel' if simwavecal else 'arcsec'
            hdinsert(exthead, 'BUNIT', yunit, 'Data units')
            hdul.append(fits.ImageHDU(data=rectimg[1]['spatial'],
                                      header=exthead, name='SLITPOS'))

            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])

            # retrieve an approximate atran data file if necessary
            if atmosthresh > 0:
                atran_dir = os.path.join(self.calres['pathcal'],
                                         'grism', 'atran')
                atran = get_atran(header, self.calres['resolution'],
                                  atran_dir=atran_dir,
                                  wmin=self.calres['wmin'],
                                  wmax=self.calres['wmax'])
            else:
                atran = None

            medprof, fitprof = \
                mkspatprof(rectimg, return_fit_profile=True,
                           ndeg=fit_order, bgsub=bg_sub, atran=atran,
                           atmosthresh=atmosthresh, smooth_sigma=None)

            # add profiles to HDUList
            hdinsert(exthead, 'BUNIT', '', 'Data units')
            hdul.append(fits.ImageHDU(data=fitprof[1],
                                      header=exthead, name='SPATIAL_MAP'))
            hdul.append(fits.ImageHDU(data=medprof[1],
                                      header=exthead, name='SPATIAL_PROFILE'))

            # save if desired
            if param.get_value('save'):
                self.write_output(hdul, outname)
            results.append(hdul)

        self.input = results
        self.set_display_data()

    def _parse_apertures(self, input_position, nfiles):
        """
        Parse aperture parameters from input string.

        Parameters
        ----------
        input_position : str
            Input parameter string.
        nfiles : int
            Number of input files expected.

        Returns
        -------
        list
            List of length `nfiles`, containing lists of floating point
            aperture values.
        """
        bad_msg = ['Could not read input_position '
                   f"parameter: '{input_position}'",
                   'Aperture positions should be comma-separated '
                   'values, in arcsec up the slit. ',
                   'To specify different values for different '
                   'input files, provide a semi-colon separated '
                   'list matching the number of input files.']

        apertures = []
        filepos = list(str(input_position).split(';'))
        if len(filepos) == 1:
            filepos = filepos * nfiles
        elif len(filepos) != nfiles:
            for msg in bad_msg:
                log.error(msg)
            raise ValueError('Invalid position parameter.')
        for fp in filepos:
            pos = list(fp.split(','))
            try:
                pos = [float(ap) for ap in pos]
            except (ValueError, TypeError):
                for msg in bad_msg:
                    log.error(msg)
                raise ValueError('Invalid position parameter.') from None
            apertures.append(pos)
        return apertures

    def _parse_bg(self, bg_string, nfiles):
        """
        Parse background parameters from input string.

        Parameters
        ----------
        bg_string : str
            Input parameter string.
        nfiles : int
            Number of input files expected.

        Returns
        -------
        list
            List of length `nfiles`, containing lists of floating point
            background start, stop values.
        """
        bad_msg = ['Could not read background region '
                   f"parameter: '{bg_string}'",
                   'Background regions should be comma-separated '
                   'values, in arcsec up the slit, as start-stop. ',
                   'To specify different values for different '
                   'input files, provide a semi-colon separated '
                   'list matching the number of input files.']
        bgr = []
        filepos = list(str(bg_string).split(';'))
        if len(filepos) == 1:
            filepos = filepos * nfiles
        elif len(filepos) != nfiles:
            for msg in bad_msg:
                log.error(msg)
            raise ValueError('Invalid background region parameter.')
        for fp in filepos:
            bg_set = list(fp.split(','))
            bg_list = []
            for bg_reg in bg_set:
                bg_range = bg_reg.split('-')
                if len(bg_range) == 1 and str(bg_range[0]).strip() == '':
                    # allow empty set for background regions
                    bg_list.append([])
                else:
                    try:
                        start, stop = bg_range
                        bg_list.append((float(start), float(stop)))
                    except (ValueError, TypeError):
                        for msg in bad_msg:
                            log.error(msg)
                        raise ValueError('Invalid background '
                                         'region parameter.') from None
            bgr.append(bg_list)
        return bgr

    def locate_apertures(self):
        """Automatically find aperture centers."""
        from sofia_redux.spectroscopy.findapertures import find_apertures

        # get parameters
        param = self.get_parameter_set()
        method = param.get_value('method')
        num_aps = param.get_value('num_aps')
        input_position = param.get_value('input_position')
        fwhm_par = param.get_value('fwhm')

        apertures = []
        if str(method).strip().lower() == 'fix to center':
            log.info('Fixing aperture to slit center.')
            positions = None
            fix_ap = True
            num_aps = 1
        elif str(method).strip().lower() == 'fix to input':
            log.info('Fixing aperture to input positions.')
            positions = self._parse_apertures(
                input_position, len(self.input))
            fix_ap = True
        elif str(method).strip().lower() == 'step up slit':
            log.info(f'Fixing aperture to {num_aps} positions.')
            positions = None
            fix_ap = True
        else:
            log.info('Finding aperture positions from Gaussian fits.')
            if str(input_position).strip() == '':
                positions = None
            else:
                positions = self._parse_apertures(
                    input_position, len(self.input))
            fix_ap = False

        for i, hdul in enumerate(self.input):
            profile = {1: [hdul['SLITPOS'].data,
                           hdul['SPATIAL_PROFILE'].data]}
            if positions is not None:
                guess = {1: positions[i]}
                npeaks = len(positions[i])
            else:
                guess = None
                npeaks = num_aps

            ap = find_apertures(profile, npeaks=npeaks, positions=guess,
                                fwhm=fwhm_par, fix=fix_ap,
                                box_width=('stddev', 3))
            apertures.append(ap[1])

        # add apertures to FITS headers
        log.info('')
        log.info('Apertures found:')
        results = []
        fit_fwhm = []
        yunit = self.input[0]['SLITPOS'].header.get('BUNIT', 'arcsec')
        for i, hdul in enumerate(self.input):
            # update output name
            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])

            log.info('  {}'.format(hdul[0].header['FILENAME']))
            appos = []
            apsign = []
            apfwhm = []
            fwhm_list = []
            for ap in apertures[i]:
                pos = ap['position']
                sign = ap['sign']
                fwhm = ap['fwhm']
                appos.append('{:.3f}'.format(pos))
                apsign.append('{:d}'.format(sign))
                apfwhm.append('{:.3f}'.format(fwhm))
                if fix_ap:
                    log.info('    {:.3f} {} '
                             '(sign: {})'.format(pos, yunit, sign))
                else:
                    log.info('    {:.3f} {} '
                             '(sign: {}, fit '
                             'FWHM: {:.3f})'.format(pos, yunit, sign, fwhm))
                    fwhm_list.append(fwhm)
            log.info('')
            fit_fwhm.extend(fwhm_list)

            # add apertures to header
            hdinsert(hdul[0].header, 'APPOSO01', ','.join(appos),
                     comment=f'Aperture positions [{yunit}]')
            hdinsert(hdul[0].header, 'APSGNO01', ','.join(apsign),
                     comment='Aperture signs')

            # add FWHM to header
            if not fix_ap:
                comment = f'Fit aperture FWHM [{yunit}]'
            else:
                comment = f'Assumed aperture FWHM [{yunit}]'
            hdinsert(hdul[0].header, 'APFWHM01', ','.join(apfwhm),
                     comment=comment)

            # save if desired
            if param.get_value('save'):
                self.write_output(hdul, outname)
            results.append(hdul)

        if len(fit_fwhm) > 0:
            log.info(f'Mean fit FWHM: '
                     f'{np.mean(fit_fwhm):.2f} '
                     f'+/- {np.std(fit_fwhm):.2f} {yunit}')

        self.input = results
        self.set_display_data()

    def _trace_region(self, header, filenum, prodtype,
                      trace_x, trace_y, calib, trace_fit,
                      fit_direction='x'):
        """
        Save a region file showing the trace fit.

        Parameters
        ----------
        header : fits.Header
            Used to determine the output filename.
        filenum : list
            Used to determine the output filename.
        prodtype : str
            Used to determine the output filename.
        trace_x : array-like
            X values for input data points.
        trace_y
            Y values for input data points.
        calib : array-like
            Independent values for trace fit.
        trace_fit : array-like
            Dependent values from trace fit.
        fit_direction : {'x', 'y'}, optional
            The direction of the independent trace fit values.

        Returns
        -------
        str
            The output file name.
        """

        region_name = self.getfilename(header, update=False,
                                       prodtype=prodtype, filenum=filenum)
        region_name = os.path.join(self.output_directory,
                                   os.path.splitext(region_name)[0] + '.reg')

        point = 'image;point({:f} {:f}) ' \
                '# point=x color=green tag={{trace_pt}}\n'
        # x1 y1 x2 y2
        line = 'wcs;linear;line({:f} {:f} {:f} {:f}) ' \
               '# color=red width=1 tag={{trace_fit_{}}}\n'

        with open(region_name, 'w') as fh:
            for x, y in zip(trace_x, trace_y):
                if not np.any(np.isnan([x, y])):
                    try:
                        point_reg = point.format(x + 1, y + 1)
                        fh.write(point_reg)
                    except TypeError:  # pragma: no cover
                        log.warning(f'Invalid point: {x}, {y}')
            if trace_fit is not None:
                for i in range(len(trace_fit)):
                    for j in range(1, len(calib)):
                        if np.any(np.isnan([trace_fit[i][j - 1],
                                            trace_fit[i][j]])):
                            continue
                        if str(fit_direction) == 'x':
                            line_reg = line.format(
                                calib[j - 1], trace_fit[i][j - 1],
                                calib[j], trace_fit[i][j], j)
                            fh.write(line_reg)
                        elif str(fit_direction) == 'y':
                            line_reg = line.format(
                                trace_fit[i][j - 1], calib[j - 1],
                                trace_fit[i][j], calib[j], j)
                            fh.write(line_reg)
        log.info('Wrote region file {}'.format(region_name))

        return region_name

    def trace_continuum(self):
        """Trace continuum at aperture locations."""
        from sofia_redux.spectroscopy.tracespec import tracespec

        # get parameters
        param = self.get_parameter_set()
        method = param.get_value('method')
        fit_order = param.get_value('fit_order')
        fit_thresh = param.get_value('fit_thresh')
        step_size = param.get_value('step_size')
        attach_pos = param.get_value('attach_trace_xy')

        if 'fix' in str(method).lower():
            log.info('Fixing trace to aperture center.')
            fix = True
        else:
            log.info('Fitting trace to continuum.')
            fix = False

        results = []
        regions = []
        for i, hdul in enumerate(self.input):

            # retrieve data from input
            wave = hdul['WAVEPOS'].data
            space = hdul['SLITPOS'].data
            appos = self._parse_apertures(hdul[0].header['APPOSO01'], 1)[0]
            apfwhm = self._parse_apertures(hdul[0].header['APFWHM01'], 1)[0]

            if fix:
                trace_fit = []
                for ap in appos:
                    trace_fit.append(np.array([ap] * len(wave)))
            else:
                rectimg = {1: {'image': hdul[0].data,
                               'wave': wave,
                               'spatial': space}}
                positions = {1: appos}
                fwhm = np.mean(apfwhm)

                # trace continuua
                trace_info = {}
                with set_log_level('ERROR'):
                    tracespec(rectimg, positions, fwhm=fwhm,
                              fitorder=fit_order,
                              info=trace_info, fitthresh=fit_thresh,
                              step=step_size, sumap=step_size,
                              box_width=('stddev', step_size))
                if len(trace_info) == 0:
                    msg = 'Trace fit failed. Try fixing to aperture center.'
                    log.error(msg)
                    raise ValueError(msg)

                # read fit data from info structure
                trace_mask = trace_info[1]['mask']
                trace_x = []
                trace_y = []
                for j, m in enumerate(trace_mask):
                    trace_x.extend(trace_info[1]['x'][j][m])
                    trace_y.extend(trace_info[1]['y'][j][m])
                trace_model = trace_info[1]['trace_model']
                if None in trace_model:
                    msg = 'Trace fit failed. Try fixing to aperture center.'
                    log.error(msg)
                    raise ValueError(msg)
                trace_fit = [m(wave) for m in trace_model]

                # make a region file to display over data
                prodname = self.prodnames[self.prodtypes[self.step_index]]
                region = self._trace_region(hdul[0].header.copy(),
                                            self.filenum[i],
                                            prodname, trace_x, trace_y,
                                            wave, trace_fit)
                regions.append(region)

                # attach fit x and y positions, if needed
                if attach_pos:
                    exthead = fits.Header()
                    hdinsert(exthead, 'BUNIT', 'pixel', 'Data units')
                    hdul.append(fits.ImageHDU(data=np.array(trace_x),
                                              header=exthead,
                                              name='APERTURE_XPOS'))
                    hdul.append(fits.ImageHDU(data=np.array(trace_y),
                                              header=exthead,
                                              name='APERTURE_YPOS'))

            # store trace fit in HDUList to use as aperture center
            exthead = fits.Header()
            hdinsert(exthead, 'BUNIT', 'arcsec', 'Data units')
            hdinsert(exthead, 'FITTRACE', (not fix),
                     'Trace from continuum fits')
            hdul.append(fits.ImageHDU(data=np.array(trace_fit),
                                      header=exthead, name='APERTURE_TRACE'))

            # update output name
            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                self.write_output(hdul, outname)

            results.append(hdul)

        self.input = results
        self.set_display_data(regions=regions)

    def set_apertures(self):
        """Set aperture radii."""
        from sofia_redux.spectroscopy.getapertures import get_apertures
        from sofia_redux.spectroscopy.mkapmask import mkapmask

        # get parameters
        param = self.get_parameter_set()
        full_slit = param.get_value('full_slit')
        refit = param.get_value('refit')
        apsign_list = param.get_value('apsign')
        aprad_list = param.get_value('aprad')
        psfrad_list = param.get_value('psfrad')
        bgr_list = param.get_value('bgr')

        fix_apsign, fix_aprad, fix_psfrad, fix_bgr = \
            False, False, False, False
        if not full_slit:
            if str(apsign_list).strip().lower() != '':
                apsign_list = self._parse_apertures(apsign_list,
                                                    len(self.input))
                fix_apsign = True
            if str(aprad_list).strip().lower() != '':
                aprad_list = self._parse_apertures(aprad_list,
                                                   len(self.input))
                fix_aprad = True
            if str(psfrad_list).strip().lower() != '':
                psfrad_list = self._parse_apertures(psfrad_list,
                                                    len(self.input))
                fix_psfrad = True
            if str(bgr_list).strip().lower() != '':
                bgr_list = self._parse_bg(bgr_list, len(self.input))
                fix_bgr = True

        results = []
        for i, hdul in enumerate(self.input):
            log.info('')
            log.info(hdul[0].header['FILENAME'])

            # retrieve data from input
            space = hdul['SLITPOS'].data
            wave = hdul['WAVEPOS'].data
            profile = hdul['SPATIAL_PROFILE'].data
            aptrace = hdul['APERTURE_TRACE'].data
            appos = self._parse_apertures(hdul[0].header['APPOSO01'], 1)[0]
            apfwhm = self._parse_apertures(hdul[0].header['APFWHM01'], 1)[0]

            if full_slit:
                half_slit = max([(space.max() - space.min()) / 2,
                                 appos[0] - space.min(),
                                 space.max() - appos[0]])
                ap = {'position': appos[0],
                      'fwhm': apfwhm[0],
                      'sign': 1,
                      'psf_radius': half_slit,
                      'aperture_radius': half_slit}
                aplist = [ap]
                aperture_regions = {'apertures': aplist,
                                    'background': {'regions': []}}
            else:
                if fix_apsign:
                    apsign = apsign_list[i]
                else:
                    apsign = self._parse_apertures(
                        hdul[0].header['APSGNO01'], 1)[0]

                aplist = []
                for j, pos in enumerate(appos):
                    ap = {'position': pos,
                          'fwhm': apfwhm[j]}
                    if len(apsign) > j:
                        ap['sign'] = apsign[j]
                    else:
                        ap['sign'] = apsign[-1]
                    if fix_aprad:
                        if len(aprad_list[i]) > j:
                            ap['aperture_radius'] = aprad_list[i][j]
                        else:
                            ap['aperture_radius'] = aprad_list[i][-1]
                    if fix_psfrad:
                        if len(psfrad_list[i]) > j:
                            ap['psf_radius'] = psfrad_list[i][j]
                        else:
                            ap['psf_radius'] = psfrad_list[i][-1]
                    aplist.append(ap)

                apertures = {1: aplist}
                profiles = {1: np.vstack([space, profile])}

                if fix_bgr:
                    aperture_regions = get_apertures(profiles, apertures,
                                                     get_bg=False,
                                                     refit_fwhm=refit)[1]
                    aperture_regions['background'] = {'regions': bgr_list[i]}
                else:
                    aperture_regions = get_apertures(profiles, apertures,
                                                     refit_fwhm=refit)[1]

            # update output name
            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])

            # log aperture values
            apsign, aprad, appsfrad = [], [], []
            for j, ap in enumerate(aperture_regions['apertures']):
                pos = ap['position']
                sign = '{:d}'.format(int(ap['sign']))
                rad = '{:.3f}'.format(ap['aperture_radius'])
                psfrad = '{:.3f}'.format(ap['psf_radius'])

                apsign.append(sign)
                aprad.append(rad)
                appsfrad.append(psfrad)

                log.info('  Aperture {}:'.format(j))
                log.info('           position: {:.3f}'.format(pos))
                log.info('               sign: {}'.format(sign))
                log.info('         PSF radius: {}'.format(psfrad))
                log.info('    aperture radius: {}'.format(rad))
                log.info('')

                # also add the trace into the aperture, for use
                # in making an aperture mask
                ap['trace'] = aptrace[j]

            log.info('Background regions:')
            bgr = []
            for start, stop in aperture_regions['background']['regions']:
                reg = '{:.3f}-{:.3f}'.format(start, stop)
                bgr.append(reg)
                log.info('  {}'.format(reg))
            if len(bgr) == 0:
                log.info('  (None)')
            log.info('')

            # add to header
            hdinsert(hdul[0].header, 'APSGNO01', ','.join(apsign),
                     comment='Aperture signs')
            hdinsert(hdul[0].header, 'APRADO01', ','.join(aprad),
                     comment='Aperture radii [arcsec]')
            hdinsert(hdul[0].header, 'PSFRAD01', ','.join(appsfrad),
                     comment='Aperture PSF radii [arcsec]')
            hdinsert(hdul[0].header, 'BGR', ','.join(bgr),
                     comment='Aperture background regions [arcsec]')

            # make aperture mask and append to hdul
            apmask = mkapmask(space, wave, aperture_regions['apertures'],
                              aperture_regions['background']['regions'])
            exthead = hdul['BADMASK'].header.copy()
            hdul.append(fits.ImageHDU(data=np.array(apmask),
                                      header=exthead, name='APERTURE_MASK'))

            # save if desired
            if param.get_value('save'):
                self.write_output(hdul, outname)
            results.append(hdul)

        self.input = results
        self.set_display_data()

    def subtract_background(self):
        """Subtract background along columns."""
        from sofia_redux.spectroscopy.extspec import col_subbg

        # get parameters
        param = self.get_parameter_set()
        skip_bg = param.get_value('skip_bg')
        bg_fit_order = param.get_value('bg_fit_order')

        if skip_bg:
            log.info('No background subtraction performed.')
            return

        results = []
        for i, hdul in enumerate(self.input):
            log.info('')
            log.info(hdul[0].header['FILENAME'])

            # retrieve data from input
            image = hdul['FLUX'].data
            err = hdul['ERROR'].data
            mask = (hdul['BADMASK'].data < 1)
            space = hdul['SLITPOS'].data

            apmask = hdul['APERTURE_MASK'].data

            has_bg = np.any(np.isnan(apmask))
            if not has_bg:
                log.info('No background regions defined.')
            else:
                # correct each column for background identified in apmask
                nwave = image.shape[1]
                for wavei in range(nwave):
                    bg_subtracted_col = col_subbg(
                        space, image[:, wavei], err[:, wavei]**2,
                        apmask[:, wavei], mask[:, wavei],
                        bg_fit_order)
                    if bg_subtracted_col is not None:
                        # result is flux, variance, coefficients
                        image[:, wavei] = bg_subtracted_col[0]
                        err[:, wavei] = np.sqrt(bg_subtracted_col[1])

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

    def extract_spectra(self):
        """Extract 1D spectra from apertures."""
        from sofia_redux.instruments.forcast.getatran import get_atran
        from sofia_redux.spectroscopy.extspec import extspec

        # get parameters
        param = self.get_parameter_set()
        use_profile = param.get_value('use_profile')
        fix_bad = param.get_value('fix_bad')
        threshold = param.get_value('threshold')
        optimal = 'optimal' in str(param.get_value('method')).lower()

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
            profile = hdul['SPATIAL_PROFILE'].data
            spatmap = hdul['SPATIAL_MAP'].data
            apmask = hdul['APERTURE_MASK'].data

            apsign = self._parse_apertures(header['APSGNO01'], 1)[0]
            rectimg = {1: {'image': image, 'variance': var, 'mask': mask,
                           'wave': wave, 'spatial': space, 'header': header,
                           'apmask': apmask, 'apsign': apsign}}

            if use_profile:
                spatmap = None
                profile = {1: profile}
            else:
                spatmap = {1: spatmap}
                profile = None

            spectra = extspec(rectimg,
                              profile=profile, spatial_map=spatmap,
                              optimal=optimal, fix_bad=fix_bad,
                              sub_background=False,
                              threshold=threshold)[1]

            # update flux, error, and mask planes -- they may have had
            # bad pixels corrected in the extraction process
            hdul['FLUX'].data = rectimg[1]['image']
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                hdul['ERROR'].data = np.sqrt(rectimg[1]['variance'])
            boolmask = rectimg[1]['mask']
            mask = np.zeros(boolmask.shape, dtype=int)
            mask[~boolmask] = 1
            hdul['BADMASK'].data = mask

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

            # attach an approximate transmission
            atran_dir = os.path.join(self.calres['pathcal'],
                                     'grism', 'atran')
            atran = get_atran(hdul[0].header, self.calres['resolution'],
                              atran_dir=atran_dir,
                              wmin=self.calres['wmin'],
                              wmax=self.calres['wmax'])
            outtrans = np.interp(hdul['WAVEPOS'].data, atran[0], atran[1],
                                 left=np.nan, right=np.nan)
            tdata = np.full_like(spectra[:, 1, :], np.nan)
            tdata[:] = outtrans
            exthead = hdul['SPECTRAL_FLUX'].header.copy()
            hdul.append(fits.ImageHDU(tdata, exthead,
                                      name='TRANSMISSION'))
            hdinsert(hdul['TRANSMISSION'].header, 'BUNIT', '', 'Data units')

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
                specname = outname.replace('SPM', 'SPC')
                spec[0].header['FILENAME'] = os.path.basename(specname)
                spec[0].header['PRODTYPE'] = 'spectra_1d'
                self.write_output(spec, specname)

            results.append(hdul)

        log.info('')
        self.input = results
        self.set_display_data()

    def merge_apertures(self):
        """Merge apertures to a single 1D spectrum."""
        from sofia_redux.spectroscopy.getspecscale import getspecscale
        from sofia_redux.toolkit.image.combine import combine_images

        # get parameters
        param = self.get_parameter_set()
        method = param.get_value('method')
        weighted = param.get_value('weighted')

        results = []
        for i, hdul in enumerate(self.input):
            log.info('')
            log.info(hdul[0].header['FILENAME'])

            # retrieve data from input
            header = hdul[0].header
            spec_flux = hdul['SPECTRAL_FLUX'].data
            spec_err = hdul['SPECTRAL_ERROR'].data
            spec_trans = hdul['TRANSMISSION'].data

            # if only 1 aperture, just reshape to 1D
            if spec_flux.ndim < 2 or spec_flux.shape[0] < 2:
                log.info('Only one aperture; no merge to perform.')
                spec_flux = spec_flux.reshape(spec_flux.size)
                spec_err = spec_err.reshape(spec_err.size)
                spec_trans = spec_trans.reshape(spec_err.size)
            else:
                n_aps = spec_flux.shape[0]
                spec_trans = spec_trans[0]

                # scale to highest flux spectrum
                medflux = np.nanmedian(spec_flux, axis=1)
                scale = getspecscale(spec_flux, refidx=np.argmax(medflux))
                log.info('Spectral scales: {}'.format(
                         ', '.join(['{:.2f}'.format(s) for s in scale])))
                spec_flux *= scale[:, None]
                spec_err *= scale[:, None]

                # mean or median combine the spectra
                log.info('Combining with {}'.format(method))
                outflux, outvar = combine_images(
                    spec_flux, variance=spec_err**2,
                    method=method, weighted=weighted, robust=False)
                spec_flux = outflux
                spec_err = np.sqrt(outvar)

                # update NEXP and EXPTIME:
                # Raw nexp is double for NMC, so total exposures is
                #     nexp + (n_aps-1)*nexp/2
                # Otherwise, will be
                #     nexp * n_aps
                nexp = header.get('NEXP', 1)
                exptime = header.get('EXPTIME', 0)
                cnmode = str(self.calres['cnmode']).lower()
                if 'nmc' in cnmode or 'scan' in cnmode:
                    newnexp = nexp + (n_aps - 1) * nexp / 2.0
                    newtime = exptime + (n_aps - 1) * exptime / 2.0
                else:
                    newnexp = nexp * n_aps
                    newtime = exptime * n_aps

                hdinsert(header, 'NEXP', newnexp)
                hdinsert(header, 'EXPTIME', newtime)

            # update data
            hdul['SPECTRAL_FLUX'].data = spec_flux
            hdul['SPECTRAL_ERROR'].data = spec_err
            hdul['TRANSMISSION'].data = spec_trans

            # update output name
            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                log.info('')
                log.info('Saving spectral image:')
                self.write_output(hdul, outname)

            if param.get_value('save_1d'):
                log.info('')
                log.info('Saving 1D spectra:')
                spec = self._make_1d(hdul)
                specname = outname.replace('MGM', 'MRG')
                spec[0].header['FILENAME'] = os.path.basename(specname)
                spec[0].header['PRODTYPE'] = 'merged_spectrum_1d'
                self.write_output(spec, specname)
            results.append(hdul)

        log.info('')
        self.input = results
        self.set_display_data()

    def _make_plot(self, filename, *args, **kwargs):
        """Make diagnostic plots for ATRAN optimization."""
        from cycler import cycler
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)

        # Brewer 11-color diverging Spectral
        # http://colorbrewer2.org/#type=diverging&scheme=Spectral&n=11
        cidx = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b',
                '#ffffbf', '#e6f598', '#abdda4', '#66c2a5',
                '#3288bd', '#5e4fa2']
        if 'brewer' in kwargs:
            ax.set_prop_cycle(cycler('color', cidx))
            del kwargs['brewer']

        if 'label' in kwargs:
            labels = kwargs['label']
            for label, y in zip(labels, args[1]):
                ax.plot(args[0], y, linewidth=0.5, label=label)
            ax.legend(fontsize='xx-small')
            del kwargs['label']
        else:
            ax.plot(*args, linewidth=0.5)
        if 'vline' in kwargs:
            ax.axvline(kwargs['vline'], linestyle=':', linewidth=1)
            del kwargs['vline']
        if 'hline' in kwargs:
            ax.axhline(kwargs['hline'], color='red',
                       linestyle=':', linewidth=1)
            del kwargs['hline']
        if 'text' in kwargs:
            ax.text(0.95, 0.05, kwargs['text'], transform=ax.transAxes,
                    fontsize='xx-small', horizontalalignment='right')
            del kwargs['text']
        ax.set(**kwargs)
        fig.savefig(filename)

    def flux_calibrate(self):
        """Calibrate spectral flux."""
        from glob import glob
        from sofia_redux.instruments.forcast.getatran import get_atran
        from sofia_redux.spectroscopy.fluxcal import fluxcal
        from sofia_redux.spectroscopy.radvel import radvel

        # get parameters
        param = self.get_parameter_set()
        skipcal = param.get_value('skip_cal')
        respfile = param.get_value('respfile')
        resolution = param.get_value('resolution')
        optimize = param.get_value('optimize_atran')
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

        # check optimize and atran dir parameters
        if optimize and atrandir is None:
            log.warning('Cannot optimize without ATRAN directory. Using '
                        'default ATRAN file.')
            optimize = False
        if optimize and atranfile is not None:
            optimize = False

        # set waveshift to None if not provided
        if str(waveshift).strip() == '' or np.allclose(waveshift, 0):
            waveshift = None
        else:
            if auto_shift:
                log.info('Disabling auto-shift since manual shift '
                         'was specified.')
                auto_shift = False

        # get response data
        try:
            with set_log_level('ERROR'):
                resp_hdul = fits.open(respfile)
            rhead = resp_hdul[0].header
            response_data = resp_hdul[0].data
            resp_hdul.close()

            detbias = rhead.get('DETBIAS', -9999.)
            syserr = rhead.get('SYSERR', 0.0)
            resname = respfile.split(self.calres['pathcal'])[-1]
            response = {1: {'wave': response_data[0, :],
                            'response': response_data[1, :],
                            'error': response_data[2, :]}}

            log.info('Using response file: {}'.format(respfile))
        except (OSError, ValueError, IndexError, TypeError):
            if param.get_value('making_response'):
                if optimize:
                    log.warning('No response file. Turning off '
                                'ATRAN optimization.')
                    optimize = False
                detbias = None
                syserr = None
                resname = 'UNKNOWN'
                response = None
            elif respfile == '':
                msg = 'Missing response file; cannot calibrate spectra'
                log.error(msg)
                raise ValueError(msg) from None
            else:
                msg = 'Bad response file: {}'.format(respfile)
                log.error(msg)
                raise ValueError(msg) from None

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
        regex2 = re.compile(r'^atran_([0-9]+)K_([0-9]+)deg_'
                            r'([0-9]+)pwv_4-50mum\.fits$')
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

            # check detbias against response
            db = header.get('DETBIAS', -9999.)
            if detbias is not None and not np.allclose(detbias, db, atol=0.1):
                log.warning('DETBIAS mismatch between response '
                            'file and input file.')
                log.warning('Response DETBIAS: {}'.format(detbias))
                log.warning('Input DETBIAS: {}'.format(db))

            # check s/n for optimization, auto shift
            test_opt = optimize
            test_auto = auto_shift
            if test_opt or test_auto:
                s2n = np.nanmedian(spec_flux / spec_err)
                if s2n < snthresh:
                    if test_opt:
                        log.warning(f'S/N {s2n:.1f} too low to optimize '
                                    f'ATRAN correction. '
                                    f'Using default ATRAN file.')
                        test_opt = False
                    if test_auto:
                        log.warning(f'S/N {s2n:.1f} too low to auto-shift '
                                    f'wavelengths. Disabling auto-shift.')
                        test_auto = False

            # get atran data, trying the specified directory first
            base_atran = get_atran(header, resolution, filename=atranfile,
                                   atran_dir=atrandir, use_wv=test_opt)
            if base_atran is None:
                if atrandir is None:
                    msg = 'No matching ATRAN files.'
                    raise ValueError(msg)
                else:
                    # if not found, try the default directory
                    base_atran = get_atran(header, resolution,
                                           filename=atranfile,
                                           atran_dir=None, use_wv=test_opt)
                    if base_atran is None:
                        msg = 'No matching ATRAN files.'
                        raise ValueError(msg)

            if test_opt:
                # read alt and za match from base file
                basefile = header['ATRNFILE']
                match = regex2.match(basefile)
                if match is None:
                    msg = 'No matching ATRAN files.'
                    log.error(msg)
                    raise ValueError(msg)

                # find all PWV matches at that alt/za
                alt = match.group(1)
                za = match.group(2)
                atranfiles = glob(
                    os.path.join(atrandir,
                                 'atran_{}K_{}deg_'
                                 '*pwv_4-50mum.fits'.format(alt, za)))

                # read all matching atran files
                atran = []
                pwv = []
                tmphead = header.copy()
                for filename in atranfiles:
                    atran.append(get_atran(tmphead, resolution,
                                           filename=filename))
                    match = regex2.match(os.path.basename(filename))
                    pwv.append(float(match.group(3)))

                # sort by PWV
                sort_idx = np.argsort(pwv)
                atranfiles = np.array(atranfiles)[sort_idx]
                pwv = np.array(pwv)[sort_idx]
                atran = np.array(atran)[sort_idx]
            else:
                atran = [base_atran]
                atranfiles = None
                pwv = None

            spectra = {1: [{'wave': wave,
                            'flux': image,
                            'error': err,
                            'spectral_flux': spec_flux,
                            'spectral_error': spec_err,
                            'wave_shift': waveshift}]}

            result = fluxcal(spectra, atran, response,
                             auto_shift=test_auto,
                             shift_limit=shift_limit,
                             model_order=model_order)
            if result is None:
                msg = 'Problem in flux calibration.'
                log.error(msg)
                raise ValueError(msg)
            result = result[1][0]

            # update average calibration error in header
            if syserr is not None:
                calerr = syserr / np.nanmean(result['response'])
                hdinsert(header, 'CALERR', calerr,
                         comment='Fractional flux calibration error')

            # update optimization results in header
            if test_opt:
                idx = result['atran_index']
                afile = os.path.basename(atranfiles[idx])
                fit_pwv = pwv[idx]

                # error value: estimate from next highest chisq value
                chisq = np.array(result['fit_chisq'])
                csort = np.argsort(chisq)
                chisqerr = np.min(chisq) + 1.0

                larger = np.where(chisq[csort] > chisqerr)[0]
                if len(larger) > 0:
                    next_idx = larger[0]
                    fit_pwv_err = np.abs(pwv[csort][next_idx] - pwv[idx])
                    # keep a floor of 1um for PWV error
                    if fit_pwv_err < 1:  # pragma: no cover
                        fit_pwv_err = 1.0
                else:  # pragma: no cover
                    # failsafe, in case chisq is very flat
                    fit_pwv_err = 1.0

                pwv_msg = 'Fit PWV: {:.1f} ' \
                          '+/- {:.1f}'.format(fit_pwv, fit_pwv_err)
                log.info(pwv_msg)

                # make diagnostic plots

                # pwv vs. chisq from fit
                pngname = self.getfilename(header, update=False,
                                           prodtype='PWV',
                                           filenum=self.filenum[i])
                pngname = os.path.join(self.output_directory,
                                       os.path.splitext(pngname)[0] + '.png')
                self._make_plot(
                    pngname, pwv, result['fit_chisq'],
                    title=hdul[0].header['FILENAME'].replace('MGM', 'CAL'),
                    xlabel='Precipitable water vapor (um)',
                    ylabel='Fit Chi^2',
                    vline=fit_pwv, hline=chisqerr,
                    text='{}\n{}'.format(afile, pwv_msg))

                # all corrected data by pwv
                pngname = pngname.replace('PWV', 'OPT')
                labels = []
                for p in pwv:
                    if p == fit_pwv:
                        labels.append('{:.0f} um (optimal)'.format(p))
                    else:
                        labels.append('{:.0f} um'.format(p))
                self._make_plot(
                    pngname, wave, np.array(result['all_corrected']),
                    label=labels, brewer=True,
                    title=hdul[0].header['FILENAME'].replace('MGM', 'CAL'),
                    xlabel='Wavelength (um)', ylabel='Flux')

                # update header
                hdinsert(header, 'ATRNFILE', afile, comment='ATRAN file')
                hdinsert(header, 'FITPWV', fit_pwv, comment='Fit PWV')
                hdinsert(header, 'FITPWVER', fit_pwv_err,
                         comment='Fit PWV Error')

            # log ATRAN file and wave shift as used
            log.info('ATRAN file used: {}'.format(header['ATRNFILE']))
            hdinsert(header, 'WAVSHIFT', result['wave_shift'],
                     'Wavelength shift (pix)')
            log.info('Wavelength shift applied: '
                     '{:.2f} pixels'.format(result['wave_shift']))

            # log response file and resolution
            hdinsert(header, 'RSPNFILE', resname, 'Response file')
            hdinsert(header, 'RP', resolution, 'Resolving power')

            # update data
            hdul[0].header = header
            hdul['FLUX'].data = result['flux']
            hdul['ERROR'].data = result['error']
            hdul['WAVEPOS'].data = wave
            hdul['SPECTRAL_FLUX'].data = result['spectral_flux']
            hdul['SPECTRAL_ERROR'].data = result['spectral_error']

            # also append response and transmission curves
            exthead = hdul['SPECTRAL_FLUX'].header.copy()
            if 'TRANSMISSION' in hdul:
                del hdul['TRANSMISSION']
            hdul.append(fits.ImageHDU(result['transmission'], exthead,
                                      name='TRANSMISSION'))
            hdul.append(fits.ImageHDU(result['response'], exthead,
                                      name='RESPONSE'))
            hdul.append(fits.ImageHDU(result['response_error'],
                                      name='RESPONSE_ERROR'))

            # update BUNIT for all extensions
            if response is not None:
                hdul[0].header['PROCSTAT'] = 'LEVEL_3'
            hdinsert(hdul[0].header, 'RAWUNITS', 'Me/s',
                     'Data units before calibration')
            for ext in ['FLUX', 'ERROR']:
                hdinsert(hdul[ext].header, 'BUNIT', 'Jy/pixel', 'Data units')
            for ext in ['SPECTRAL_FLUX', 'SPECTRAL_ERROR']:
                hdinsert(hdul[ext].header, 'BUNIT', 'Jy', 'Data units')
            hdinsert(hdul['TRANSMISSION'].header, 'BUNIT', '', 'Data units')
            hdinsert(hdul['RESPONSE'].header, 'BUNIT', 'Me/s/Jy',
                     'Data units')
            hdinsert(hdul['RESPONSE_ERROR'].header, 'BUNIT', 'Me/s/Jy',
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

    def _make_1d(self, hdul, wavecal=True):
        spechdr = hdul[0].header.copy()

        # add some spextool-required header keywords
        if wavecal:
            hdinsert(spechdr, 'XUNITS', 'um', 'Spectral wavelength units')
        else:
            hdinsert(spechdr, 'XUNITS', 'pixels', 'Spectral wavelength units')
        hdinsert(spechdr, 'YUNITS',
                 hdul['SPECTRAL_FLUX'].header.get('BUNIT', 'UNKNOWN'),
                 'Spectral flux units')
        if hdul['SPECTRAL_FLUX'].data.ndim > 1:
            naps = hdul['SPECTRAL_FLUX'].data.shape[0]
        else:
            naps = 1
        hdinsert(spechdr, 'NAPS', naps, 'Number of apertures')
        hdinsert(spechdr, 'NORDERS', 1, 'Number of orders')
        hdinsert(spechdr, 'DISPO01', spechdr['CDELT1'],
                 'Dispersion [um pixel-1]')

        try:
            del spechdr['BUNIT']
        except KeyError:  # pragma: no cover
            pass

        specset = []
        for n in range(naps):
            if naps > 1:
                speclist = [hdul['WAVEPOS'].data,
                            hdul['SPECTRAL_FLUX'].data[n],
                            hdul['SPECTRAL_ERROR'].data[n]]
            else:
                speclist = [hdul['WAVEPOS'].data,
                            hdul['SPECTRAL_FLUX'].data,
                            hdul['SPECTRAL_ERROR'].data]
            if 'TRANSMISSION' in hdul:
                speclist.append(hdul['TRANSMISSION'].data)
            elif wavecal:
                # if transmission isn't present, attach an approximate one
                from sofia_redux.instruments.forcast.getatran import get_atran
                atran_dir = os.path.join(self.calres['pathcal'],
                                         'grism', 'atran')
                atran = get_atran(spechdr, self.calres['resolution'],
                                  atran_dir=atran_dir,
                                  wmin=self.calres['wmin'],
                                  wmax=self.calres['wmax'])
                adata = np.interp(hdul['WAVEPOS'].data, atran[0], atran[1],
                                  left=np.nan, right=np.nan)
                speclist.append(adata)
            if 'RESPONSE' in hdul:
                speclist.append(hdul['RESPONSE'].data)
            specdata = np.vstack(speclist)
            if naps == 1:
                specset = specdata
            else:
                specset.append(specdata)
        specset = np.array(specset)
        spec = fits.HDUList(fits.PrimaryHDU(data=specset,
                                            header=spechdr))
        return spec

    def combine_spectra(self):
        """
        Combine spectra.

        Calls `sofia_redux.instruments.forcast.coadd.coadd` and
        `sofia_redux.toolkit.image.combine.combine_images`
        for coaddition. The combination method may be configured in
        parameters.
        """
        from sofia_redux.instruments.forcast.hdmerge import hdmerge
        from sofia_redux.instruments.forcast.coadd import coadd
        from sofia_redux.instruments.forcast.register_datasets \
            import get_shifts
        from sofia_redux.toolkit.image.combine import combine_images

        # get parameters
        param = self.get_parameter_set()
        registration = param.get_value('registration')
        method = param.get_value('method')
        weighted = param.get_value('weighted')
        robust = param.get_value('robust')
        sigma = param.get_value('threshold')
        maxiters = param.get_value('maxiters')
        order = param.get_value('fit_order')
        window = param.get_value('fit_window')
        smoothing = param.get_value('smoothing')
        adaptive = param.get_value('adaptive_algorithm')
        edge = param.get_value('edge_threshold')
        combine_aps = param.get_value('combine_aps')

        if 'cube' in method:
            cube = True
            method = 'mean'
        else:
            cube = False

        # check for data to combine
        n_input = len(self.input)
        if n_input < 2 and not cube and not combine_aps:
            log.info('No data to combine.')

            if param.get_value('save'):
                hdul = self.input[0]
                outname = self.update_output(hdul, self.filenum,
                                             self.prodtypes[self.step_index])
                log.info('')
                log.info('Saving 1D spectra only:')
                spec = self._make_1d(hdul)
                specname = outname.replace('COA', 'CMB')
                spec[0].header['FILENAME'] = os.path.basename(specname)
                spec[0].header['PRODTYPE'] = 'combined_spectrum'
                self.write_output(spec, specname)
            return

        hdr_list = []
        data_list = []
        var_list = []
        spec_list = []
        spec_var_list = []
        trans_list = []
        resp_list = []
        exp_list = []
        test_wave = None
        for i, hdul in enumerate(self.input):
            if i == 0:
                test_wave = hdul['WAVEPOS'].data
            else:
                if not np.allclose(hdul['WAVEPOS'].data, test_wave):
                    msg = 'Mismatched wavelengths. Spectra cannot be combined.'
                    log.error(msg)
                    raise ValueError(msg)
            hdr_list.append(hdul[0].header)
            data_list.append(hdul['FLUX'].data)
            var_list.append(hdul['ERROR'].data ** 2)

            # gather spectra, accounting for possible multiple apertures
            n_ap = 1
            specflux = hdul['SPECTRAL_FLUX'].data
            specvar = hdul['SPECTRAL_ERROR'].data ** 2
            if 'TRANSMISSION' in hdul:
                spectrans = hdul['TRANSMISSION'].data
            else:
                spectrans = None
            if 'RESPONSE' in hdul:
                specresp = hdul['RESPONSE'].data
            else:
                specresp = None

            if specflux.ndim > 1:
                n_ap = specflux.shape[0]
                if n_ap == 1:
                    specflux = specflux[0]
                    specvar = specvar[0]
                    if spectrans is not None:
                        spectrans = spectrans[0]
                    if specresp is not None:
                        specresp = specresp[0]
            if combine_aps and n_ap > 1:
                for j in range(n_ap):
                    spec_list.append(specflux[j])
                    spec_var_list.append(specvar[j])
                    if spectrans is not None:
                        trans_list.append(spectrans[j])
                    if specresp is not None:
                        resp_list.append(specresp[j])
            else:
                spec_list.append(specflux)
                spec_var_list.append(specvar)
                if spectrans is not None:
                    trans_list.append(spectrans)
                if specresp is not None:
                    resp_list.append(specresp)

            exp_list.append(np.full_like(hdul['FLUX'].data,
                                         hdul[0].header.get('EXPTIME', 0.0)))

        reference = 'first'
        if 'header' in registration.lower():
            log.info('Applying dither offsets from the header.')
            datasets = list(zip(data_list, hdr_list))
            shifts = get_shifts(datasets, algorithm='HEADER',
                                do_wcs_shift=True, wcskey='A')

            for idx, s in enumerate(shifts):
                if s is None or not np.all(np.isfinite(s)):
                    log.warning("Failed to register dataset %i; "
                                "setting shift to 0." % idx)
                    shifts[idx] = np.array([0., 0.])

            # keep the first as reference position
            shifts -= shifts[0]

            shiftstr = ';'.join(['{:.2f},{:.2f}'.format(*s) for s in shifts])
            log.debug(f'CRPIX shifts applied: {shiftstr}')

            # update the headers with the shifts, keeping
            # the first as reference position
            for i, hdr in enumerate(hdr_list):
                hdr['CRPIX2A'] += shifts[i][1] - shifts[0][1]
                hdr['CRPIX3A'] += shifts[i][0] - shifts[0][0]
        else:
            if 'target' in registration.lower():
                log.info('Correcting for target motion, if necessary.')
                reference = 'target'
            else:
                log.info('Using WCS as is.')

        # combine all the data arrays
        if len(data_list) > 1 or cube:
            outhdr, outdata, outvar, outexp = \
                coadd(hdr_list, data_list, var_list, exp_list,
                      method=method, weighted=weighted,
                      robust=robust, sigma=sigma, maxiters=maxiters,
                      wcskey='A', rotate=cube, cube=cube, spectral=True,
                      fit_order=order, window=window,
                      smoothing=smoothing, adaptive_algorithm=adaptive,
                      edge_threshold=edge, reference=reference)
        else:
            outhdr = hdr_list[0].copy()
            outdata, outvar, outexp = data_list[0], var_list[0], exp_list[0]

        if len(spec_list) > 1:
            outspec, outspecvar = combine_images(
                spec_list, variance=spec_var_list, method=method,
                weighted=weighted, robust=robust, sigma=sigma,
                maxiters=maxiters)
        else:
            outspec, outspecvar = spec_list[0], spec_var_list[0]
        if len(trans_list) > 0:
            if n_input > 1:
                outtrans, _ = combine_images(trans_list, method=method,
                                             robust=False)
            else:
                outtrans = trans_list[0]
        else:
            outtrans = None
        if len(resp_list) > 0:
            if n_input > 1:
                outresp, _ = combine_images(resp_list, method=method,
                                            robust=False)
            else:
                outresp = resp_list[0]
        else:
            outresp = None

        # remove old WCS keys
        for hdr in hdr_list:
            for key in self.wcs_keys:
                if key in hdr:
                    del hdr[key]

        # merge all the headers
        outhdr = hdmerge(hdr_list, reference_header=outhdr)

        # update integration time from map
        exptime = np.nanmax(outexp)
        hdinsert(outhdr, 'EXPTIME', exptime,
                 comment='Nominal on-source integration time [s]')

        # make a header for the exposure and error maps
        # with appropriate WCS
        errhead = fits.Header()
        exphead = fits.Header()
        for key in self.wcs_keys:
            if cube:
                # 3D WCS now is primary -- remove 'A' suffix
                if key.endswith('A'):
                    key = key[:-1]
                    if key in outhdr:
                        hdinsert(errhead, key, outhdr[key],
                                 outhdr.comments[key])
                        if '3' not in key:
                            hdinsert(exphead, key, outhdr[key],
                                     outhdr.comments[key])
            else:
                if key in outhdr:
                    hdinsert(errhead, key, outhdr[key], outhdr.comments[key])
                    hdinsert(exphead, key, outhdr[key], outhdr.comments[key])

        # store output data: final extensions are
        # FLUX, ERROR, EXPOSURE, WAVEPOS,
        # SPECTRAL_FLUX, SPECTRAL_ERROR, TRANSMISSION, RESPONSE
        template = self.input[0]
        primary = fits.PrimaryHDU(data=outdata, header=outhdr)
        hdul = fits.HDUList(primary)
        hdinsert(errhead, 'BUNIT', outhdr.get('BUNIT', 'UNKNOWN'),
                 'Data units')
        hdul.append(fits.ImageHDU(data=np.sqrt(outvar),
                                  header=errhead,
                                  name='ERROR'))
        hdinsert(exphead, 'BUNIT', 's', 'Data units')
        hdul.append(fits.ImageHDU(data=outexp,
                                  header=exphead,
                                  name='EXPOSURE'))

        # update WAVEPOS with the output wavelength for the cube
        if cube:
            cube_wcs = WCS(outhdr)
            old_wave = template['WAVEPOS'].data
            new_wave = cube_wcs.wcs_pix2world(
                [0], [0], np.arange(outdata.shape[0]), 0)[2] * 1e6

            # interpolate transmission and response to match if necessary
            if outtrans is not None:
                outtrans = np.interp(new_wave, old_wave, outtrans,
                                     left=np.nan, right=np.nan)
            if outresp is not None:
                outresp = np.interp(new_wave, old_wave, outresp,
                                    left=np.nan, right=np.nan)

            template['WAVEPOS'].data = new_wave

        hdul.append(fits.ImageHDU(data=template['WAVEPOS'].data,
                                  header=template['WAVEPOS'].header,
                                  name='WAVEPOS'))

        # don't attach 1D spectra for cubes
        if not cube:
            hdul.append(fits.ImageHDU(data=outspec,
                                      header=template['SPECTRAL_FLUX'].header,
                                      name='SPECTRAL_FLUX'))
            hdul.append(fits.ImageHDU(data=np.sqrt(outspecvar),
                                      header=template['SPECTRAL_ERROR'].header,
                                      name='SPECTRAL_ERROR'))

        # but do attach transmission and response
        if outtrans is not None:
            hdul.append(fits.ImageHDU(data=outtrans,
                                      header=template['TRANSMISSION'].header,
                                      name='TRANSMISSION'))
        if outresp is not None:
            hdul.append(fits.ImageHDU(data=outresp,
                                      header=template['RESPONSE'].header,
                                      name='RESPONSE'))
        outname = self.update_output(hdul, self.filenum,
                                     self.prodtypes[self.step_index])

        # save if desired
        if param.get_value('save'):
            log.info('')
            if cube:
                # save to different product type
                outname = outname.replace('_COA_', '_SCB_')
                hdul[0].header['PRODTYPE'] = 'spectral_cube'
                hdul[0].header['PROCSTAT'] = 'LEVEL_4'
                hdul[0].header['FILENAME'] = os.path.basename(outname)
                log.info('Full product (3D spectral cube):')
                self.write_output(hdul, outname)
            else:
                # save 1D Spextool-style final products
                log.info('1D spectra:')
                spec = self._make_1d(hdul)
                specname = outname.replace('_COA_', '_CMB_')
                spec[0].header['FILENAME'] = os.path.basename(specname)
                spec[0].header['PRODTYPE'] = 'combined_spectrum'
                self.write_output(spec, specname)

                # also save full product
                log.info('')
                log.info('Full product (2D images and 1D spectra):')
                self.write_output(hdul, outname)

        self.input = [hdul]
        self.set_display_data()

    def make_response(self):
        """Make spectral response curve."""
        from sofia_redux.instruments.forcast.getmodel import get_model

        # get parameters
        param = self.get_parameter_set()
        save = param.get_value('save')
        modelfile = param.get_value('model_file')
        if str(modelfile).strip() == '':
            modelfile = None

        results = []
        filenames = []
        for i, hdul in enumerate(self.input):
            log.info('')
            log.info(hdul[0].header['FILENAME'])

            header = hdul[0].header
            wave = hdul['WAVEPOS'].data
            sflux = hdul['SPECTRAL_FLUX'].data
            serr = hdul['SPECTRAL_ERROR'].data
            if 'RESPONSE' in hdul:
                resp = hdul['RESPONSE'].data
                resp_err = hdul['RESPONSE_ERROR'].data
            else:
                log.warning('No response extension found.')
                resp = np.ones_like(sflux)
                resp_err = np.zeros_like(sflux)

            # get smoothed model and interpolate onto wavelengths
            if 'RP' in header:
                resolution = header['RP']
            else:
                resolution = self.calres['resolution']
            model_dir = os.path.join(self.calres['pathcal'], 'grism',
                                     'standard_models')
            log.debug(f'Model directory: {model_dir}')
            model = get_model(header, resolution, filename=modelfile,
                              model_dir=model_dir)
            if model is None:
                msg = 'Cannot create response file.'
                log.error(msg)
                raise ValueError(msg)

            mwave, mdata = model
            mmatch = np.interp(wave, mwave, mdata,
                               left=np.nan, right=np.nan)

            # un-add response error to flux error
            serr = np.sqrt((resp * serr)**2 - (resp_err * sflux)**2)

            # multiply old response back out
            sflux *= resp

            # divide by standard model
            hdul['SPECTRAL_FLUX'].data = sflux / mmatch
            hdul['SPECTRAL_ERROR'].data = serr / mmatch

            # store model as response so it will be inserted in the
            # correct row in the 1D product
            # (full product is not saved for this step)
            if 'RESPONSE' in hdul:
                hdul['RESPONSE'].data = mmatch
            else:
                hdul.append(fits.ImageHDU(mmatch, name='RESPONSE'))

            # update BUNIT for spectral flux
            runit = header.get('RAWUNITS', 'Me/s')
            hdul['SPECTRAL_FLUX'].header['BUNIT'] = f'{runit}/Jy'

            hdul = self._make_1d(hdul)
            results.append(hdul)

            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])

            # save if desired
            if save:
                self.write_output(hdul, outname)
                filenames.append(outname)

        log.info('')
        self.input = results
        self.set_display_data(filenames=filenames)

    def combine_response(self):
        """Scale, combine, and smooth instrument response."""
        from sofia_redux.instruments.forcast.hdmerge import hdmerge
        from sofia_redux.spectroscopy.getspecscale import getspecscale
        from sofia_redux.toolkit.image.combine import combine_images

        # get parameters
        param = self.get_parameter_set()
        method = param.get_value('method')
        weighted = param.get_value('weighted')
        robust = param.get_value('robust')
        sigma = param.get_value('threshold')
        maxiters = param.get_value('maxiters')
        scale_method = param.get_value('scale_method')
        scale_index = param.get_value('scale_index')
        fwhm = param.get_value('fwhm')
        combine_aps = param.get_value('combine_aps')

        # read in 1D response spectra
        hdr_list = []
        spec_list = []
        spec_var_list = []
        trans_list = []
        test_wave = None
        for i, hdul in enumerate(self.input):
            if hdul[0].data.ndim > 2:
                n_ap = hdul[0].data.shape[0]
                spec_wave = hdul[0].data[:, 0]
                spec_flux = hdul[0].data[:, 1]
                spec_var = hdul[0].data[:, 2] ** 2
                spec_tran = hdul[0].data[:, 3]
            else:
                n_ap = 1
                spec_wave = hdul[0].data[0]
                spec_flux = hdul[0].data[1]
                spec_var = hdul[0].data[2] ** 2
                spec_tran = hdul[0].data[3]
            if i == 0:
                if combine_aps and n_ap > 1:
                    test_wave = spec_wave[0]
                else:
                    test_wave = spec_wave
            else:
                if not np.allclose(spec_wave, test_wave):
                    msg = 'Mismatched wavelengths. Spectra cannot be combined.'
                    log.error(msg)
                    raise ValueError(msg)

            hdr_list.append(hdul[0].header)
            if combine_aps and n_ap > 1:
                for j in range(n_ap):
                    spec_list.append(spec_flux[j])
                    spec_var_list.append(spec_var[j])
                    trans_list.append(spec_tran[j])
            else:
                spec_list.append(spec_flux)
                spec_var_list.append(spec_var)
                trans_list.append(spec_tran)

        nspec = len(spec_list)
        spec_list = np.array(spec_list)
        spec_var_list = np.array(spec_var_list)

        # scale spectra as needed
        if scale_method != 'none' and nspec > 1:
            log.info('')
            log.info(f'Scaling spectra with method {scale_method}')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                medflux = np.nanmedian(spec_list, axis=1)
            if scale_method == 'highest':
                refidx = np.argmax(medflux)
            elif scale_method == 'lowest':
                refidx = np.argmin(medflux)
            elif scale_method == 'index':
                refidx = scale_index
            else:
                refidx = None

            scale = getspecscale(spec_list, refidx=refidx)
            log.info('Spectral scales: {}'.format(
                ', '.join(['{:.2f}'.format(s) for s in scale])))
            if spec_list.ndim > 2:
                spec_list *= scale[:, None, None]
                spec_var_list *= scale[:, None, None] ** 2
            else:
                spec_list *= scale[:, None]
                spec_var_list *= scale[:, None] ** 2

        # combine data
        if nspec > 1:
            log.info('')
            log.info(f'Combining spectra with method {method}')
            outdata, outvar = combine_images(spec_list,
                                             variance=spec_var_list,
                                             method=method, weighted=weighted,
                                             robust=robust, sigma=sigma,
                                             maxiters=maxiters)
            outtrans, _ = combine_images(trans_list, method=method,
                                         robust=False)
            outhdr = hdmerge(hdr_list, reference_header=hdr_list[0])
        else:
            outdata = spec_list[0]
            outvar = spec_var_list[0]
            outtrans = trans_list[0]
            outhdr = hdr_list[0]

        # check for multiple apertures
        n_ap = 1
        if outdata.ndim > 1:
            n_ap = outdata.shape[0]

        # smooth spectra as needed
        if fwhm > 0:
            log.info('')
            log.info(f'Smoothing output with Gaussian FHWM {fwhm}')
            sigma = gaussian_fwhm_to_sigma * fwhm
            kernel = Gaussian1DKernel(stddev=sigma)
            if n_ap > 1:
                for j in range(n_ap):
                    outdata[j] = convolve(outdata[j], kernel,
                                          fill_value=np.nan,
                                          preserve_nan=True)
                    outvar[j] = convolve(outvar[j], kernel,
                                         fill_value=np.nan,
                                         preserve_nan=True)
            else:
                outdata = convolve(outdata, kernel, fill_value=np.nan,
                                   preserve_nan=True)
                outvar = convolve(outvar, kernel, fill_value=np.nan,
                                  preserve_nan=True)

        # make output
        if n_ap > 1:
            spectrum = np.empty((n_ap, 4, outdata.shape[1]))
            spectrum[:, 0] = test_wave
            spectrum[:, 1] = outdata
            spectrum[:, 2] = np.sqrt(outvar)
            spectrum[:, 3] = outtrans
        else:
            spectrum = np.array([test_wave, outdata,
                                 np.sqrt(outvar), outtrans])

        hdul = fits.HDUList(fits.PrimaryHDU(spectrum, header=outhdr))
        outname = self.update_output(hdul, self.filenum,
                                     self.prodtypes[self.step_index])

        # update procstat to level 4
        hdul[0].header['PROCSTAT'] = 'LEVEL_4'

        # save if desired
        log.info('')
        if param.get_value('save'):
            self.write_output(hdul, outname)
            filenames = [outname]
        else:
            filenames = None

        self.input = [hdul]
        self.set_display_data(filenames=filenames)

    def specmap(self):
        """
        Generate a quick-look image and spectral plot.

        Calls `sofia_redux.visualization.quicklook.make_image`.

        The output from this step is identical to the input, so is
        not saved.  As a side effect, a PNG file is saved to disk to the
        same base name as the input file, with a '.png' extension.
        """
        from astropy import units as u
        from astropy.coordinates import Angle
        from scipy.ndimage import gaussian_filter

        from sofia_redux.visualization.quicklook \
            import make_image, make_spectral_plot

        # get parameters
        param = self.get_parameter_set()
        colormap = param.get_value('colormap')
        scale = param.get_value('scale')
        n_contour = param.get_value('n_contour')
        contour_color = param.get_value('contour_color')
        fill_contours = param.get_value('fill_contours')
        grid = param.get_value('grid')
        watermark = param.get_value('watermark')
        override_slice = param.get_value('override_slice')
        override_point = param.get_value('override_point')
        ignore_outer = param.get_value('ignore_outer')
        atran_plot = param.get_value('atran_plot')
        error_plot = param.get_value('error_plot')
        spec_scale = param.get_value('spec_scale')

        # format override parameters
        override_slice = re.sub(r'[\s+\[\]\'\"]', '',
                                str(override_slice).lower())
        if override_slice not in ['', 'none']:
            try:
                override_w = int(override_slice)
            except ValueError:
                raise ValueError(f'Bad input for parameter '
                                 f'override_slice ({override_slice}). '
                                 f'Specify as an integer.') from None
        else:
            override_w = None
        override_point = re.sub(r'[\s+\[\]\'\"]', '',
                                str(override_point).lower())
        if override_point not in ['', 'none']:
            try:
                override_xy = [int(f) for f in override_point.split(',')]
                assert len(override_xy) == 2
            except (ValueError, IndexError, AssertionError):
                raise ValueError(f'Bad input for parameter '
                                 f'override_point ({override_point}). '
                                 f'Specify as 2 comma-separated integers '
                                 f'(x, y).') from None
        else:
            override_xy = None

        # set plot scale to default, if it is full range
        if spec_scale[0] <= 0 and spec_scale[1] >= 100:
            spec_scale = None

        for i, hdul in enumerate(self.input):
            header = hdul[0].header
            one_d = False
            labels = None
            aplot = None

            try:
                wave = hdul['WAVEPOS'].data
                xunit = hdul['WAVEPOS'].header.get('BUNIT', 'um')
            except KeyError:
                # 1D spectrum
                one_d = True
                if hdul[0].data.ndim > 2:
                    # multi-order or multi-ap
                    wave = hdul[0].data[:, 0]
                    nspec = hdul[0].data.shape[0]
                    labels = [f'Spectrum {j + 1}' for j in range(nspec)]
                else:
                    wave = hdul[0].data[0]
                xunit = hdul[0].header.get('XUNITS', 'UNKNOWN')

            # set text for title and subtitle in plot
            obj = header.get('OBJECT', 'UNKNOWN')
            spectel = self.calres['spectel']
            basename = os.path.basename(header.get('FILENAME', 'UNKNOWN'))
            subtitle = f'Filename: {basename}'

            # if cube, grab the brightest slice
            if header.get('PRODTYPE', 'UNKNOWN').lower() == 'spectral_cube':
                cube = True
                flux = hdul['FLUX'].data
                err = hdul['ERROR'].data
                atran = hdul['TRANSMISSION'].data

                if override_xy is not None:
                    x, y = override_xy
                else:
                    # smooth over a couple pixels with a Gaussian filter
                    sigma = 2.0
                    sflux = gaussian_filter(flux, sigma=sigma,
                                            mode='constant', cval=np.nan,
                                            truncate=2)
                    serr = gaussian_filter(err, sigma=sigma,
                                           mode='constant', cval=np.nan,
                                           truncate=2)

                    try:
                        img_peak = np.unravel_index(np.nanargmax(sflux / serr),
                                                    flux.shape)[1:]
                    except ValueError:
                        img_peak = flux.shape[1] / 2, flux.shape[2] / 2

                    y, x = [int(i) for i in img_peak]
                spec_flux = flux[:, y, x]
                spec_err = err[:, y, x]

                if override_w is not None:
                    spec_peak = override_w
                else:
                    # peak in spectrum at selected spatial point
                    spec_peak = int(np.nanargmax(spec_flux))
                spec_wv = wave[spec_peak]
                subsubtitle = f'Wavelength slice at {spec_wv:.2f} {xunit}'

                hwcs = WCS(header)
                ra, dec, wv = hwcs.wcs_pix2world(x, y, spec_peak, 0)
                yunit = hdul['FLUX'].header.get('BUNIT', 'UNKNOWN')

                ra_str = Angle(ra, unit=u.deg).to_string(unit=u.hour)
                dec_str = Angle(dec, unit=u.deg).to_string(unit=u.deg)
                stitle = f'Spectrum at RA={ra_str}, Dec={dec_str}'
                marker = [spec_wv, spec_flux[spec_peak]]

                log.debug(f'Spectral slice at w={spec_peak}')
                log.debug(f'Spatial point at x={x}, y={y}')
            else:
                cube = False
                subsubtitle = ''
                flux, spec_peak, spec_wv, x, y, stitle, marker = \
                    None, None, None, None, None, None, None
                if not one_d:
                    spec_flux = hdul['SPECTRAL_FLUX'].data
                    spec_err = hdul['SPECTRAL_ERROR'].data
                    atran = hdul['TRANSMISSION'].data
                    yunit = hdul['SPECTRAL_FLUX'].header.get('BUNIT',
                                                             'UNKNOWN')
                    # check for multiple apertures in new-style data
                    if spec_flux.ndim > 1 and wave.ndim == 1:
                        nspec = spec_flux.shape[0]
                        wave = np.tile(wave, (nspec, 1))
                        labels = [f'Spectrum {j + 1}' for j in range(nspec)]
                else:
                    if hdul[0].data.ndim > 2:
                        spec_flux = hdul[0].data[:, 1]
                        spec_err = hdul[0].data[:, 2]
                        try:
                            atran = hdul[0].data[:, 3]
                        except IndexError:  # pragma: no cover
                            # Old data may not have atran extension
                            atran = np.full_like(spec_flux, np.nan)
                    else:
                        spec_flux = hdul[0].data[1]
                        spec_err = hdul[0].data[2]
                        try:
                            atran = hdul[0].data[3]
                        except IndexError:  # pragma: no cover
                            atran = np.full_like(spec_flux, np.nan)
                    yunit = hdul[0].header.get('YUNITS', 'UNKNOWN')
                if ignore_outer > 0:
                    # set the outer N% of frames to NaN
                    if spec_flux.ndim > 1:
                        wstart = int(ignore_outer * wave.shape[1])
                        wend = int((1 - ignore_outer) * wave.shape[1])
                        spec_flux[:, :wstart] = np.nan
                        spec_flux[:, wend:] = np.nan
                    else:
                        wstart = int(ignore_outer * len(wave))
                        wend = int((1 - ignore_outer) * len(wave))
                        spec_flux[:wstart] = np.nan
                        spec_flux[wend:] = np.nan
                    atran[:wstart] = np.nan
                    atran[wend:] = np.nan
                    log.debug(f'Plotting between w={wstart} and w={wend}')

            # make the image figure
            title = f'Object: {obj}, Grism: {spectel}'
            if not one_d:
                fig = make_image(hdul, colormap=colormap,
                                 scale=scale, n_contour=n_contour,
                                 contour_color=contour_color,
                                 fill_contours=fill_contours, title=title,
                                 subtitle=subtitle, subsubtitle=subsubtitle,
                                 grid=grid, beam=False,
                                 plot_layout=(2, 1), cube_slice=spec_peak,
                                 watermark=watermark)
                ax1 = fig.axes[0]
                # add a spectral plot
                ax = fig.add_subplot(2, 1, 2)
            else:
                # just make the spectral plot
                fig = Figure(figsize=(8, 5))
                FigureCanvas(fig)
                ax1 = None
                ax = fig.add_subplot(1, 1, 1)
                ax.set_title(title)

            if atran_plot and not np.all(np.isnan(atran)):
                if wave.ndim > 1:
                    aplot = [wave.T, atran.T]
                else:
                    aplot = [wave, atran]

            if not error_plot:
                spec_err = None

            # plot spectral flux
            make_spectral_plot(ax, wave, spec_flux, spectral_error=spec_err,
                               scale=spec_scale, labels=labels,
                               colormap=colormap,
                               xunit=xunit, yunit=yunit,
                               title=stitle, marker=marker,
                               marker_color=contour_color,
                               overplot=aplot,
                               overplot_label='Atmospheric Transmission',
                               overplot_color=contour_color,
                               watermark=watermark)

            # mark the spectrum position on the image
            if cube:
                ax1.scatter(x, y, c=contour_color, marker='x', s=20,
                            alpha=0.8)

            # output filename for image
            fname = os.path.splitext(basename)[0] + '.png'
            outname = os.path.join(self.output_directory, fname)

            fig.savefig(outname, dpi=300)
            log.info(f'Saved image to {outname}')
