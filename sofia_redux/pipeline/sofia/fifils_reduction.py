# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FIFI-LS Reduction pipeline steps"""

import os
import re
import warnings

from astropy import log
from astropy.io import fits
from astropy.stats import gaussian_fwhm_to_sigma
import numpy as np
import psutil

from sofia_redux.pipeline.sofia.sofia_exception import SOFIAImportError
try:
    from sofia_redux.instruments import fifi_ls
except ImportError:
    raise SOFIAImportError('FIFI-LS modules not installed')

from sofia_redux.pipeline.gui.qad_viewer import QADViewer
from sofia_redux.pipeline.reduction import Reduction
from sofia_redux.pipeline.sofia.parameters.fifils_parameters \
    import FIFILSParameters
from sofia_redux.toolkit.utilities.fits import get_key_value, getheader, \
    write_hdul, gethdul, hdinsert

# this import is not used here, but is needed to avoid
# a numba bug on Linux systems
from sofia_redux.toolkit.resampling import resample_utils
from sofia_redux.spectroscopy import smoothres
assert resample_utils
assert smoothres


class HeaderValidationError(ValueError):
    pass


class FIFILSReduction(Reduction):
    """
    FIFI-LS reduction steps.

    Primary image reduction algorithms are defined in the FIFI-LS
    package (`fifi-ls`).  Some utilities come from the `sofia_redux.toolkit`
    package or the `sofia_redux.spectroscopy` package.  This reduction object
    requires that all three packages be installed.

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
    """
    def __init__(self):
        """Initialize the reduction object."""
        super().__init__()

        # descriptive attributes
        self.name = 'Redux'
        self.instrument = 'FIFI-LS'
        self.mode = 'IFS'
        self.data_keys = ['File Name', 'OBJECT', 'OBSTYPE',
                          'AOR_ID', 'MISSN-ID', 'DATE-OBS',
                          'OBS_ID', 'INSTCFG', 'INSTMODE',
                          'NODSTYLE', 'DETCHAN',
                          'DICHROIC', 'G_ORD_B',
                          'SPECTEL1', 'SPECTEL2',
                          'ALTI_STA', 'ZA_START',
                          'EXPTIME', 'NODBEAM']

        self.pipe_name = "FIFI_LS_REDUX"
        self.pipe_version = fifi_ls.__version__.replace('.', '_')

        # parameters for the steps: will be set on load
        self.parameters = None

        # product type definitions for FIFI-LS steps
        self.prodtype_map = {'checkhead': 'raw',
                             'split_grating_and_chop': 'grating_chop_split',
                             'fit_ramps': 'ramps_fit',
                             'subtract_chops': 'chop_subtracted',
                             'combine_nods': 'nod_combined',
                             'lambda_calibrate': 'wavelength_calibrated',
                             'spatial_calibrate': 'spatial_calibrated',
                             'apply_static_flat': 'flat_fielded',
                             'combine_grating_scans': 'scan_combined',
                             'telluric_correct': 'telluric_corrected',
                             'flux_calibrate': 'flux_calibrated',
                             'correct_wave_shift': 'wavelength_shifted',
                             'resample': 'resampled',
                             'specmap': 'specmap'
                             }

        # invert the map for quick lookup of step from type
        self.step_map = {v: k for k, v in self.prodtype_map.items()}

        # this will be populated when the recipe is set
        self.prodtypes = []

        # product name codes for each step
        self.prodnames = {'split_grating_and_chop': 'CP0',
                          'fit_ramps': 'RP0',
                          'subtract_chops': 'CSB',
                          'combine_nods': 'NCM',
                          'lambda_calibrate': 'WAV',
                          'spatial_calibrate': 'XYC',
                          'apply_static_flat': 'FLF',
                          'combine_grating_scans': 'SCM',
                          'telluric_correct': 'TEL',
                          'flux_calibrate': 'CAL',
                          'correct_wave_shift': 'WSH',
                          'resample': 'WXY',
                          'specmap': 'SMP'
                          }

        # default recipe and step names
        self.recipe = [
            'checkhead', 'split_grating_and_chop',
            'fit_ramps', 'subtract_chops', 'combine_nods',
            'lambda_calibrate', 'spatial_calibrate',
            'apply_static_flat', 'combine_grating_scans',
            'telluric_correct', 'flux_calibrate',
            'correct_wave_shift', 'resample', 'specmap']
        self.processing_steps = {
            'checkhead': 'Check Headers',
            'split_grating_and_chop': 'Split Grating/Chop',
            'fit_ramps': 'Fit Ramps',
            'subtract_chops': 'Subtract Chops',
            'combine_nods': 'Combine Nods',
            'lambda_calibrate': 'Lambda Calibrate',
            'spatial_calibrate': 'Spatial Calibrate',
            'apply_static_flat': 'Apply Flat',
            'combine_grating_scans': 'Combine Scans',
            'telluric_correct': 'Telluric Correct',
            'flux_calibrate': 'Flux Calibrate',
            'correct_wave_shift': 'Correct Wave Shift',
            'resample': 'Resample',
            'specmap': 'Make Spectral Map'}

        # reduction information
        self.check_input = True
        self.output_directory = os.getcwd()
        self.atran = None
        self.response = None

        # set up for potential parallel processing via joblib
        self.max_cores = psutil.cpu_count() // 2
        if self.max_cores < 2:
            self.max_cores = None

    def __setattr__(self, name, value):
        """Check if recipe is being set. If so, also set prodtypes."""
        if name == 'recipe':
            # set the prodtype list
            try:
                self.prodtypes = [self.prodtype_map[step] for step in value]
            except AttributeError:
                self.prodtypes = []
        super().__setattr__(name, value)

    def set_display_data(self, raw=False):
        """
        Store display data for QAD Viewer.

        Parameters
        ----------
        raw : bool
            If True, display data is taken from self.rawfiles.
            If False, display data is taken from self.input
        """
        if raw:
            data_list = [[hdu.header for hdu in hdul] for hdul in self.input]
        else:
            data_list = [gethdul(hdul) for hdul in self.input]
        self.display_data = {'QADViewer': data_list}

    def update_output(self, hdul, prodtype):
        """
        Update output FITS file after a pipeline step.

        Adds a HISTORY message with the pipeline step name and parameters.

        Parameters
        ----------
        hdul : `astropy.io.fits.HDUList`
           Output FITS HDUList.
        """
        # add a history message with step name
        step_name = self.step_map[prodtype]
        step_display_name = self.processing_steps[step_name]
        msg = '-- Pipeline step: {}'.format(step_display_name)
        hdinsert(hdul[0].header, 'HISTORY', msg)

        hdinsert(hdul[0].header, 'HISTORY', 'Parameters:')
        params = self.get_parameter_set()
        for param in params:
            if params[param]['hidden']:
                # don't record hidden parameters
                continue
            msg = "  {} = {}".format(param, params.get_value(param))
            hdinsert(hdul[0].header, 'HISTORY', msg)

        hdinsert(hdul[0].header, 'HISTORY', '--')
        hdinsert(hdul[0].header, 'HISTORY', '')

    def write_output(self, hdul):
        """
        Write an output FITS file to disk.

        Outname is joined to self.output_directory, before writing.

        Parameters
        ----------
        hdul : `astropy.io.fits.HDUList`
           FITS HDUList to write.

        Returns
        -------
        str
            Full path to the output file.
        """
        hdul = gethdul(hdul)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            outname = write_hdul(hdul, outdir=self.output_directory)
        self.record_outfile(outname)
        return outname

    def load(self, data):
        """
        Load input data to make it available to reduction steps.

        The process is:

        - Call the parent load method to initialize data
          reduction variables.
        - Use the product type in the first FITS header to determine
          the data processing recipe.
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
        """
        # call the parent method to initialize
        # reduction variables
        super().load(data)

        # read and save the first FITS header
        basehead = getheader(self.raw_files[0])

        # get product type to determine recipe
        intermediate = False
        prodtype = get_key_value(basehead, 'PRODTYPE').lower()

        # translate an old prodtype, for backward compatibility
        if prodtype == 'wxy_resampled':  # pragma: no cover
            prodtype = 'resampled'

        # get remaining recipe
        if prodtype in self.prodtypes:
            pidx = self.prodtypes.index(prodtype)
            self.recipe = self.recipe[pidx + 1:]
            if len(self.recipe) == 0:
                msg = "No steps to run for prodtype '{}'.".format(prodtype)
                log.error(msg)
                raise ValueError(msg)
            intermediate = True
        elif prodtype.upper() != 'UNKNOWN':
            # necessary for pipeline resets: prodtype
            # is not in pre-modified recipe, but file should
            # be loaded in immediately
            intermediate = True

        # initialize parameters
        self.parameters = FIFILSParameters(basehead=basehead)

        # if not starting from raw data, load the files in
        # immediately
        if intermediate:
            self.load_fits()
        else:
            # just load headers
            self.input = []
            for datafile in self.raw_files:
                try:
                    hdr = fits.getheader(datafile)
                except (IndexError, ValueError, TypeError, OSError):
                    log.warning('Header could not be read '
                                'from {}'.format(os.path.basename(datafile)))
                    hdr = fits.Header()
                self.input.append(hdr)

        # log all input files
        for datafile in self.raw_files:
            log.info("Input: {}".format(datafile))

        # log all processing steps
        log.info('Processing steps: {}'.format(self.recipe))

    def load_fits(self, check_header=False):
        """
        Load FITS data into memory.

        Handles raw data, as well as intermediate data.

        Loaded data are stored in the input attribute.
        """
        from sofia_redux.instruments.fifi_ls.readfits import readfits

        self.input = []
        bad_header = ''
        for i, datafile in enumerate(self.raw_files):
            log.debug('Reading input: {}'.format(datafile))

            # read the fits file and standardize/check header
            if check_header:
                input_hdul, success = readfits(datafile,
                                               checkheader=check_header)
            else:
                input_hdul = readfits(datafile,
                                      checkheader=check_header)
                success = True
            if not success:
                bad_header = 'Invalid headers'
                log.error(bad_header)
            if input_hdul is None:
                msg = 'Unable to read FITS file'
                log.error(msg)
                raise ValueError(msg)

            # touch the data to make sure it is in memory
            # and store copies in a new HDUList
            # (this avoids storing a file handle in the HDUList)
            hdul = fits.HDUList()
            for hdu in input_hdul:
                assert hdu.data is None or hdu.data[0] is not None
                hdul.append(hdu.copy())
            input_hdul.close()

            self.input.append(hdul)

        # store the data in display variables
        self.set_display_data(raw=True)

        # raise error for checkhead to catch and handle
        if bad_header != '':
            raise HeaderValidationError(bad_header)

    def register_viewers(self):
        """Return a new QADViewer."""
        viewers = [QADViewer()]
        return viewers

    def checkhead(self):
        """
        Check input headers.

        Compares header keywords to requirements.  Halts reduction if the
        abort parameter is True and the headers do not meet requiremnts.
        """
        # get parameters
        param = self.get_parameter_set()

        # read fits files into memory for processing
        # with header checking
        try:
            self.load_fits(check_header=True)
        except HeaderValidationError:
            if param.get_value('abort'):
                msg = 'Invalid headers.'
                log.error(msg)
                self.error = msg
                self.input = []
                return

    def split_grating_and_chop(self):
        """Split data into separate files/extensions by chop/grating."""
        from sofia_redux.instruments.fifi_ls.split_grating_and_chop \
            import wrap_split_grating_and_chop

        # get parameters
        param = self.get_parameter_set()
        save = param.get_value('save')
        parallel = param.get_value('parallel')
        if parallel:
            jobs = self.max_cores
        else:
            jobs = None

        result = wrap_split_grating_and_chop(self.input, write=False,
                                             jobs=jobs,
                                             allow_errors=True)
        if not result:
            msg = 'Problem in fifi_ls.split_grating_and_chop.'
            log.error(msg)
            self.error = msg
            self.input = []
            return
        for hdul in result:
            self.update_output(hdul, self.prodtypes[self.step_index])
        self.input = list(result)
        self.set_display_data()
        if save:
            for hdul in self.input:
                self.write_output(hdul)

    def fit_ramps(self):
        """Fit voltage ramps to derive average flux from samples."""
        from sofia_redux.instruments.fifi_ls.fit_ramps import wrap_fit_ramps

        # get parameters
        param = self.get_parameter_set()
        save = param.get_value('save')
        parallel = param.get_value('parallel')
        s2n = param.get_value('s2n')
        thresh = param.get_value('thresh')
        remove = param.get_value('remove_first')
        subtract_bias = param.get_value('subtract_bias')
        indpos_sigma = param.get_value('indpos_sigma')
        badpix_file = param.get_value('badpix_file')
        if str(badpix_file).strip() == '':
            badpix_file = None
        if parallel:
            jobs = self.max_cores
        else:
            jobs = None

        result = wrap_fit_ramps(self.input, write=False, jobs=jobs,
                                allow_errors=True, s2n=s2n,
                                threshold=thresh, remove_first=remove,
                                subtract_bias=subtract_bias,
                                indpos_sigma=indpos_sigma,
                                badpix_file=badpix_file)
        if not result:
            msg = 'Problem in fifi_ls.fit_ramps.'
            log.error(msg)
            self.error = msg
            self.input = []
            return
        for hdul in result:
            self.update_output(hdul, self.prodtypes[self.step_index])
        self.input = list(result)
        self.set_display_data()
        if save:
            for hdul in self.input:
                self.write_output(hdul)

    def subtract_chops(self):
        """Subtract chop pairs for sky background removal."""
        from sofia_redux.instruments.fifi_ls.subtract_chops \
            import wrap_subtract_chops

        # get parameters
        param = self.get_parameter_set()
        save = param.get_value('save')
        parallel = param.get_value('parallel')
        if parallel:
            jobs = self.max_cores
        else:
            jobs = None

        # rearrange input into chop pairs by obsid
        pairs = {}
        for hdul in self.input:
            obsid = get_key_value(hdul[0].header, 'OBS_ID')
            if obsid in pairs:
                pairs[obsid].append(hdul)
            else:
                pairs[obsid] = [hdul]
        input_list = list(pairs.values())

        result = wrap_subtract_chops(input_list, write=False,
                                     jobs=jobs,
                                     allow_errors=True)
        if not result:
            msg = 'Problem in fifi_ls.subtract_chops.'
            log.error(msg)
            self.error = msg
            self.input = []
            return
        for hdul in result:
            self.update_output(hdul, self.prodtypes[self.step_index])
        self.input = list(result)
        self.set_display_data()
        if save:
            for hdul in self.input:
                self.write_output(hdul)

    def combine_nods(self):
        """Combine nod pairs."""
        from sofia_redux.instruments.fifi_ls.combine_nods import combine_nods

        # get parameters
        param = self.get_parameter_set()
        save = param.get_value('save')
        offbeam = param.get_value('offbeam')
        b_nod_method = param.get_value('b_nod_method')

        # this function returns a dataframe as the result
        result = combine_nods(self.input, write=False,
                              offbeam=offbeam, b_nod_method=b_nod_method)
        if result is None or result.empty:
            msg = 'Problem in fifi_ls.combine_nods.'
            log.error(msg)
            self.error = msg
            self.input = []
            return

        # extract the A beams, and check for a combined file
        abeams = result[result['nodbeam'] == 'A']
        chdul = abeams['chdul']

        if not chdul.isnull().all():
            log.info('Combined A and B nods.')
            # propagate combined hduls
            self.input = list(chdul[chdul.notnull()])
            if len(self.input) < len(abeams):
                log.warning('{}/{} A nods did not find B nods; '
                            'these will not be propagated.'.format(
                                len(abeams) - len(self.input),
                                len(abeams)))
        elif len(abeams) > 0:
            # leave self.input as is
            log.info('No B nods found; propagating A nods')
        else:
            self.input = []
            msg = 'No A nods found. Use offbeam=True if B ' \
                  'nod propagation is desired.'
            log.error(msg)
            self.error = msg
            return

        for hdul in self.input:
            self.update_output(hdul, self.prodtypes[self.step_index])
        self.set_display_data()
        if save:
            for hdul in self.input:
                self.write_output(hdul)

    def lambda_calibrate(self):
        """Calibrate wavelengths."""
        from sofia_redux.instruments.fifi_ls.lambda_calibrate \
            import wrap_lambda_calibrate

        # get parameters
        param = self.get_parameter_set()
        save = param.get_value('save')
        parallel = param.get_value('parallel')
        if parallel:
            jobs = self.max_cores
        else:
            jobs = None

        result = wrap_lambda_calibrate(self.input, write=False,
                                       jobs=jobs, allow_errors=True)
        if not result:
            msg = 'Problem in fifi_ls.lambda_calibrate.'
            log.error(msg)
            self.error = msg
            self.input = []
            return
        for hdul in result:
            self.update_output(hdul, self.prodtypes[self.step_index])
        self.input = list(result)
        self.set_display_data()
        if save:
            for hdul in self.input:
                self.write_output(hdul)

    def spatial_calibrate(self):
        """Calibrate pixel positions."""
        from sofia_redux.instruments.fifi_ls.spatial_calibrate \
            import wrap_spatial_calibrate

        # get parameters
        param = self.get_parameter_set()
        save = param.get_value('save')
        parallel = param.get_value('parallel')
        rotate = param.get_value('rotate')
        flipsign = param.get_value('flipsign')
        if flipsign == 'flip':
            flipsign = True
        elif flipsign == 'no flip':
            flipsign = False
        else:
            flipsign = None
        if parallel:
            jobs = self.max_cores
        else:
            jobs = None

        result = wrap_spatial_calibrate(self.input, write=False,
                                        jobs=jobs, allow_errors=True,
                                        rotate=rotate, flipsign=flipsign)
        if not result:
            msg = 'Problem in fifi_ls.spatial_calibrate.'
            log.error(msg)
            self.error = msg
            self.input = []
            return
        for hdul in result:
            self.update_output(hdul, self.prodtypes[self.step_index])
        self.input = list(result)
        self.set_display_data()
        if save:
            for hdul in self.input:
                self.write_output(hdul)

    def apply_static_flat(self):
        """Apply flat correction."""
        from sofia_redux.instruments.fifi_ls.apply_static_flat \
            import wrap_apply_static_flat

        # get parameters
        param = self.get_parameter_set()
        save = param.get_value('save')
        parallel = param.get_value('parallel')
        skip_flat = param.get_value('skip_flat')
        skip_err = param.get_value('skip_err')
        if parallel:
            jobs = self.max_cores
        else:
            jobs = None

        if skip_flat:
            log.info('No flat correction performed.')
            return

        result = wrap_apply_static_flat(self.input, write=False,
                                        skip_err=skip_err,
                                        jobs=jobs, allow_errors=True)
        if not result:
            msg = 'Problem in fifi_ls.apply_static_flat.'
            log.error(msg)
            self.error = msg
            self.input = []
            return
        for hdul in result:
            self.update_output(hdul, self.prodtypes[self.step_index])
        self.input = list(result)
        self.set_display_data()
        if save:
            for hdul in self.input:
                self.write_output(hdul)

    def combine_grating_scans(self):
        """Combine grating scans."""
        from sofia_redux.instruments.fifi_ls.combine_grating_scans \
            import wrap_combine_grating_scans

        # get parameters
        param = self.get_parameter_set()
        save = param.get_value('save')
        parallel = param.get_value('parallel')
        bias = param.get_value('bias')
        if parallel:
            jobs = self.max_cores
        else:
            jobs = None

        result = wrap_combine_grating_scans(self.input, write=False,
                                            jobs=jobs,
                                            allow_errors=True,
                                            correct_bias=bias)
        if not result:
            msg = 'Problem in fifi_ls.combine_grating_scans.'
            log.error(msg)
            self.error = msg
            self.input = []
            return
        for hdul in result:
            self.update_output(hdul, self.prodtypes[self.step_index])
        self.input = list(result)
        self.set_display_data()
        if save:
            for hdul in self.input:
                self.write_output(hdul)

    def telluric_correct(self):
        """Apply telluric correction."""
        from sofia_redux.instruments.fifi_ls.telluric_correct \
            import wrap_telluric_correct

        # get parameters
        param = self.get_parameter_set()
        save = param.get_value('save')
        parallel = param.get_value('parallel')
        skip_tell = param.get_value('skip_tell')
        cutoff = param.get_value('cutoff')
        atran_dir = param.get_value('atran_dir')
        use_wv = param.get_value('use_wv')
        if parallel:
            jobs = self.max_cores
        else:
            jobs = None

        if str(atran_dir).strip() == '':
            atran_dir = None

        if skip_tell:
            log.info('ATRAN file is attached, but no correction performed.')

        result = wrap_telluric_correct(self.input, write=False,
                                       jobs=jobs, allow_errors=True,
                                       atran_dir=atran_dir, cutoff=cutoff,
                                       use_wv=use_wv, skip_corr=skip_tell)
        if not result:
            msg = 'Problem in fifi_ls.telluric_correct.'
            log.error(msg)
            self.error = msg
            self.input = []
            return
        for hdul in result:
            self.update_output(hdul, self.prodtypes[self.step_index])
        self.input = list(result)
        self.set_display_data()
        if save:
            for hdul in self.input:
                self.write_output(hdul)

    def flux_calibrate(self):
        """Calibrate flux to physical units."""
        from sofia_redux.instruments.fifi_ls.flux_calibrate \
            import wrap_flux_calibrate

        # get parameters
        param = self.get_parameter_set()
        save = param.get_value('save')
        parallel = param.get_value('parallel')
        skip_cal = param.get_value('skip_cal')
        response = param.get_value('response_file')
        if parallel:
            jobs = self.max_cores
        else:
            jobs = None
        if skip_cal:
            log.info('No flux calibration performed.')
            return

        if response.strip() == '':
            response = None
        self.response = response

        result = wrap_flux_calibrate(self.input, write=False,
                                     jobs=jobs,
                                     allow_errors=True,
                                     response_file=response)
        if not result:
            msg = 'Problem in fifi_ls.flux_calibrate.'
            log.error(msg)
            self.error = msg
            self.input = []
            return
        for hdul in result:
            self.update_output(hdul, self.prodtypes[self.step_index])
        self.input = list(result)
        self.set_display_data()
        if save:
            for hdul in self.input:
                self.write_output(hdul)

    def correct_wave_shift(self):
        """Correct wavelengths for barycentric shift."""
        from sofia_redux.instruments.fifi_ls.correct_wave_shift \
            import wrap_correct_wave_shift

        # get parameters
        param = self.get_parameter_set()
        save = param.get_value('save')
        parallel = param.get_value('parallel')
        skip_shift = param.get_value('skip_shift')
        if parallel:
            jobs = self.max_cores
        else:
            jobs = None

        if skip_shift:
            log.info('No wavelength shift correction performed.')
            return

        result = wrap_correct_wave_shift(self.input, write=False,
                                         jobs=jobs,
                                         allow_errors=True)
        if not result:
            # in this case, just pass through -- this is likely
            # because telluric correction was not done, but skip
            # was not set
            log.warning('No wavelength shift correction performed.')
            return
        for hdul in result:
            self.update_output(hdul, self.prodtypes[self.step_index])
        self.input = list(result)
        self.set_display_data()
        if save:
            for hdul in self.input:
                self.write_output(hdul)

    def resample(self):
        """Resample pixels onto a regular grid."""
        from sofia_redux.instruments.fifi_ls.resample import resample

        # get parameters
        param = self.get_parameter_set()
        save = param.get_value('save')
        parallel = param.get_value('parallel')
        skip_coadd = param.get_value('skip_coadd')
        interp = param.get_value('interpolate')
        error_weighting = param.get_value('error_weighting')
        xy_oversample = param.get_value('xy_oversample')
        w_oversample = param.get_value('w_oversample')
        xy_pixsize = param.get_value('xy_pixel_size')
        w_pixsize = param.get_value('w_pixel_size')
        xy_order = param.get_value('xy_order')
        w_order = param.get_value('w_order')
        xy_window = param.get_value('xy_window')
        adaptive_algorithm = param.get_value('adaptive_algorithm')
        append_weights = param.get_value('append_weights')
        w_window = param.get_value('w_window')
        xy_smoothing = param.get_value('xy_smoothing')
        w_smoothing = param.get_value('w_smoothing')
        fitthresh = param.get_value('fitthresh')
        posthresh = param.get_value('posthresh')
        negthresh = param.get_value('negthresh')
        xythresh = param.get_value('xy_edge_threshold')
        wthresh = param.get_value('w_edge_threshold')

        # fix thresholds to expected defaults
        if fitthresh <= 0:
            log.debug('Turning off fit rejection')
            fitthresh = None
        if negthresh <= 0:
            log.debug('Turning off negative rejection pass')
            negthresh = None
        if posthresh <= 0:
            log.debug('Turning off outlier rejection')
            posthresh = None

        # check for empty pixel size fields
        test = str(xy_pixsize).strip().lower()
        if test == 'none' or test == '':
            xy_pixsize = None
        test = str(w_pixsize).strip().lower()
        if test == 'none' or test == '':
            w_pixsize = None

        if skip_coadd:
            # iterate over all files
            files = [[f] for f in self.input]
        else:
            # reduce all files together
            files = [self.input]

        if parallel:
            jobs = self.max_cores
        else:
            jobs = None

        # set adaptive smoothing for spatial dimensions if desired
        if adaptive_algorithm in ['scaled', 'shaped']:
            # in this case, smoothing must be set to 1 FWHM
            if not np.allclose(xy_smoothing, gaussian_fwhm_to_sigma,
                               atol=.01):
                log.warning('Setting x/y smoothing radius '
                            'to Gaussian sigma for adaptive case.')
            xy_smoothing = gaussian_fwhm_to_sigma
            adaptive_threshold = (1.0, 1.0, 0.0)
        else:
            log.debug('Turning off adaptive smoothing')
            adaptive_algorithm = None
            adaptive_threshold = None

        results = []
        for inp_set in files:
            result = resample(inp_set, write=False, interp=interp,
                              oversample=(xy_oversample, w_oversample),
                              spatial_size=xy_pixsize, spectral_size=w_pixsize,
                              window=(xy_window, xy_window, w_window),
                              adaptive_algorithm=adaptive_algorithm,
                              adaptive_threshold=adaptive_threshold,
                              error_weighting=error_weighting,
                              smoothing=(xy_smoothing, xy_smoothing,
                                         w_smoothing),
                              order=(xy_order, xy_order, w_order),
                              robust=posthresh,
                              neg_threshold=negthresh,
                              fit_threshold=fitthresh,
                              edge_threshold=(xythresh, xythresh, wthresh),
                              append_weights=append_weights,
                              jobs=jobs)
            if not result:
                msg = 'Problem in fifi_ls.resample.'
                log.error(msg)
                self.error = msg
                self.input = []
                return
            results.append(result)

        for hdul in results:
            self.update_output(hdul, self.prodtypes[self.step_index])
        self.input = results
        self.set_display_data()

        if save:
            for hdul in self.input:
                self.write_output(hdul)

    def specmap(self):
        """
        Generate a quick-look image and spectral plot.

        Calls `sofia_redux.visualization.quicklook.make_image` to
        make the image.  Calls `sofia_redux.instruments.fifi_ls.get_lines`
        to get commonly-observed FIFI-LS wavelengths.

        The output from this step is identical to the input, so is
        not saved.  As a side effect, a PNG file is saved to disk to the
        same base name as the input file, with a '.png' extension.
        """
        from astropy import units as u
        from astropy.coordinates import Angle
        from astropy.wcs import WCS
        from scipy.ndimage import gaussian_filter
        from sofia_redux.visualization.quicklook import \
            make_image, make_spectral_plot

        # get parameters
        param = self.get_parameter_set()
        skip_preview = param.get_value('skip_preview')

        # check whether to continue
        if skip_preview:
            log.info('Not making preview image.')
            return

        extension = param.get_value('extension')
        slice_method = param.get_value('slice_method')
        point_method = param.get_value('point_method')
        override_slice = param.get_value('override_slice')
        override_point = param.get_value('override_point')
        colormap = param.get_value('colormap')
        scale = param.get_value('scale')
        n_contour = param.get_value('n_contour')
        contour_color = param.get_value('contour_color')
        fill_contours = param.get_value('fill_contours')
        grid = param.get_value('grid')
        beam = param.get_value('beam')
        watermark = param.get_value('watermark')
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
                slice_method = 'reference'
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
                point_method = 'reference'
        else:
            override_xy = None

        # set plot scale to default, if it is full range
        if spec_scale[0] <= 0 and spec_scale[1] >= 100:
            spec_scale = None

        for i, hdul in enumerate(self.input):
            header = hdul[0].header
            wave = hdul['WAVELENGTH'].data
            xunit = hdul['WAVELENGTH'].header.get('BUNIT', 'um')

            # set text for title and subtitle in plot
            obj = header.get('OBJECT', 'UNKNOWN')
            channel = header.get('CHANNEL', 'UNKNOWN')
            basename = os.path.basename(header.get('FILENAME', 'UNKNOWN'))

            # get reference wavelength and position
            # These keys should always be in the header - let it
            # fail if they aren't
            if channel == 'RED':
                ref_wave = header['G_WAVE_R']
            else:
                ref_wave = header['G_WAVE_B']
            ref_ra = header['OBSRA'] * 15.0
            ref_dec = header['OBSDEC']

            # check for older style beam units in header
            if 'BMAJ' in header and 'arcsec' in header.comments['BMAJ']:
                for hdu in hdul:
                    if 'BMAJ' in hdu.header:
                        hdu.header['BMAJ'] /= 3600
                        hdu.header['BMIN'] /= 3600

            # get flux
            if 'UNCOR' in str(extension).upper():
                log.info('Displaying UNCORRECTED_FLUX')
                flux = hdul['UNCORRECTED_FLUX'].data
                err = hdul['UNCORRECTED_ERROR'].data
            elif extension == 'FLUX':
                flux = hdul['FLUX'].data
                err = hdul['ERROR'].data
            else:
                raise ValueError(f'Invalid extension for plot: {extension}')

            # pull data from cube with specified method
            x, y, spec_peak, spec_wv, spec_flux, spec_err = \
                None, None, None, None, None, None

            if slice_method == 'reference':
                spec_peak, spec_wv = None, None
                if override_w is not None:
                    nearest = override_w
                else:
                    # don't allow out of range references
                    if wave[0] > ref_wave or wave[-1] < ref_wave:
                        nearest = -1
                    else:
                        # wavelength slice is nearest one to the reference
                        nearest = int(np.argmin(np.abs(wave - ref_wave)))

                if (0 <= nearest < flux.shape[0]
                        and not np.all(np.isnan(flux[nearest]))):
                    spec_peak = nearest
                    spec_wv = wave[nearest]
                if spec_peak is None:
                    log.warning(f'Reference wavelength slice at index '
                                f'{nearest} is empty; using peak method.')
                    slice_method = 'peak'
                else:
                    if override_w is not None:
                        log.info(f'Plotting at {spec_wv:.3f} {xunit}, '
                                 f'index {nearest}')
                    else:
                        log.info(f'Plotting at {spec_wv:.3f} {xunit}, '
                                 f'near reference wavelength at '
                                 f'{ref_wave} {xunit}')

                    # image pixel is highest flux in selected slice
                    try:
                        img_peak = np.unravel_index(
                            np.nanargmax(flux[spec_peak]),
                            (flux.shape[1], flux.shape[2]))
                    except ValueError:  # pragma: no cover
                        # this shouldn't be reachable under ordinary
                        # circumstances, since NaNs are accounted for in
                        # the line search
                        img_peak = flux.shape[1] / 2, flux.shape[2] / 2
                    y, x = [int(i) for i in img_peak]

            if slice_method == 'peak':
                # looking for best signal-to-noise voxel in cube,
                # smoothed by a couple pixels
                sigma = 2.0
                sflux = gaussian_filter(flux, sigma=sigma,
                                        mode='constant', cval=np.nan,
                                        truncate=2)
                serr = gaussian_filter(err, sigma=sigma,
                                       mode='constant', cval=np.nan,
                                       truncate=2)

                # set the outer N% of frames to NaN
                wstart = int(ignore_outer * len(wave))
                wend = int((1 - ignore_outer) * len(wave))
                sflux[:wstart] = np.nan
                sflux[wend:] = np.nan

                try:
                    img_peak = np.unravel_index(np.nanargmax(sflux / serr),
                                                flux.shape)[1:]
                except ValueError:
                    img_peak = flux.shape[1] / 2, flux.shape[2] / 2

                # peak in the real spectrum at that location
                y, x = [int(i) for i in img_peak]
                spec_flux = flux[:, y, x]
                try:
                    spec_peak = \
                        int(np.nanargmax(spec_flux[wstart:wend])) + wstart
                except ValueError:
                    try:
                        spec_peak = int(np.nanargmax(spec_flux))
                    except ValueError:
                        log.error('No good data; not creating map')
                        return
                spec_wv = wave[spec_peak]
                log.info(f'Plotting at S/N peak {spec_wv:.3f} {xunit}')

            # x and y selected so far are at peak; override if
            # reference is desired
            hwcs = WCS(hdul[extension].header)
            if point_method == 'reference':
                if override_xy is not None:
                    rx, ry = override_xy
                else:
                    rx, ry, rw = hwcs.wcs_world2pix(ref_ra, ref_dec,
                                                    spec_wv, 0)
                if (np.any(np.isnan([rx, ry]))
                        or rx <= 0 or rx >= flux.shape[2]
                        or ry <= 0 or ry >= flux.shape[1]):
                    if override_xy:
                        log.warning(f'Reference spatial position at index '
                                    f'{rx},{ry} out of range; using peak '
                                    f'pixel at {x},{y}.')
                    else:
                        log.warning(f'Reference spatial position at '
                                    f'RA={ref_ra}, Dec={ref_dec} out of '
                                    f'range; using peak pixel at {x},{y}.')
                else:
                    if override_xy:
                        log.info(f'Spectral plot taken at pixel '
                                 f'{rx},{ry}')
                    else:
                        log.info(f'Spectral plot taken at reference position '
                                 f'near RA={ref_ra} Dec={ref_dec}')
                    x, y = int(rx), int(ry)
            else:
                log.info(f'Spectral plot taken at peak flux, pixel {x},{y}')

            # get spectral flux at selected image point
            spec_flux = flux[:, y, x]
            spec_err = err[:, y, x]

            # get sky location of chosen point for display
            ra, dec, wv = hwcs.wcs_pix2world(x, y, spec_peak, 0)
            ra_str = Angle(ra, unit=u.deg).to_string(unit=u.hour)
            dec_str = Angle(dec, unit=u.deg).to_string(unit=u.deg)

            # get atran data for overplot if needed
            if atran_plot:
                aplot = [wave, hdul['TRANSMISSION'].data]
            else:
                aplot = None

            # titles for plot
            title = f'Object: {obj}, Channel: {channel}'
            subtitle = f'Filename: {basename}'
            subsubtitle = f'Wavelength slice at {spec_wv:.2f} {xunit}'
            spectitle = f'Spectrum at RA={ra_str}, Dec={dec_str}'
            yunit = hdul['FLUX'].header.get('BUNIT', 'UNKNOWN')

            # make the image figure
            fig = make_image(hdul, extension=extension, colormap=colormap,
                             scale=scale, n_contour=n_contour,
                             contour_color=contour_color,
                             fill_contours=fill_contours, title=title,
                             subtitle=subtitle, subsubtitle=subsubtitle,
                             grid=grid, beam=beam,
                             plot_layout=(2, 1), cube_slice=spec_peak,
                             watermark=watermark)

            # mark the spectral position
            ax1 = fig.axes[0]
            ax1.scatter(x, y, c=contour_color, marker='x', s=20,
                        alpha=0.8)

            if not error_plot:
                spec_err = None

            # add a spectral plot
            ax = fig.add_subplot(2, 1, 2)
            make_spectral_plot(ax, wave, spec_flux, spectral_error=spec_err,
                               scale=spec_scale, colormap=colormap,
                               xunit=xunit, yunit=yunit, title=spectitle,
                               marker=[spec_wv, spec_flux[spec_peak]],
                               marker_color=contour_color,
                               overplot=aplot,
                               overplot_label='Atmospheric Transmission',
                               watermark=watermark)

            # output filename for image
            fname = os.path.splitext(basename)[0] + '.png'
            outname = os.path.join(self.output_directory, fname)

            fig.savefig(outname, dpi=300)
            log.info(f'Saved image to {outname}')
