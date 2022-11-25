# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""EXES Reduction pipeline steps"""

import copy
import os
import re
import time
import warnings

from astropy import log
from astropy import constants as const
from astropy.io import fits
import numpy as np

from sofia_redux.pipeline.sofia.sofia_exception import SOFIAImportError

try:
    import sofia_redux.instruments.exes as ex
except ImportError:
    raise SOFIAImportError('EXES modules not installed')

from sofia_redux.instruments.exes.mergehdr import mergehdr
from sofia_redux.instruments.exes.utils import parse_central_wavenumber
from sofia_redux.pipeline.gui.qad_viewer import QADViewer
from sofia_redux.pipeline.gui.matplotlib_viewer import MatplotlibViewer
from sofia_redux.pipeline.reduction import Reduction
from sofia_redux.pipeline.sofia.parameters.exes_parameters \
    import EXESParameters
from sofia_redux.pipeline.sofia.sofia_utilities import \
    parse_bg, parse_apertures
from sofia_redux.toolkit.utilities.fits import \
    hdinsert, getheader, set_log_level, gethdul
from sofia_redux.toolkit.image.adjust import rotate90, unrotate90
from sofia_redux.toolkit.image.combine import combine_images
from sofia_redux.visualization.redux_viewer import EyeViewer

__all__ = ['EXESReduction']


class EXESReduction(Reduction):
    """
    EXES reduction steps.

    Primary image reduction algorithms are defined in the EXES
    package (`sofia_redux.instruments.exes`).  Spectroscopy-related
    algorithms are pulled from the `sofia_redux.spectroscopy` package,
    and some utilities and display tools come from the `sofia_redux.toolkit`,
    `sofia_redux.calibration`, and `sofia_redux.visualization` packages.

    This reduction object defines a recipe for data reduction and a
    method for each pipeline step that calls the appropriate algorithm
    from its source packages.

    Attributes
    ----------
    prodtype_map : dict
        Maps the pipeline step to a product type, to assign
        to the PRODTYPE header key. Keys are pipeline step function
        names.
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
        Processing recipe used for standard science processing.
    default_prodtype_map : list
        Product type map used for standard science processing.
    default_prodnames : list
        Product names used for standard science processing.
    default_step_map : list
        Step name map used for standard science processing.
    sky_spectrum : bool
        Flag to indicate that data should be reduced as a sky spectrum
        extraction, instead of standard science processing.
    sky_prodtype_map : list
        Alternate product types to use for sky spectrum extraction.
    sky_prodnames : list
        Alternate product names to use for sky spectrum extraction.
    sky_step_map : list
        Alternate step name map to use for sky spectrum extraction.
    wcs_keys : list
        List of header keywords used for tracking and propagating the
        spectral world coordinate system.
    spec1d_prodtype : list
        List of product types currently and historically used to designate
        1D spectral products, used for display purposes and
        backwards-compatible handling.
    """

    def __init__(self):
        """Initialize the reduction object."""
        super().__init__()

        # descriptive attributes
        self.name = 'Redux'
        self.instrument = 'EXES'
        self.mode = 'Spectroscopy'

        self.data_keys = ['File Name', 'OBJECT', 'OBSTYPE',
                          'AOR_ID', 'MISSN-ID', 'DATE-OBS',
                          'INSTCFG', 'INSTMODE',
                          'SPECTEL1', 'SPECTEL2', 'WAVENO0',
                          'NODBEAM', 'EXPTIME']

        self.pipe_name = 'EXES_REDUX'
        self.pipe_version = ex.__version__.replace('.', '_')

        # parameters will be instantiated on load
        self.parameters = None

        # product type definitions for EXES steps
        self.prodtype_map = {
            'load_data': 'raw',
            'coadd_readouts': 'readouts_coadded',
            'make_flat': 'flat_appended',
            'despike': 'despiked',
            'debounce': 'debounced',
            'subtract_nods': 'nods_subtracted',
            'flat_correct': 'flat_corrected',
            'clean_badpix': 'cleaned',
            'undistort': 'undistorted',
            'correct_calibration': 'calibration_corrected',
            'coadd_pairs': 'coadded',
            'convert_units': 'calibrated',
            'make_profiles': 'rectified_image',
            'locate_apertures': 'apertures_located',
            'set_apertures': 'apertures_set',
            'subtract_background': 'background_subtracted',
            'extract_spectra': 'spectra',
            'combine_spectra': 'coadded_spectrum',
            'refine_wavecal': 'wavecal_refined',
            'merge_orders': 'orders_merged',
            'specmap': 'specmap'}
        self.prodnames = {
            'raw': 'RAW',
            'readouts_coadded': 'RDC',
            'flat_appended': 'FTA',
            'despiked': 'DSP',
            'debounced': 'DBC',
            'nods_subtracted': 'NSB',
            'flat_corrected': 'FTD',
            'cleaned': 'CLN',
            'undistorted': 'UND',
            'calibration_corrected': 'CCR',
            'coadded': 'COA',
            'calibrated': 'CAL',
            'rectified_image': 'RIM',
            'apertures_located': 'LOC',
            'apertures_set': 'APS',
            'background_subtracted': 'BGS',
            'spectra': 'SPM',
            'coadded_spectrum': 'COM',
            'wavecal_refined': 'WRF',
            'orders_merged': 'MRM',
            'specmap': 'SMP'}

        # invert the map for quick lookup of step from type
        self.step_map = {v: k for k, v in self.prodtype_map.items()}

        # This will be populated when the recipe is set
        self.prodtypes = list()

        # Default recipe and step names
        self.recipe = [
            'load_data', 'coadd_readouts', 'make_flat', 'despike',
            'debounce', 'subtract_nods', 'flat_correct',
            'clean_badpix', 'undistort', 'correct_calibration',
            'coadd_pairs', 'convert_units',
            'make_profiles', 'locate_apertures',
            'set_apertures', 'subtract_background',
            'extract_spectra', 'combine_spectra',
            'refine_wavecal', 'merge_orders', 'specmap']
        self.processing_steps = {
            'load_data': 'Load Data',
            'coadd_readouts': 'Coadd Readouts',
            'make_flat': 'Make Flat',
            'despike': 'Despike',
            'debounce': 'Debounce',
            'subtract_nods': 'Subtract Nods',
            'flat_correct': 'Flat Correct',
            'clean_badpix': 'Clean Bad Pixels',
            'undistort': 'Undistort',
            'correct_calibration': 'Correct Calibration',
            'coadd_pairs': 'Coadd Pairs',
            'convert_units': 'Convert Units',
            'make_profiles': 'Make Profiles',
            'locate_apertures': 'Locate Apertures',
            'set_apertures': 'Set Apertures',
            'subtract_background': 'Subtract Background',
            'extract_spectra': 'Extract Spectra',
            'combine_spectra': 'Combine Spectra',
            'refine_wavecal': 'Refine Wavecal',
            'merge_orders': 'Merge Orders',
            'specmap': 'Make Spectral Map'}

        # reduction information
        self.output_directory = os.getcwd()
        self.filenum = list()

        # keep a copy of the default recipe and products, for later restoration
        self.default_recipe = self.recipe.copy()
        self.default_prodtype_map = self.prodtype_map.copy()
        self.default_prodnames = self.prodnames.copy()
        self.default_step_map = self.step_map.copy()

        # store some alternate product types and codes, for sky spectra
        self.sky_spectrum = False
        self.sky_prodtype_map = {
            'subtract_nods': 'sky_nods_subtracted',
            'flat_correct': 'sky_flat_corrected',
            'clean_badpix': 'sky_cleaned',
            'undistort': 'sky_undistorted',
            'correct_calibration': 'sky_calibration_corrected',
            'coadd_pairs': 'sky_coadded',
            'convert_units': 'sky_calibrated',
            'make_profiles': 'sky_rectified_image',
            'locate_apertures': 'sky_apertures_located',
            'set_apertures': 'sky_apertures_set',
            'subtract_background': 'sky_background_subtracted',
            'extract_spectra': 'sky_spectra',
            'combine_spectra': 'sky_coadded_spectrum',
            'refine_wavecal': 'sky_wavecal_refined',
            'merge_orders': 'sky_orders_merged',
            'specmap': 'sky_specmap'}
        self.sky_prodnames = {
            'sky_nods_subtracted': 'SNS',
            'sky_flat_corrected': 'SFT',
            'sky_cleaned': 'SCN',
            'sky_undistorted': 'SUN',
            'sky_calibration_corrected': 'SCR',
            'sky_coadded': 'SCO',
            'sky_calibrated': 'SCL',
            'sky_rectified_image': 'SRM',
            'sky_apertures_located': 'SLC',
            'sky_apertures_set': 'SAP',
            'sky_background_subtracted': 'SBG',
            'sky_spectra': 'SSM',
            'sky_coadded_spectrum': 'SCM',
            'sky_wavecal_refined': 'SWR',
            'sky_orders_merged': 'SMM',
            'sky_specmap': 'SSS'}
        self.sky_step_map = {v: k for k, v in self.sky_prodtype_map.items()}

        # store some WCS keys used for tracking and propagating
        # the 2D spectral WCS
        self.wcs_keys = [
            'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2',
            'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2',
            'CDELT1', 'CDELT2', 'CROTA2', 'SPECSYS']

        # store some 1D spectrum types, for display and
        # historical accommodation
        self.spec1d_prodtype = [
            'spectra_1d', 'merged_spectrum_1d', 'calibrated_spectrum_1d',
            'combined_spectrum', 'combined_spectrum_1d',
            'wavecal_refined_1d', 'orders_merged_1d',
            'sky_spectra_1d', 'sky_merged_spectrum_1d',
            'sky_calibrated_spectrum_1d', 'sky_combined_spectrum',
            'sky_combined_spectrum_1d', 'sky_wavecal_refined_1d',
            'sky_orders_merged_1d',
            'spec', 'combspec', 'calspec', 'mrgspec', 'mrgordspec',
            'sky_spec', 'sky_combspec', 'sky_mrgordspec']

    def __setattr__(self, name, value):
        """Check if recipe is being set. If so, also set prodtypes."""
        if name == 'recipe':
            # set the prodtype list
            try:
                self.prodtypes = [self.prodtype_map[step] for step in value]
            except AttributeError:
                self.prodtypes = list()
        super().__setattr__(name, value)

    # input/output methods

    @staticmethod
    def get_filenum(filename):
        """
        Get a file number from an EXES file name.

        Formats expected are for raw EXES files (e.g. sirius.sci.10001.fits)
        or for intermediate processed files (e.g.
        F0001_EX_SPE_0101123_EXEELONEXEECHL_COA_10001-10002.fits.
        In either case, the field just before the .fits is expected to
        contain the file number.

        If the input file references a range of files, e.g. '10001-10002',
        the first and last file numbers are returned in a list. Otherwise,
        a single integer is returned.  If neither pattern is found,
        'UNKNOWN' is returned.

        Parameters
        ----------
        filename : str
            FITS file name, raw or intermediate.

        Returns
        -------
        filenum : str or list of str
            Integer file numbers referenced in the input file name.
        """
        split_filename = re.split('[_.]', os.path.basename(filename))

        # check for single file with trailing serial number
        # (for split coadds)
        if len(split_filename) > 3:
            if (re.match(r'^\d+$', split_filename[-2])
                    and re.match(r'^\d+$', split_filename[-3])):
                filenum = '_'.join(split_filename[-3:-1])
                return filenum

        # otherwise check for ranges of raw file numbers
        try:
            num = split_filename[-2]
            try:
                filenum = f'{int(num):04d}'
            except ValueError:
                filenum = [f'{int(n):04d}' for n in num.split('-')]
        except (ValueError, IndexError):
            filenum = 'UNKNOWN'
        return filenum

    @staticmethod
    def concatenate_filenum(filenum):
        """
        Concatenate file numbers, first-last.

        File numbers are sorted alphanumerically.

        Parameters
        ----------
        filenum : str or list
            File numbers to concatenate.

        Returns
        -------
        str
            Concatenated file number.
        """
        if not isinstance(filenum, list):
            filenum = [filenum]

        filenums = []
        for fn in filenum:
            if isinstance(fn, list):
                # one more layer of possible lists
                for f in fn:
                    if isinstance(f, list):
                        filenums.extend(f)
                    else:
                        filenums.append(f)
            else:
                filenums.append(fn)

        filenums = [str(fn) for fn in filenums]
        if len(filenums) > 1:
            # strip any trailing serial numbers
            filenums = [f.split('_')[0] for f in filenums]

            # keep the first and last, if different
            filenums.sort()
            if filenums[0] == filenums[-1]:
                filenum_str = filenums[0]
            else:
                filenum_str = '-'.join([filenums[0], filenums[-1]])
        else:
            filenum_str = filenums[0]

        return filenum_str

    def get_filename(self, header, filenum=None, prodtype=None, update=True):
        """
        Create an output filename from an input header.

        Parameters
        ----------
        header : astropy.io.fits.Header
            Header to create filename from.
        filenum : str or list, optional
            List of file numbers to concatenate for filename.
        prodtype : str, optional
            Three letter product type designator.
        update : bool, optional
            If set, the FILENAME key will be added or updated
            in the header.

        Returns
        -------
        filename : str
           The output name.
        """
        # Get flight number
        missn = header.get('MISSN-ID', 'UNKNOWN')
        flight = missn.split('_')[-1].lstrip('F')
        try:
            flight = 'F{0:04d}'.format(int(flight))
        except ValueError:
            # check for line ops mission
            flight = missn.split('_')[-1].lstrip('L')
            try:
                flight = 'L{0:04d}'.format(int(flight))
            except ValueError:
                flight = 'UNKNOWN'

        # Get instrument
        try:
            data_type = header['DATATYPE']
        except KeyError:
            inst = 'EX_SPE'
        else:
            if data_type == 'IMAGE':
                inst = 'EX_IMA'
            else:
                inst = 'EX_SPE'

        # Get AOR-ID
        try:
            aorid = header['AOR_ID']
        except KeyError:
            aorid = 'UNKNOWN'
        aorid = aorid.replace('_', '')

        # Get SPECTEL
        try:
            spectel1 = header['SPECTEL1']
        except KeyError:
            spectel1 = 'UNKNOWN'
        try:
            spectel2 = header['SPECTEL2']
        except KeyError:
            spectel2 = 'UNKNOWN'
        spectel = (spectel1.replace('_', '').strip()
                   + spectel2.replace('_', '').strip())

        # get file number string
        if filenum is None:
            fn = 'UNKNOWN'
        else:
            fn = self.concatenate_filenum(filenum)

        # Get product type
        if prodtype is None:
            prodtype = 'UNKNOWN'

        filename = f'{flight}_{inst}_{aorid}_{spectel}_{prodtype}_{fn}.fits'

        if update:
            hdinsert(header, 'FILENAME', filename, comment='File name')

        return filename

    def update_sofia_keys(self, header):
        """
        Update required SOFIA header keywords.

        Keywords added or updated are:

            - PROCSTAT: set to LEVEL_2
            - ASSC_AOR: copied from AOR_ID
            - ASSC_OBS: copied from OBS_ID
            - ASSC_MSN: copied from MISSN-ID
            - OBS_ID: prepended with 'P\\_'
            - PIPELINE: set to pipe_name
            - PIPEVERS: set to pipe_version

        Parameters
        ----------
        header : `astropy.io.fits.Header`
            Header to update.
        """
        # update procstat if necessary
        procstat = header.get('PROCSTAT', 'UNKNOWN').upper()
        if procstat in ['UNKNOWN', 'LEVEL_0', 'LEVEL_1']:
            hdinsert(header, 'PROCSTAT', 'LEVEL_2',
                     comment='Processing status')

        # get AOR ID from header if science, base otherwise
        aorid = header.get('AOR_ID', 'UNKNOWN').strip().upper()
        obstype = header.get('OBSTYPE', 'UNKNOWN').strip().upper()
        if (obstype in ['FLAT', 'DARK', 'SKY']
                and self.parameters is not None
                and self.parameters.base_header is not None):
            base_aorid = self.parameters.base_header.get(
                'AOR_ID', 'UNKNOWN').strip().upper()
            if base_aorid != 'UNKNOWN' and base_aorid != aorid:
                aorid = base_aorid
                header['AOR_ID'] = aorid

        # copy AOR_ID, OBS_ID, and MISSN-ID to ASSC* keys
        # if not yet present
        assc_aor = header.get('ASSC_AOR', 'UNKNOWN').upper()
        if assc_aor == 'UNKNOWN':
            hdinsert(header, 'ASSC_AOR', aorid,
                     comment='All input AOR-IDs')

        obsid = header.get('OBS_ID', 'UNKNOWN').upper()
        assc_obs = header.get('ASSC_OBS', 'UNKNOWN').upper()
        if assc_obs == 'UNKNOWN':
            hdinsert(header, 'ASSC_OBS', obsid,
                     comment='All input OBS-IDs')

        msnid = header.get('MISSN-ID', 'UNKNOWN').upper()
        assc_msn = header.get('ASSC_MSN', 'UNKNOWN').upper()
        if assc_msn == 'UNKNOWN':
            hdinsert(header, 'ASSC_MSN', msnid,
                     comment='All input MISSN-IDs')

        # update OBS_ID if necessary
        if obsid != 'UNKNOWN' and not obsid.startswith('P_'):
            hdinsert(header, 'OBS_ID', 'P_' + obsid)

        # always update pipeline and pipevers info
        hdinsert(header, 'PIPELINE', self.pipe_name,
                 comment='Pipeline creating this file')
        hdinsert(header, 'PIPEVERS', self.pipe_version,
                 comment='Pipeline version')

        # and the current date/time
        hdinsert(header, 'DATE', time.strftime('%Y-%m-%dT%H:%M:%S'),
                 comment='Date of file creation')

    def update_output(self, hdul, filenum, prodtype):
        """
        Update output FITS file after a pipeline step.

        Sets the PRODTYPE key, adds a HISTORY message with
        the pipeline step name, and updates the FILENAME
        key with a new filename, appropriate to the prodtype.
        The new filename is returned from the function.

        Parameters
        ----------
        hdul : astropy.io.fits.HDUList
           Output FITS HDUList.
        filenum : str or list
           File number(s) to use in the filename.
        prodtype : str
           Product type for the completed step.

        Returns
        -------
        str
            File base name for the output product.
        """
        hdinsert(hdul[0].header, 'PRODTYPE', prodtype,
                 comment='Product type')

        # Add a history message with the step name
        step_name = self.step_map[prodtype]
        step_display_name = self.processing_steps[step_name]
        message = f'-- Pipeline step: {step_display_name}'
        hdinsert(hdul[0].header, 'HISTORY', message)

        params = self.get_parameter_set()
        for param in params:
            if params[param]['hidden']:
                continue
            message = f'  {param} = {params.get_value(param)}'
            hdinsert(hdul[0].header, 'HISTORY', message)

        hdinsert(hdul[0].header, 'HISTORY', '--')
        hdinsert(hdul[0].header, 'HISTORY', '')

        outname = self.get_filename(hdul[0].header, filenum,
                                    self.prodnames[prodtype])
        return outname

    def write_output(self, hdul, outname):
        """
        Write an output FITS file to disk.

        Outname is joined to self.output_directory, before writing.

        Parameters
        ----------
        hdul : astropy.io.fits.HDUList
           FITS HDUList to write.
        outname : str
           File basename to write.

        Returns
        -------
        str
            Full path to the output file.
        """
        outname = os.path.join(self.output_directory, outname)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            hdul.writeto(outname, overwrite=True)
        self.record_outfile(outname)
        return outname

    def _set_sky_products(self, sky=True):
        """
        Set alternate products for sky spectrum extraction.

        If `sky` is set, product types and names are updated for
        sky products. Parameters are also updated to set defaults
        appropriate for sky reductions, if they have not already
        been set for sky spectra.

        If `sky` is not set, default product types and names are
        restored.  Parameters are also updated to restore defaults,
        if they were previously set for sky reductions.

        Parameters
        ----------
        sky : bool, optional
            If True, sky products are set. If False, products are reset
            to default types.
        """
        self.sky_spectrum = sky
        self.prodtype_map = self.default_prodtype_map.copy()
        self.prodnames = self.default_prodnames.copy()
        self.step_map = self.default_step_map.copy()

        if sky:
            log.info('Setting sky product names.')
            self.prodtype_map.update(self.sky_prodtype_map)
            self.prodnames.update(self.sky_prodnames)
            self.step_map.update(self.sky_step_map)
        else:
            log.debug('Setting default product names.')

        # update product types if possible
        try:
            self.prodtypes = [self.prodtype_map[step] for step in self.recipe]
        except AttributeError:  # pragma: no cover
            self.prodtypes = list()

        # update parameters if available and if the setting has
        # changed from initial load
        if (self.parameters is not None
                and self.parameters.base_header is not None):
            hdinsert(self.parameters.base_header, 'SKYSPEC', sky,
                     'Sky spectrum product')
            if self.parameters.sky_spec_set != sky:
                log.info('Updating parameters for new sky setting.')
                self.update_parameters()
                self.parameters.sky_spec_set = sky

    def load(self, data):
        """
        Load input data to make it available to reduction steps.

        The process is:

        - Call the parent load method to initialize data
          reduction variables.
        - Use the first loaded FITS header that is not a flat or dark file
          to set the observation configuration for the reduction set.
        - Use the product type in the base header to determine the
          data processing recipe.
        - Load parameters for all steps.
        - Load the data immediately if it is a 1D spectrum that needs
          visualization only; otherwise, just load the raw headers and
          defer loading the data from the FITS files.

        After this step, the input attribute is populated as required
        for the first pipeline step in the recipe.

        Parameters
        ----------
        data : list of str or str
            Input file paths to load.
        """
        Reduction.load(self, data)

        # sort raw files by filename, for reduction consistency
        self.raw_files.sort(key=os.path.basename)

        # read the first non-flat FITS header
        # (or the last flat/dark if no science found)
        basehead = fits.Header()
        for infile in self.raw_files:
            basehead = getheader(infile)
            if (str(basehead.get('OBSTYPE', 'UNKNOWN')).strip().upper()
                    not in ['FLAT', 'DARK', 'SKY']):
                break

        # get product type to determine recipe
        prodtype = str(basehead.get('PRODTYPE', 'UNKNOWN')).lower().strip()

        # override for special cases (final png image only)
        one_d = False
        if prodtype in self.spec1d_prodtype:
            one_d = True
            self.recipe = ['specmap']
        else:
            if prodtype in self.sky_prodnames.keys():
                # sky spectrum intermediate products
                sky_spec = True
            else:
                sky_spec = False

            # make sure parameters and products are set accordingly
            self._set_sky_products(sky=sky_spec)
            hdinsert(basehead, 'SKYSPEC', sky_spec, 'Sky spectrum product')

            # self.prodtypes is set from the prodtype map
            # when self.recipe is set
            self.recipe = self.default_recipe

            if prodtype in self.prodtypes:
                pidx = self.prodtypes.index(prodtype)
                recipe = self.recipe[pidx + 1:]
                if len(recipe) == 0:
                    msg = f"No steps to run for prodtype '{prodtype}'."
                    log.error(msg)
                    raise ValueError(msg)

                # if the first step isn't load_data, add it in so that
                # header overrides are always allowed
                if recipe[0] != 'load_data':
                    recipe = ['load_data'] + recipe
                self.recipe = recipe

            elif prodtype.upper() != 'UNKNOWN':
                msg = f"Unrecognized prodtype '{prodtype}'."
                log.error(msg)
                raise ValueError(msg)

        self.parameters = EXESParameters(base_header=basehead)

        if not one_d:
            # just load headers, since load data is always called first
            self.input = list()
            for datafile in self.raw_files:
                self.input.append(fits.getheader(datafile))
        else:
            # load fits directly, no header updates
            self.load_fits()

        # log all planned processing steps
        log.info('Processing steps: {}'.format(self.recipe))

        # pass data to QAD for viewing
        self.set_display_data(raw=True)

    def load_fits(self):
        """Load FITS data into the input attribute."""
        self.input = list()
        self.filenum = list()

        for i, datafile in enumerate(self.raw_files):
            input_hdul = fits.open(datafile, lazy_load_hdus=False,
                                   memmap=False)

            # touch the data to make sure it is in memory
            # and store copies in a new HDUList
            # (this avoids storing a file handle in the HDUList)
            hdul = fits.HDUList()
            for hdu in input_hdul:
                assert hdu.data[0] is not None
                hdul.append(hdu.copy())
            input_hdul.close()

            # keep file numbers
            self.filenum.append(self.get_filenum(datafile))
            self.input.append(hdul)

    # display methods

    def register_viewers(self):
        """Return a new QADViewer, ProfileViewer, and SpectralViewer."""
        prof = MatplotlibViewer()
        prof.name = 'ProfileViewer'
        prof.title = 'Spatial Profiles'
        prof.layout = 'rows'
        # todo: check if this value is needed/correct
        prof.max_plot = 20

        spec = EyeViewer()
        spec.name = 'SpectralViewer'
        spec.title = 'Spectra'
        spec.layout = 'rows'

        viewers = [QADViewer(), prof, spec]

        return viewers

    def _get_profiles(self, hdul):
        """
        Assemble profile data for display in a plot viewer.

        Parameters
        ----------
        hdul : astropy.io.fits.HDUList
            The data containing profile information.

        Returns
        -------
        display_data : list of dict
            Display data formatted for a `MatplotlibViewer`.
        """
        disp_plot = []
        header = hdul[0].header
        title = header.get('FILENAME', '')
        orders = [int(x) for x in header.get('ORDERS', '1').split(',')]
        for order in orders:
            ordnum = f'{order:02d}'
            suffix = f'_ORDER_{ordnum}'
            if f'SPATIAL_PROFILE{suffix}' in hdul:
                slitpos = hdul[f'SLITPOS{suffix}'].data
                profile = hdul[f'SPATIAL_PROFILE{suffix}'].data
                yunit = hdul[f'SLITPOS{suffix}'].header.get('BUNIT', 'arcsec')
                disp = {'args': [slitpos, profile],
                        'kwargs': {'title': f'{title} \nOrder {ordnum}',
                                   'titlesize': 'medium',
                                   'xlabel': f'Slit position ({yunit})',
                                   'ylabel': 'Normalized median flux'},
                        'plot_kwargs': {'color': 'gray',
                                        'drawstyle': 'steps-mid'}}
                overplots = []

                if f'APPOSO{ordnum}' in header:
                    aps = parse_apertures(
                        header[f'APPOSO{ordnum}'], 1)[0]
                    if f'APRADO{ordnum}' in header:
                        rads = parse_apertures(
                            header[f'APRADO{ordnum}'], 1)[0]
                    else:
                        rads = [None] * len(aps)
                    if f'PSFRAD{ordnum}' in header:
                        psfs = parse_apertures(
                            header[f'PSFRAD{ordnum}'], 1)[0]
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
                    if f'BGR_O{ordnum}' in header:
                        bgrs = parse_bg(header[f'BGR_O{ordnum}'], 1)[0]
                        for reg in bgrs:
                            if len(reg) == 2:
                                idx = (slitpos >= reg[0]) & (slitpos <= reg[1])
                                overplots.append(
                                    {'args': [slitpos[idx],
                                              profile[idx]],
                                     'kwargs': {'color': '#d62728'}})
                if overplots:
                    disp['overplot'] = overplots
                disp_plot.append(disp)
        return disp_plot

    @staticmethod
    def _is_spectral(hdul):
        """
        Determine if data is displayable by the spectral viewer.

        Currently only subsets of products are displayable: 1D
        spectra and the orders_merged product.

        Parameters
        ----------
        hdul : astropy.io.fits.HDUList
            The FITS structure containing spectral data.

        Returns
        -------
        bool
            True if input is displayable spectral data; False otherwise.
        """
        # todo: update when Eye handles EXES intermediate data
        header = hdul[0].header
        ptype = str(header.get('PRODTYPE', 'UNKNOWN')).strip().lower()
        allowed = ['orders_merged', 'sky_orders_merged',
                   'spectra_1d', 'combined_spectrum_1d',
                   'wavecal_refined_1d', 'orders_merged_1d',
                   'sky_spectra_1d', 'sky_combined_spectrum_1d',
                   'sky_wavecal_refined_1d', 'sky_orders_merged_1d']
        return ptype in allowed

    @staticmethod
    def _is_1d(hdul):
        """
        Determine if data contains a 1D spectrum.

        The test is for a specific set of product types, produced
        by this pipeline.

        Parameters
        ----------
        hdul : astropy.io.fits.HDUList
            The FITS structure containing spectral data.

        Returns
        -------
        bool
            True if input is 1D spectral data; False otherwise.
        """
        # todo: update when Eye handles EXES intermediate data
        header = hdul[0].header
        ptype = str(header.get('PRODTYPE', 'UNKNOWN')).strip().lower()
        one_d = ['spectra_1d', 'combined_spectrum_1d', 'wavecal_refined_1d',
                 'orders_merged_1d',
                 'sky_spectra_1d', 'sky_combined_spectrum_1d',
                 'sky_wavecal_refined_1d', 'sky_orders_merged_1d']
        return ptype in one_d

    def set_display_data(self, raw=False, filenames=None):
        """
        Store display data for viewer.

        Parameters
        ----------
        raw : bool
            If True, display data is taken from self.rawfiles.
            If False, display data is taken from self.input.
        filenames : list of str, optional
            If provided and `raw` is False, file names will be
            passed to the viewer instead of self.input.
        """
        self.display_data = dict()
        if raw:
            data_list = self.raw_files
        elif filenames is not None:
            data_list = filenames
        else:
            data_list = self.input

        # set profile plot and spectral data if necessary
        if not raw:
            disp_qad = []
            disp_plot = []
            disp_spec = []
            for hdul in data_list:
                # get data from disk if necessary
                with set_log_level('CRITICAL'):
                    test_hdul = gethdul(hdul)
                if test_hdul is None:
                    continue
                disp_plot.extend(self._get_profiles(test_hdul))
                if self._is_spectral(test_hdul):
                    disp_spec.append(hdul)
                if not self._is_1d(test_hdul):
                    disp_qad.append(hdul)

            self.display_data['ProfileViewer'] = disp_plot
            if len(disp_spec) > 0:
                self.display_data['SpectralViewer'] = disp_spec
            if len(disp_qad) > 0:
                self.display_data['QADViewer'] = disp_qad
        else:
            self.display_data['QADViewer'] = data_list

    # exes module pipeline steps and helpers

    @staticmethod
    def _override_header_values(header, params):
        """
        Override header values for distortion parameters with user input.

        Input parameters may contain values for the following keys:

           - cent_wave
           - hrfl
           - xdfl
           - slit_rot
           - det_rot
           - hrr
           - flattamb
           - flatemis

        Parameters
        ----------
        header : astropy.io.fits.Header
            FITS header to update.
        params : ParameterSet
            Parameter set containing override values.
        """
        try:
            wavenum = float(params.get_value('cent_wave'))
        except (TypeError, ValueError):
            pass
        else:
            if wavenum > 0:
                log.info(f'Overriding central wavenumber with {wavenum}')
                header['WNO0'] = wavenum

        try:
            hrfl = float(params.get_value('hrfl'))
        except (TypeError, ValueError):
            pass
        else:
            if hrfl != -9999:
                log.info(f'Overriding HR focal length with {hrfl}')
                header['HRFL0'] = hrfl

        try:
            xdfl = float(params.get_value('xdfl'))
        except (TypeError, ValueError):
            pass
        else:
            if xdfl != -9999:
                log.info(f'Overriding XD focal length with {xdfl}')
                header['XDFL0'] = xdfl

        try:
            slit_rot = float(params.get_value('slit_rot'))
        except (TypeError, ValueError):
            pass
        else:
            if slit_rot != -9999:
                log.info(f'Overriding slit rotation angle with {slit_rot}')
                header['SLITROT'] = slit_rot

        try:
            det_rot = float(params.get_value('det_rot'))
        except (TypeError, ValueError):
            pass
        else:
            if det_rot != -9999:
                log.info(f'Overriding detector rotation angle with {det_rot}')
                header['DETROT'] = det_rot

        try:
            hrr = float(params.get_value('hrr'))
        except (TypeError, ValueError):
            pass
        else:
            if hrr != -9999:
                log.info(f'Overriding echelon R number with {hrr}')
                header['HRR'] = hrr

        try:
            flattamb = float(params.get_value('flattamb'))
        except (TypeError, ValueError):
            pass
        else:
            if flattamb != -9999:
                log.info(f'Overriding flat mirror ambient temperature '
                         f'with {flattamb}')
                header['FLATTAMB'] = flattamb

        try:
            flatemis = float(params.get_value('flatemis'))
        except (TypeError, ValueError):
            pass
        else:
            if flatemis != -9999:
                log.info(f'Overriding flat mirror emissivity with {flatemis}')
                header['FLATEMIS'] = flatemis

    def load_data(self):
        """
        Load FITS data into memory and standardize headers.

        Calls `sofia_redux.instruments.exes.readhdr` to standardize
        headers and load expected default values.

        If the sky_spec parameter is set, product types and names
        are updated for sky spectrum extraction.
        """
        from sofia_redux.instruments.exes.readhdr import readhdr

        param = self.get_parameter_set()
        abort = param.get_value('abort')
        sky_spec = param.get_value('sky_spec')

        # validate headers and update with defaults
        all_valid = True
        updated = list()
        for header in self.input:
            log.info(f"Input: {header['FILENAME']}")
            self._override_header_values(header, param)
            new_header, valid = readhdr(header, check_header=True)
            all_valid &= valid
            updated.append(new_header)
            log.info('')
        if not all_valid and abort:
            msg = 'Invalid headers.'
            log.error(msg)
            self.error = msg
            self.input = list()
            return

        # if all passed validation, read fits files
        # into self.input for processing
        self.load_fits()

        # replace header values with updated values
        for i, hdul in enumerate(self.input):
            hdul[0].header = updated[i]

            # always update sofia keys last
            self.update_sofia_keys(hdul[0].header)

        # update product types and parameters
        # for sky spectrum defaults
        self._set_sky_products(sky=sky_spec)

        # show loaded data with updated headers
        self.set_display_data()

    def _check_raw_rasterized(self):
        """
        Check raw data to determine if it is a rasterized flat.

        The output dictionary has items:

            - 'flag' : array-like of int
                 One entry per input file, where 1 indicates a
                 raster flat, -1 indicates a raster dark, and 0
                 indicates neither.
            - 'dark' : array-like of float
                 Raw dark data cube, read from the first raster dark
                 encountered. None if not found.
            - 'dark_header' : astropy.io.fits.Header
                 FITS header for the first raster dark encountered.
                 None if not found.

        Returns
        -------
        raster : dict
            Raster flat settings for input data.
        """
        raster_flat = np.zeros(len(self.raw_files), dtype=int)
        raster_dark = None
        raster_header = None
        for i, hdul in enumerate(self.input):
            header = hdul[0].header
            calsrc = str(header.get('CALSRC', 'UNKNOWN')).lower().strip()
            instmode = str(header['INSTMODE']).lower().strip()
            obstype = str(header['OBSTYPE']).lower().strip()
            naxis = int(header['NAXIS2'])

            if (calsrc == 'blackbody' and instmode == 'stare'
                    and obstype == 'sky' and naxis < 1024):
                # flag flat to derasterize
                raster_flat[i] = 1
            elif obstype == 'dark' and naxis < 1024:
                # keep the first raster dark
                if raster_dark is None:
                    raster_dark = hdul[0].data
                    raster_header = header
                raster_flat[i] = -1

        raster = {'flag': raster_flat, 'dark': raster_dark,
                  'dark_header': raster_header}
        return raster

    def coadd_readouts(self):
        """
        Coadd raw readouts.

        Calls `sofia_redux.instruments.exes.derasterize` for raster
        flats and darks and `sofia_redux.instruments.exes.readraw`
        for all other data.

        Optionally, if the 'fix_row_gains' parameter is set,
        `exes.correct_row_gains` is called after readouts are coadded.
        """
        from sofia_redux.instruments.exes.readraw import readraw
        from sofia_redux.instruments.exes.derasterize import derasterize
        from sofia_redux.instruments.exes.correct_row_gains \
            import correct_row_gains

        param = self.get_parameter_set()

        save = param.get_value('save')
        lin_corr = param.get_value('lin_corr')
        fix_row_gains = param.get_value('fix_row_gains')
        toss_int_sci = param.get_value('toss_int_sci')
        toss_int_flat = param.get_value('toss_int_flat')
        toss_int_dark = param.get_value('toss_int_dark')
        copy_int = param.get_value('copy_integrations')
        algorithm = str(param.get_value('algorithm')).lower().strip()

        # parse algorithm from:
        #  options = ['Default for read mode',
        #             'Last destructive only',
        #             'First / last frame only'
        #             'Second / penultimate frame only']
        if algorithm.startswith('last'):
            algorithm = 0
        elif algorithm.startswith('first'):
            algorithm = 2
        elif algorithm.startswith('second'):
            # beware: this option will not work with all
            # readout patterns
            algorithm = 3
        else:
            algorithm = None

        raster = self._check_raw_rasterized()
        results = list()
        filenum = list()
        for i, hdul in enumerate(self.input):

            data = hdul[0].data
            header = hdul[0].header
            log.info(f"Input: {header['FILENAME']}")

            # drop any raster darks --
            # they will be handled with raster flats
            if raster['flag'][i] == -1:
                log.warning('Skipping direct handling for raster dark.')
                log.info('')
                continue
            filenum.append(self.filenum[i])

            # pass through any processed flats
            if str(header.get('PRODTYPE',
                              'UNKNOWN')).upper().strip() == 'FLAT':
                log.info('Passing processed flat, unmodified.')
                log.info('')
                results.append(hdul)
                continue

            # otherwise, assemble raw data
            if raster['flag'][i] == 1:
                # derasterize flat
                log.info('Processing raster flat.')
                data, variance, lin_mask = derasterize(
                    data, header, dark_data=raster['dark'],
                    dark_header=raster['dark_header'])
                log.info('')
            else:
                # get toss parameter by obstype
                obstype = str(header['OBSTYPE']).upper()
                if obstype == 'FLAT':
                    toss_nint = toss_int_flat
                elif obstype == 'DARK':
                    toss_nint = toss_int_dark
                else:
                    toss_nint = toss_int_sci

                data, variance, lin_mask = readraw(
                    data, header, do_lincor=lin_corr, algorithm=algorithm,
                    toss_nint=toss_nint, copy_int=copy_int)

            # if desired, fix odd/even row gains
            if fix_row_gains:
                data = correct_row_gains(data)

            new_hdul = fits.HDUList()
            new_hdul.append(fits.ImageHDU(data=data, header=header,
                                          name='FLUX'))
            # add BUNIT to extension headers
            exthead = fits.Header()
            hdinsert(exthead, 'BUNIT', 'ct', 'Data units')

            new_hdul.append(fits.ImageHDU(data=np.sqrt(variance),
                                          header=exthead,
                                          name='ERROR'))
            exthead['BUNIT'] = ''
            new_hdul.append(fits.ImageHDU(data=(~lin_mask).astype(int),
                                          header=exthead,
                                          name='MASK'))

            outname = self.update_output(new_hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])
            if save:
                self.write_output(new_hdul, outname)

            results.append(new_hdul)
            log.info('')

        self.input = results
        self.filenum = filenum
        self.set_display_data()

    @staticmethod
    def _combine_flats(flats):
        """
        Combine separate input flats.

        Performs a mean on all input flat data, frame by frame,
        weighted by the associated errors. Input flat data shapes
        should match.

        Parameters
        ----------
        flats : list of astropy.io.fits.HDUList
            List of input flat files. Extensions FLUX, ERROR, and MASK
            are expected.

        Returns
        -------
        combined_flat : astropy.io.fits.HDUList
            Output file containing combined flat data.
        """
        # start with the first input file
        header = flats[0][0].header
        prodtype = str(header.get('PRODTYPE', 'UNKNOWN')).upper().strip()
        if prodtype == 'FLAT':
            log.warning(f"Previously processed flats: using the first "
                        f"one only ({header.get('FILENAME', 'UNKNOWN')}).")
            return flats[0]

        # more than one raw flat: average each card
        log.info('Mean-combining input flat frames')
        new_hdul = copy.deepcopy(flats[0])

        all_cards = []
        all_var = []
        all_mask = []
        all_hdr = []
        test_shape = None
        for hdul in flats:
            if test_shape is None:
                test_shape = hdul['FLUX'].data.shape
            elif hdul['FLUX'].data.shape != test_shape:
                msg = 'Flat files do not match and cannot be combined.'
                raise ValueError(msg)
            all_cards.append(hdul['FLUX'].data)
            all_var.append(hdul['ERROR'].data ** 2)
            all_mask.append(hdul['MASK'].data)
            all_hdr.append(hdul[0].header)

        # mean combine each card separately
        # (not robust because likely to be 2 or 3 input files at most)
        all_cards = np.array(all_cards)
        all_var = np.array(all_var)

        for n in range(test_shape[0]):
            mean, var = combine_images(all_cards[:, n, :, :],
                                       all_var[:, n, :, :], method='mean',
                                       weighted=True, robust=False,
                                       returned=True)
            new_hdul['FLUX'].data[n] = mean
            new_hdul['ERROR'].data[n] = np.sqrt(var)

        # directly 'or' the 2D mask
        new_hdul['MASK'].data = np.any(all_mask, axis=0).astype(int)

        # merge all headers
        new_hdul[0].header = mergehdr(all_hdr)

        return new_hdul

    def _write_flat(self, flat, error, illum, filenum,
                    dark=None, save=True):
        """
        Write a processed flat file.

        The output file has PRODTYPE = flat, file code FLT, and extensions
        FLAT, FLAT_ERROR, DARK (optional), TORT_FLAT, TORT_FLAT_ERROR,
        ILLUMINATION, WAVECAL, SPATCAL, and ORDER_MASK.

        The TORT_FLAT and TORT_FLAT_ERROR extensions are generated with
        `exes.tort`, by undistorting the data and error in the FLAT
        and FLAT_ERROR extensions. The WAVECAL, SPATCAL, and ORDER_MASK
        extensions are generated with `exes.wavecal` from parameters in
        the FLAT extension header.

        If saved to disk, this file can be used in future data reductions
        in place of raw flat and/or dark files.

        Parameters
        ----------
        flat : astropy.io.fits.ImageHDU
            HDU with processed flat data and updated tort parameters
            in header.
        error : astropy.io.fits.ImageHDU
            HDU containing error data associated with the flat array.
        illum : numpy.ndarray
            Illumination array for the undistorted flat.
        filenum : str or list of str
            Input file number(s) to use in the output flat name.
        dark : astropy.io.fits.ImageHDU, optional
            If provided, will be appended to the output flat file.
        save : bool, optional
            If set, the output file is written to disk.

        Returns
        -------
        processed_flat : astropy.io.fits.HDUList
            HDU list containing processed flat extensions.
        """
        from sofia_redux.instruments.exes.wavecal import wavecal
        from sofia_redux.instruments.exes.tort import tort

        header = flat.header.copy()
        hdul = fits.HDUList()
        hdul.append(fits.PrimaryHDU(flat.data, header))
        hdul.append(error)

        # also append torted flat info
        wave_header = header.copy()
        wavemap = wavecal(wave_header)
        rotation = wave_header.get('ROTATION', 0)

        # remove old unit keywords if necessary:
        for key in ['BUNIT1', 'BUNIT2', 'BUNIT3']:
            if key in wave_header:
                del wave_header[key]

        iflat = 1 / flat.data
        iflatvar = error.data ** 2 * iflat ** 4
        tort_flat, tort_flat_var = tort(iflat, header, iflatvar, skew=True)

        if dark is not None:
            hdul.append(dark)

        hdul.append(fits.ImageHDU(rotate90(tort_flat, rotation),
                                  wave_header, name='TORT_FLAT'))
        hdul.append(fits.ImageHDU(rotate90(np.sqrt(tort_flat_var), rotation),
                                  wave_header, name='TORT_FLAT_ERROR'))
        wave_header['BUNIT'] = ''
        hdul.append(fits.ImageHDU(rotate90(illum, rotation),
                                  wave_header, name='ILLUMINATION'))
        wave_header['BUNIT'] = 'cm-1'
        hdul.append(fits.ImageHDU(rotate90(wavemap[0], rotation),
                                  wave_header, name='WAVECAL'))
        wave_header['BUNIT'] = 'arcsec'
        hdul.append(fits.ImageHDU(rotate90(wavemap[1], rotation),
                                  wave_header, name='SPATCAL'))
        wave_header['BUNIT'] = ''
        hdul.append(fits.ImageHDU(rotate90(wavemap[2], rotation).astype((int)),
                                  wave_header, name='ORDER_MASK'))

        outname = self.update_output(hdul, filenum,
                                     self.prodtypes[self.step_index])
        outname = outname.replace('FTA', 'FLT')
        hdul[0].header['PRODTYPE'] = 'flat'
        hdul[0].header['FILENAME'] = os.path.basename(outname)

        # write to disk if desired
        if save:
            self.write_output(hdul, outname)

        return hdul

    def make_flat(self):
        """
        Make a processed flat file.

        If the input data is already a processed flat, it is passed
        through without modification.  Otherwise, `exes.makeflat` is
        called to process the flat file.

        If multiple unprocessed flat files are provided, they are
        mean-combined before processing. If multiple processed flat
        files are provided, only the first one is used.

        The output data from this step includes only the science files
        with new flat extensions attached. Any flat or dark files are
        dropped from further processing.
        """
        from sofia_redux.instruments.exes import makeflat as mf
        param = self.get_parameter_set()

        krot = param.get_value('start_rot')
        spacing = param.get_value('predict_spacing')
        threshold_factor = param.get_value('threshold')

        method = str(param.get_value('edge_method')).lower().strip()
        if method.startswith('sobel'):
            edge_method = 'sobel'
        elif method.startswith('square'):
            edge_method = 'sqderiv'
        else:
            edge_method = 'deriv'

        custom_wavemap = str(param.get_value('custom_wavemap')).strip()
        opt_rot = param.get_value('opt_rot')
        fix_tort = (not opt_rot)

        # Determine number of flats loaded in input
        flats = list()
        flat_filenums = list()
        darks = list()
        dark_filenums = list()
        sources = list()
        source_filenums = list()
        for i, hdul in enumerate(self.input):
            header = hdul[0].header
            obstype = str(header['OBSTYPE']).upper()
            if obstype == 'FLAT':
                flats.append(hdul)
                flat_filenums.append(self.filenum[i])
            elif obstype == 'DARK':
                darks.append(hdul)
                dark_filenums.append(self.filenum[i])
            else:
                sources.append(hdul)
                source_filenums.append(self.filenum[i])

        if len(flats) == 0:
            log.info('No flat file loaded.')
            return
        elif len(flats) > 1:
            # combine flats
            flat_hdul = self._combine_flats(flats)
        else:
            # use the first flat provided
            flat_hdul = flats[0]

        flat_header = flat_hdul[0].header
        flat_prodtype = str(flat_header.get('PRODTYPE',
                                            'UNKNOWN')).upper().strip()
        if flat_prodtype == 'FLAT':
            log.info('Processed flat provided, using as is.')
            flat_data = None
            flat_var = None
        else:
            flat_data = flat_hdul['FLUX'].data
            flat_var = flat_hdul['ERROR'].data ** 2

        if len(darks) == 0:
            dark_header = None
            dark_data = None
            dark_hdu = None
        else:
            # Only use the first dark provided
            dark_hdul = darks[0]
            dark_header = dark_hdul[0].header
            dark_data = np.squeeze(dark_hdul['FLUX'].data)
            dark_hdu = fits.ImageHDU(dark_data, dark_header, name='DARK')
            dark_file = dark_header.get('FILENAME', 'UNKNOWN')
            if len(darks) > 1:
                log.warning(f"More than 1 dark loaded; using the first one "
                            f"({dark_file}).")
            else:
                log.info(f'Using slit dark {dark_file}')

        if flat_data is not None:
            if str(krot).strip() != '':
                flat_header['KROT'] = krot
            if str(spacing).strip() != '':
                flat_header['SPACING'] = spacing
            if str(threshold_factor).strip() != '':
                flat_header['THRFAC'] = threshold_factor

            central_waveno = parse_central_wavenumber(flat_header)

            log.info('\nUsing tort parameters: ')
            log.info(f'Central wavenumber: {central_waveno}')
            labels = {'HR focal length': 'hrfl', 'XD focal length': 'xdfl',
                      'Slit rotation': 'slitrot', 'Det. rotation': 'detrot',
                      'HRR': 'hrr', 'Flat mirror T_amb': 'flattamb',
                      'Flat mirror emissivity': 'flatemis'}
            for label, field in labels.items():
                log.info(f'{label:>22s}: {flat_header[field.upper()]}')
            log.info('')

            kwargs = {'dark': dark_data,
                      'start_pixel': param.get_value('start_pix'),
                      'end_pixel': param.get_value('end_pix'),
                      'top_pixel': param.get_value('top_pix'),
                      'bottom_pixel': param.get_value('bottom_pix'),
                      'fix_tort': fix_tort,
                      'edge_method': edge_method,
                      'custom_wavemap': custom_wavemap}
            for key, value in kwargs.items():
                if str(value).strip() == '':
                    kwargs[key] = None
            try:
                flat_param = mf.makeflat(flat_data, flat_header, flat_var,
                                         **kwargs)
            except ValueError as err:
                log.error('Error in makeflat:')
                log.error(str(err))
                raise ValueError('Error in makeflat') from None

            log.info('')

            # get updated flat header from returned params
            flat_header = flat_param['header']

            # store dark if used
            if dark_data is not None:
                hdinsert(flat_header, 'SLITDRKF',
                         dark_header.get('FILENAME', 'UNKNOWN'),
                         'Slit dark file')
                hdinsert(flat_header, 'SLITDRKR',
                         dark_header.get('RAWFNAME', 'UNKNOWN'),
                         'Slit dark raw file name')

            flat = fits.ImageHDU(flat_param['flat'], flat_header, name='FLAT')
            flat_error = fits.ImageHDU(np.sqrt(flat_param['flat_variance']),
                                       flat_header, name='FLAT_ERROR')
            illum_header = flat_header.copy()
            illum_header['BUNIT'] = ''
            flat_mask = fits.ImageHDU(flat_param['illum'],
                                      illum_header, name='FLAT_ILLUMINATION')

            # write out untorted flat and wavecal map
            uflat = self._write_flat(flat, flat_error, flat_param['illum'],
                                     flat_filenums, dark=dark_hdu,
                                     save=param.get_value('save_flat'))

        else:
            # pre-processed flat
            uflat = flat_hdul
            flat_header = uflat[0].header
            flat = fits.ImageHDU(uflat['FLAT'].data, flat_header, name='FLAT')
            flat_error = fits.ImageHDU(uflat['FLAT_ERROR'].data,
                                       flat_header, name='FLAT_ERROR')
            rotation = flat_header.get('ROTATION', 0)
            illum_header = flat_header.copy()
            illum_header['BUNIT'] = ''
            flat_mask = fits.ImageHDU(
                unrotate90(uflat['ILLUMINATION'].data, rotation),
                illum_header, name='FLAT_ILLUMINATION')

            # allow copying of dark from flat if present
            # and not directly provided
            if dark_hdu is None and 'DARK' in uflat:
                dark_hdu = uflat['DARK']

        # drop flats from input
        self.input = sources
        self.filenum = source_filenums

        # append flat info to sources and save if desired
        copy_keys = [
            # tort and wavecal values
            'DETROT', 'HRFL', 'HRFL0',
            'HRR', 'KROT', 'XDFL', 'XDFL0', 'SLITROT',
            'WNO0', 'WAVENO0',
            # flat calibration values
            'BB_TEMP', 'FLATTAMB', 'FLATEMIS', 'BNU_T',
            # derived spectral values
            'SPACING', 'NT', 'NORDERS',
            'ORDERS', 'XORDER1', 'NBELOW',
            'ORDR_B', 'ORDR_T', 'ORDR_S', 'ORDR_E',
            'SLTH_ARC', 'SLTH_PIX',
            'SLTW_ARC', 'SLTW_PIX', 'SLITWID',
            'ROTATION', 'RP',
            # dark file names
            'SLITDRKF', 'SLITDRKR']
        for i, hdul in enumerate(self.input):
            hdul.append(copy.deepcopy(flat))
            hdul.append(copy.deepcopy(flat_error))
            hdul.append(copy.deepcopy(flat_mask))

            if dark_hdu is not None:
                hdul.append(copy.deepcopy(dark_hdu))

            # update or add important keys from the flat values
            for key in copy_keys:
                if key in flat_header:
                    hdinsert(hdul[0].header, key, flat_header[key],
                             flat_header.comments[key])

            # also add the name of the source flat
            hdinsert(hdul[0].header, 'FLTFILE',
                     uflat[0].header.get('FILENAME', 'UNKNOWN'),
                     'Processed flat file')
            hdinsert(hdul[0].header, 'FLTRFILE',
                     uflat[0].header.get('RAWFNAME', 'UNKNOWN'),
                     'Raw flat file name')

            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])
            if param.get_value('save'):
                self.write_output(hdul, outname)

        # set only the flat in the display data
        self.set_display_data(filenames=[uflat])

    @staticmethod
    def _get_beams(header, flag, n_frames):
        """
        Get the frame indices for the specified beam.

        For INSTMODE = NOD_ON_SLIT or NOD_OFF_SLIT, A and B beams
        alternate.  If NODBEAM = A, A beams are every other frame,
        starting at 0.  If NODBEAM = B, A beams start at 1.

        For INSTMODE = MAP, B beams are the last three frames; A
        beams are all others.

        For any other INSTMODE, all frames are assumed to be A frames.

        Parameters
        ----------
        header : astropy.io.fits.Header
            FITS header for the data, containing INSTMODE and NODBEAM.
        flag : {'A', 'B'}
            The beam to retrieve.  A is on-source or position 1; B
            is off-source or position 2.
        n_frames : int
            The number of frames in the input data.

        Returns
        -------
        beams : array-like of int
            Indices of frames in the specified beam. Size may range
            from 0 to n_frames.
        """
        if n_frames == 0:
            return np.empty(0)
        obs_mode = header['INSTMODE'].upper()
        nod_beam = header['NODBEAM'].upper()
        if obs_mode in ['NOD_OFF_SLIT', 'NOD_ON_SLIT']:
            if n_frames < 2:
                if flag == 'A':
                    beams = np.array([0])
                else:
                    beams = np.array([])
            else:
                if flag == 'A':
                    if nod_beam == 'B':
                        beams = np.arange(n_frames // 2) * 2 + 1
                    else:
                        beams = np.arange(n_frames // 2) * 2
                else:
                    if nod_beam == 'B':
                        beams = np.arange(n_frames // 2) * 2
                    else:
                        beams = np.arange(n_frames // 2) * 2 + 1
        elif obs_mode == 'MAP':
            if n_frames < 5:
                if flag == 'A':
                    beams = np.arange(n_frames)
                else:
                    beams = np.empty(0)
            else:
                if flag == 'A':
                    beams = np.arange(n_frames - 3)
                else:
                    beams = np.arange(n_frames - 3, n_frames)
        else:
            if flag == 'A':
                beams = np.arange(n_frames)
            else:
                beams = np.empty(0)
        return beams

    def despike(self):
        """
        Flag temporal outliers (spikes).

        Calls `sofia_redux.instruments.exes.despike` on each input file.

        Typically, A beams within are compared separately from B beams,
        but if desired, the beam designation can be ignored and all
        frames compared together.

        It is also possible to combine all input files before running
        despike. In this case, all input will be treated as a single
        file in all subsequent steps.
        """
        from sofia_redux.instruments.exes import despike as ed

        log.info('Running despike')
        param = self.get_parameter_set()

        save = param.get_value('save')
        mark_trashed = param.get_value('mark_trash')
        spike_factor = param.get_value('spike_fac')
        combine_all = param.get_value('combine_all')
        ignore_beams = param.get_value('ignore_beams')
        propagate_nan = param.get_value('propagate_nan')

        # check for missing input, following flat step
        if len(self.input) == 0:
            err = 'No source files loaded'
            log.error(err)
            self.error = err
            return

        results = list()
        if spike_factor == 0:
            log.info('Spike factor is 0; despike not applied')
            return

        if combine_all:
            self._combine_nod_pairs(mask2d=True)

        # Despike each image
        for i, hdul in enumerate(self.input):
            data = hdul[0].data
            var = hdul['ERROR'].data ** 2
            mask = hdul['MASK'].data
            good_data = ~(mask.astype(bool))
            header = hdul[0].header

            log.info('')
            log.info(f"Input: {header['FILENAME']}")

            if data.ndim > 2:
                n_frames = data.shape[0]
            else:
                n_frames = 1

            if ignore_beams:
                a_beams = np.arange(n_frames)
                b_beams = np.empty(0)
            else:
                a_beams = self._get_beams(header, 'A', n_frames)
                b_beams = self._get_beams(header, 'B', n_frames)

            hdinsert(header, 'SPIKEFAC', spike_factor,
                     comment='Spike factor')
            if not mark_trashed:
                hdinsert(header, 'TRASH', 0.0)

            try:
                output = ed.despike(data, header, var,
                                    a_beams, b_beams,
                                    good_data,
                                    propagate_nan=propagate_nan)
                spike_data, spike_mask, good_frames = output
            except (RuntimeError, ValueError) as err:
                log.error('Error in despike algorithm:')
                log.error(str(err))
                raise ValueError('Error in despike') from None

            hdul['FLUX'].data = spike_data
            hdul['ERROR'].data = np.sqrt(var)

            # mark trashed frames if desired
            if mark_trashed:
                if n_frames == 1:
                    spike_mask[:] = False
                else:
                    for j in range(n_frames):
                        if j not in good_frames:
                            spike_mask[j][:] = False

            if propagate_nan:
                # mark spikes as bad
                hdul['MASK'].data = (~spike_mask).astype(int)
                hdul['FLUX'].data[~spike_mask] = np.nan
                hdul['ERROR'].data[~spike_mask] = np.nan
            else:
                # just modify mask to match data frames
                hdul['MASK'].data = np.zeros(hdul['FLUX'].data.shape,
                                             dtype=int)
                hdul['MASK'].data[:] = mask

            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])
            if save:
                self.write_output(hdul, outname)

            # check for completely trashed file and drop from reduction
            if np.all(hdul['MASK'].data):
                log.info('')
                log.warning(f"All data trashed for file "
                            f"{hdul[0].header['FILENAME']}; "
                            f"dropping from reduction.")
            else:
                results.append(hdul)

        self.input = results
        self.set_display_data()

    def debounce(self):
        """
        Correct for optical shifts (bounces).

        Calls `sofia_redux.instruments.exes.debounce`.
        """
        import sofia_redux.instruments.exes.debounce as ed
        param = self.get_parameter_set()
        save = param.get_value('save')
        bounce_factor = param.get_value('bounce_fac')
        spectral_bounce = param.get_value('spec_direction')

        if bounce_factor == 0:
            log.info('Bounce factor is 0; debounce not applied.')
            return

        results = list()
        for i, hdul in enumerate(self.input):
            data = hdul['FLUX'].data
            variance = hdul['ERROR'].data ** 2
            header = hdul[0].header
            mask = ~(hdul['MASK'].data.astype(bool))
            flat = hdul['FLAT'].data
            illum = hdul['FLAT_ILLUMINATION'].data

            # store the bounce factor in the header:
            # the debounce algorithm will pick it up from there
            header['BOUNCE'] = bounce_factor

            # get beams
            if data.ndim > 2:
                n_frames = data.shape[0]
            else:
                n_frames = 1
            a_beams = self._get_beams(header, 'A', n_frames)
            b_beams = self._get_beams(header, 'B', n_frames)

            # debounce
            data = ed.debounce(data, header, a_beams, b_beams, flat,
                               mask, illum, variance, spectral=spectral_bounce)

            # store data
            hdul['FLUX'].data = data

            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])
            if save:
                self.write_output(hdul, outname)

            results.append(hdul)
        self.input = results
        self.set_display_data()

    def subtract_nods(self):
        """
        Subtract B nods from A nods for background correction.

        Calls `sofia_redux.instruments.exes.diff_arr` to do the array
        subtraction.

        If desired, for INSTMODE = NOD_OFF_SLIT, `exes.cirrus`
        may be additionally called to correct for residual background.
        """

        from sofia_redux.instruments.exes import cirrus as ec
        from sofia_redux.instruments.exes import diff_arr as ed
        param = self.get_parameter_set()

        save = param.get_value('save')
        skip_nod = param.get_value('skip_nod')
        subtract_bg = param.get_value('subtract_sky')
        subtract_dark = param.get_value('subtract_dark')

        # Subtract nods for each image
        results = list()
        for i, hdul in enumerate(self.input):
            header = hdul[0].header
            data = hdul['FLUX'].data
            variance = hdul['ERROR'].data ** 2
            flat = hdul['FLAT'].data
            mask = hdul['MASK'].data
            if 'DARK' in hdul:
                dark = hdul['DARK'].data
            else:
                dark = None

            if data.ndim > 2:
                n_frames = data.shape[0]
            else:
                n_frames = 1

            a_beams = self._get_beams(header, 'A', n_frames)
            b_beams = self._get_beams(header, 'B', n_frames)

            no_dark = (subtract_dark and dark is None)
            if no_dark:
                log.warning('No dark frame available, skipping '
                            'nod subtraction instead.')
            if skip_nod or no_dark:
                if i == 0:
                    log.info('Skipping sky subtraction')
            elif len(b_beams) == 0 and not subtract_dark:
                log.info(f'No B beams identified. Not subtracting nods '
                         f'for {os.path.basename(header["FILENAME"])}')
            else:
                if subtract_bg:
                    if header['INSTMODE'] == 'NOD_OFF_SLIT':
                        log.info('Subtracting continuum signal')
                        data = ec.cirrus(data, header, a_beams, b_beams, flat)
                if subtract_dark and dark is not None:
                    black_dark = True
                    if len(b_beams) == 0:
                        b_beams = a_beams.copy()
                else:
                    black_dark = False
                data, variance, mask = ed.diff_arr(
                    data, header, a_beams, b_beams,
                    variance, mask=mask,
                    dark=dark, black_dark=black_dark)

            # Get output filenames
            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])

            # remove first empty dimension if necessary
            hdul['FLUX'].data = np.squeeze(data)
            hdul['ERROR'].data = np.sqrt(np.squeeze(variance))
            hdul['MASK'].data = np.squeeze(mask)

            # remove dark if present - no longer needed
            if 'DARK' in hdul:
                del hdul['DARK']

            if save:
                self.write_output(hdul, outname)

            results.append(hdul)
        self.input = results
        self.set_display_data()

    def flat_correct(self):
        """
        Calibrate and flat correct science data.

        Input data must have FLAT extensions attached.

        Calls `sofia_redux.instruments.exes.calibrate`.
        """
        import sofia_redux.instruments.exes.calibrate as ec
        param = self.get_parameter_set()
        save = param.get_value('save')
        skip_flat = param.get_value('skip_flat')

        if skip_flat:
            log.info('Skipping flat correction.')
            return

        results = list()
        for i, hdul in enumerate(self.input):
            data = hdul['FLUX'].data
            variance = hdul['ERROR'].data ** 2
            header = hdul[0].header
            flat = hdul['FLAT'].data
            flat_var = hdul['FLAT_ERROR'].data ** 2

            data, variance = ec.calibrate(data, header, flat,
                                          variance, flat_var)

            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])

            hdul['FLUX'].data = data
            hdul['ERROR'].data = np.sqrt(variance)
            hdinsert(header, 'BUNIT', 'erg s-1 cm-2 sr-1 (cm-1)-1',
                     'Data units')
            hdinsert(hdul[0].header, 'RAWUNITS', 'ct',
                     'Data units before calibration')
            hdinsert(hdul['ERROR'].header, 'BUNIT',
                     'erg s-1 cm-2 sr-1 (cm-1)-1', 'Data units')

            if save:
                self.write_output(hdul, outname)

            results.append(hdul)
        self.input = results
        self.set_display_data()

    def clean_badpix(self):
        """
        Clean or flag bad pixels.

        Calls `sofia_redux.instruments.exes.clean`.
        """
        import sofia_redux.instruments.exes.clean as ec
        param = self.get_parameter_set()
        save = param.get_value('save')
        bp_threshold = param.get_value('bp_threshold')
        propagate_nan = param.get_value('propagate_nan')

        results = list()
        for i, hdul in enumerate(self.input):
            data = hdul['FLUX'].data
            error = hdul['ERROR'].data
            mask = ~(hdul['MASK'].data.astype(bool))
            header = hdul[0].header

            if data.ndim == 2:
                data, error = ec.clean(data, header, error, mask,
                                       threshold=bp_threshold,
                                       propagate_nan=propagate_nan)
            else:
                nframe = data.shape[0]
                for j in range(nframe):
                    data[j], error[j] = ec.clean(data[j], header,
                                                 error[j], mask[j],
                                                 threshold=bp_threshold,
                                                 propagate_nan=propagate_nan)

            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])

            hdul['FLUX'].data = data
            hdul['ERROR'].data = error

            if save:
                self.write_output(hdul, outname)

            results.append(hdul)
        self.input = results
        self.set_display_data()

    def undistort(self):
        """
        Correct for optical distortion.

        Calls `sofia_redux.instruments.exes.tort` to do the distortion
        correction and `sofia_redux.instruments.exes.wavecal` to generate
        wavenumber and spatial calibration images.

        After this step, all spectral images are rotated as needed
        to align the spectral axis with the x-axis.
        """
        from sofia_redux.instruments.exes.tort import tort
        from sofia_redux.instruments.exes.wavecal import wavecal

        param = self.get_parameter_set()
        save = param.get_value('save')
        interpolation_method = param.get_value('interpolation_method')
        spline_order = param.get_value('spline_order')
        block = param.get_value('block_unilluminated')

        results = list()
        for i, hdul in enumerate(self.input):
            header = hdul[0].header
            data = hdul['FLUX'].data
            variance = hdul['ERROR'].data ** 2
            flat_header = hdul['FLAT'].header
            flat = hdul['FLAT'].data
            flat_var = hdul['FLAT_ERROR'].data ** 2

            # set 0 values in flat to NaN so they don't get
            # combined with real data in interpolation
            flat[flat == 0] = np.nan

            data, variance = tort(data, header, variance, skew=True,
                                  interpolation_method=interpolation_method,
                                  order=spline_order)
            flat, flat_var = tort(flat, flat_header, flat_var, skew=True,
                                  interpolation_method=interpolation_method,
                                  order=spline_order)

            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])

            # get wavecal map for undistorted pixels
            wave_header = header.copy()
            wavemap = wavecal(wave_header)
            rotation = wave_header.get('ROTATION', 0)

            # remove old keywords if necessary:
            del header['ROTATION']
            for key in ['BUNIT1', 'BUNIT2', 'BUNIT3', 'ROTATION']:
                if key in wave_header:
                    del wave_header[key]

            # rotate data and block any unilluminated pixels
            hdul['FLAT'].data = rotate90(flat, rotation)
            hdul['FLAT_ERROR'].data = rotate90(np.sqrt(flat_var), rotation)
            illum = rotate90(hdul['FLAT_ILLUMINATION'].data, rotation)
            hdul['FLAT_ILLUMINATION'].data = illum
            order_mask = rotate90(wavemap[2], rotation).astype((int))
            out_of_range = (illum != 1) | (order_mask == 0)
            if data.ndim == 2:
                hdul['FLUX'].data = rotate90(data, rotation)
                hdul['ERROR'].data = rotate90(np.sqrt(variance), rotation)

                if block:
                    hdul['FLUX'].data[out_of_range] = np.nan
                    hdul['ERROR'].data[out_of_range] = np.nan
            else:
                for j in range(data.shape[0]):
                    hdul['FLUX'].data[j] = rotate90(data[j], rotation)
                    hdul['ERROR'].data[j] = rotate90(np.sqrt(variance[j]),
                                                     rotation)
                    if block:
                        hdul['FLUX'].data[j, out_of_range] = np.nan
                        hdul['ERROR'].data[j, out_of_range] = np.nan

            hdul['MASK'].data = np.zeros(hdul['FLUX'].data.shape)
            hdul['MASK'].data[np.isnan(hdul['FLUX'].data)] = 1

            wave_header['BUNIT'] = 'cm-1'
            hdul.append(fits.ImageHDU(
                rotate90(wavemap[0], rotation),
                wave_header, name='WAVECAL'))
            wave_header['BUNIT'] = 'arcsec'
            hdul.append(fits.ImageHDU(
                rotate90(wavemap[1], rotation),
                wave_header, name='SPATCAL'))
            wave_header['BUNIT'] = ''
            hdul.append(fits.ImageHDU(
                rotate90(wavemap[2], rotation).astype((int)),
                wave_header, name='ORDER_MASK'))

            if save:
                self.write_output(hdul, outname)

            results.append(hdul)
        self.input = results
        self.set_display_data()

    def correct_calibration(self):
        """
        Correct calibration for blackbody variation by wavenumber.

        The `flat_correct` step calibrates the spectral flux to
        physical units based on the blackbody function at the
        central wavenumber in the flat. This step corrects the
        calibration for the variation of the blackbody with respect
        to wavenumber values at each pixel in the spectral image.

        Calls `sofia_redux.instruments.exes.makeflat.bnu` on wavenumber
        values in the WAVECAL extension to determine a correction image
        for the FLUX. This correction image is directly multiplied into
        the FLUX data array.
        """
        from sofia_redux.instruments.exes.makeflat import bb_cal_factor

        param = self.get_parameter_set()
        save = param.get_value('save')
        skip_correction = param.get_value('skip_correction')

        if skip_correction:
            log.info('Skipping calibration correction.')
            return

        results = list()
        for i, hdul in enumerate(self.input):
            header = hdul[0].header
            data = hdul['FLUX'].data
            error = hdul['ERROR'].data
            wavecal = hdul['WAVECAL'].data

            # use the wavecal map to correct the blackbody calibration
            # by wavenumber value
            wno0 = parse_central_wavenumber(header)
            norm = bb_cal_factor(wno0, header['BB_TEMP'],
                                 header['FLATTAMB'], header['FLATEMIS'])
            bb_image = bb_cal_factor(wavecal, header['BB_TEMP'],
                                     header['FLATTAMB'], header['FLATEMIS'])

            hdul['FLUX'].data = bb_image * data / norm
            hdul['ERROR'].data = bb_image * error / norm

            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])
            if save:
                self.write_output(hdul, outname)

            results.append(hdul)
        self.input = results
        self.set_display_data()

    def _blank_frames(self, bad_frames):
        """
        Set bad frames in input data cubes to NaN.

        Parameters
        ----------
        bad_frames : list of int
            Indices for bad frames.
        """
        excluded = False
        for i, hdul in enumerate(self.input):
            data = hdul['FLUX'].data
            error = hdul['ERROR'].data
            filename = hdul[0].header.get('FILENAME', 'UNKNOWN')

            if data.ndim > 2:
                n_frames = data.shape[0]
            else:
                n_frames = 1

            for bad_frame in bad_frames[i]:
                n = int(bad_frame)
                if 1 <= n <= n_frames:
                    log.info(f'Excluding frame {n} from file {filename}')
                    excluded = True
                    if n_frames == 1:
                        data[:] = np.nan
                        error[:] = np.nan
                    else:
                        data[n - 1] = np.nan
                        error[n - 1] = np.nan
        if not excluded:
            log.warning('No valid frame numbers passed; '
                        'no exclusions performed.')
        log.info('')

    def _split_nod_pairs(self):
        """Split each input file into new files, one per frame."""
        results = []
        filenums = []
        for i, hdul in enumerate(self.input):
            data = hdul['FLUX'].data
            error = hdul['ERROR'].data
            flat = hdul['FLAT'].data
            illum = hdul['FLAT_ILLUMINATION'].data
            base_filenum = self.filenum[i]

            if data.ndim > 2:
                n_frames = data.shape[0]
            else:
                n_frames = 1

            if n_frames == 1:
                results.append(hdul)
                filenums.append(base_filenum)
            else:
                for j in range(n_frames):
                    new_hdul = copy.deepcopy(hdul)
                    new_hdul['FLUX'].data = data[j]
                    new_hdul['ERROR'].data = error[j]
                    new_hdul['FLAT'].data = flat.copy()
                    new_hdul['FLAT_ILLUMINATION'].data = illum.copy()

                    new_filenum = f'{base_filenum}_{j + 1}'
                    self.update_output(new_hdul, new_filenum,
                                       new_hdul[0].header['PRODTYPE'])
                    results.append(new_hdul)
                    filenums.append(new_filenum)

        self.input = results
        self.filenum = filenums

    def _combine_nod_pairs(self, mask2d=False):
        """Combine all input files into a single file, stacking frames."""
        # nothing to do if less than 2 files
        n_files = len(self.input)
        if n_files < 2:
            return

        # start with the first input file
        new_hdul = copy.deepcopy(self.input[0])

        # loop through the rest, appending data and error values
        new_data = []
        new_error = []
        new_mask = []
        headers = []
        for i, hdul in enumerate(self.input):
            data = hdul['FLUX'].data
            error = hdul['ERROR'].data
            mask = hdul['MASK'].data

            if data.ndim == 2:
                data = np.array([data])
                error = np.array([error])
                if not mask2d:
                    mask = np.array([mask])

            if i > 0:
                if data.shape[-2:] != new_data[0].shape[-2:]:
                    raise ValueError(f"Data in {hdul[0].header['FILENAME']} "
                                     f"does not match dimensions in "
                                     f"{new_hdul[0].header['FILENAME']}; "
                                     f"files cannot be coadded.")
            new_data.append(data)
            new_error.append(error)
            new_mask.append(mask)
            headers.append(hdul[0].header)

        # stack the arrays along the first axis
        new_hdul['FLUX'].data = np.concatenate(new_data, axis=0)
        new_hdul['ERROR'].data = np.concatenate(new_error, axis=0)

        if mask2d:
            new_hdul['MASK'].data = np.any(new_mask, axis=0).astype(int)
        else:
            new_hdul['MASK'].data = np.concatenate(new_mask, axis=0)

        # merge all headers
        new_hdul[0].header = mergehdr(headers)

        self.input = [new_hdul]

        # all file numbers now used for single file
        self.filenum = [self.filenum]

    def coadd_pairs(self):
        """
        Coadd nod pairs.

        Calls `sofia_redux.instruments.exes.coadd` to mean combine input
        frames in each file.

        Optionally, all files may be combined into a single file
        to be coadded together. Aternatively, all files may be split
        into separate files, one per frame, so that no frames are coadded.

        Prior to the coadd, it is possible to apply a small spatial shift
        to align frames (`exes.spatial_shift`). It is also possible to
        subtract the mean value at each column to correct residual background
        levels, if INSTMODE = NOD_ON_SLIT.

        To inspect the effect of these options, it is possible to save
        intermediate pre-coadd files to disk. If saved, they will have
        PRODTYPE = coadd_input and file code COI.
        """
        from sofia_redux.instruments.exes.coadd import coadd
        from sofia_redux.instruments.exes.spatial_shift import spatial_shift
        from sofia_redux.instruments.exes.submean import submean

        param = self.get_parameter_set()
        save = param.get_value('save')
        save_intermediate = param.get_value('save_intermediate')
        subtract_sky = param.get_value('subtract_sky')
        do_shift = param.get_value('shift')
        shift_method = str(param.get_value('shift_method')).lower().strip()
        skip_coadd = param.get_value('skip_coadd')
        coadd_all = param.get_value('coadd_all_files')
        exclude_pairs = param.get_value('exclude_pairs')
        threshold = param.get_value('threshold')
        weight_method = str(param.get_value('weight_method')).lower().strip()
        override_weights = param.get_value('override_weights')

        # parse weighting method from:
        #  options = ['Uniform weights', 'Weight by flat',
        #             'Weight by variance']
        if 'flat' in weight_method:
            weight_mode = None
            stdwt = False
        elif 'variance' in weight_method:
            weight_mode = None
            stdwt = True
        else:
            weight_mode = 'unweighted'
            stdwt = True

        # parse shift method from:
        #  options = ['Maximize signal-to-noise',
        #             'Maximize signal (sharpen)']
        if 'sharp' in shift_method:
            sharpen = True
        else:
            sharpen = False

        # set any bad frames to NaN
        if str(exclude_pairs).strip() != '':
            error_message = ['Could not read exclude pairs '
                             f"parameter: '{exclude_pairs}'",
                             'Values should be comma-separated integers, '
                             'starting with 1 for the first frame.',
                             'To specify different values for different '
                             'input files, provide a semi-colon separated '
                             'list matching the number of input files.']
            bad_frames = parse_apertures(exclude_pairs, len(self.input),
                                         error_message=error_message,
                                         allow_empty=True)
            self._blank_frames(bad_frames)

        # parse override weights if provided
        overrides = None
        if str(override_weights).strip() != '':
            error_message = [f'Could not read override weights '
                             f"parameter: '{override_weights}'",
                             'Values should be comma-separated numbers, '
                             'one for each frame.',
                             'To specify different values for different '
                             'input files, provide a semi-colon separated '
                             'list matching the number of input files.']
            overrides = parse_apertures(override_weights, len(self.input),
                                        error_message=error_message,
                                        allow_empty=False)

        # split or combine nod pairs across files if desired
        if skip_coadd:
            self._split_nod_pairs()

            # weights are now irrelevant
            overrides = None
        elif coadd_all:
            self._combine_nod_pairs()

            # flatten weights list
            if overrides is not None:
                all_weights = []
                for w in overrides:
                    all_weights.extend(w)
                overrides = [all_weights]

        # loop through all files, coadding nod pairs within each
        results = list()
        for i, hdul in enumerate(self.input):
            log.info('')
            header = hdul[0].header
            data = hdul['FLUX'].data
            variance = hdul['ERROR'].data ** 2
            flat = hdul['FLAT'].data
            illum = hdul['FLAT_ILLUMINATION'].data
            order_mask = hdul['ORDER_MASK'].data

            # get number of frames, reshape data if necessary
            if data.ndim > 2:
                n_frames = data.shape[0]
            else:
                n_frames = 1
                data = np.array([data])
                variance = np.array([variance])

            # subtract background before coadd if desired
            instmode = str(header.get('INSTMODE')).lower()
            if subtract_sky and 'on_slit' in instmode:
                data = submean(data, header, flat, illum, order_mask)

            # get good frames
            frame_mask = np.any(~np.isnan(data), axis=(1, 2))
            good_frames = np.arange(n_frames)[frame_mask]

            # shift data before coadd if desired
            if do_shift and n_frames > 1:
                data, variance = spatial_shift(
                    data, header, flat, variance,
                    illum=illum, good_frames=good_frames, sharpen=sharpen)
                log.info('')

            # save modified data prior to coadd, if desired
            if save_intermediate:
                log.info('Saving intermediate file:')

                coi_hdul = copy.deepcopy(hdul)
                outname = self.update_output(coi_hdul, self.filenum[i],
                                             self.prodtypes[self.step_index])
                coi_header = coi_hdul[0].header
                coi_outname = outname.replace('COA', 'COI')
                coi_header['FILENAME'] = os.path.basename(coi_outname)
                coi_header['PRODTYPE'] = 'coadd_input'
                coi_hdul[0].header = coi_header
                coi_hdul['FLUX'].data = data.copy()
                coi_hdul['ERROR'].data = np.sqrt(variance)

                self.write_output(coi_hdul, coi_outname)
                log.info('')

            if overrides is not None:
                weights = (np.array(overrides[i], dtype=float)
                           / np.sum(overrides[i]))
                mode = 'useweights'
            else:
                weights = None
                mode = weight_mode
            cdata, cvar = coadd(data, header, flat, variance,
                                threshold=threshold,
                                illum=illum, good_frames=good_frames,
                                std_wt=stdwt, weight_mode=mode,
                                weights=weights)

            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])

            hdul['FLUX'].data = cdata
            hdul['ERROR'].data = np.sqrt(cvar)

            hdul['MASK'].data = np.zeros(hdul['FLUX'].data.shape)
            hdul['MASK'].data[np.isnan(hdul['FLUX'].data)] = 1

            if save:
                self.write_output(hdul, outname)

            results.append(hdul)
        self.input = results
        self.set_display_data()

    def convert_units(self):
        """
        Convert calibrated units to flux in Jy/pixel.

        Uses the OMEGAP keyword to correct for the solid angle per pixel.
        """
        param = self.get_parameter_set()
        save = param.get_value('save')
        skip_conversion = param.get_value('skip_conversion')
        cal_factor = param.get_value('cal_factor')
        zero_level = param.get_value('zero_level')

        if skip_conversion:
            log.info('Skipping unit conversion.')
            return

        results = list()
        for i, hdul in enumerate(self.input):

            conversion = hdul[0].header['OMEGAP'] * 1e21 / const.c.value

            for extname in ['FLUX', 'ERROR', 'FLAT', 'FLAT_ERROR']:
                hdu = hdul[extname]
                hdu.data *= conversion * cal_factor
                if 'ERROR' not in extname:
                    hdu.data += zero_level
                if 'FLAT' in extname:
                    hdu.header['BUNIT'] = 'Jy/(pixel ct)'
                else:
                    hdu.header['BUNIT'] = 'Jy/pixel'

            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])
            if save:  # pragma: no cover
                self.write_output(hdul, outname)

            results.append(hdul)
        self.input = results
        self.set_display_data()

    # spectroscopy module pipeline steps and helpers

    def make_profiles(self):
        """
        Produce spatial profile fits from rectified images.

        The rectified images and profiles are stored in self.input
        for continued processing.

        Calls `sofia_redux.spectroscopy.mkspatprof` and
        `sofia_redux.spectroscopy.rectify`.
        """
        from sofia_redux.spectroscopy.mkspatprof import mkspatprof
        from sofia_redux.spectroscopy.rectify import rectify

        # get parameters
        param = self.get_parameter_set()
        fit_order = param.get_value('fit_order')
        bg_sub = param.get_value('bg_sub')

        # loop through input, making profiles
        results = []
        for i, hdul in enumerate(self.input):
            header = hdul[0].header
            data = hdul['FLUX'].data
            var = hdul['ERROR'].data ** 2
            omask = hdul['ORDER_MASK'].data
            wavecal = hdul['WAVECAL'].data
            spatcal = hdul['SPATCAL'].data
            flat = hdul['FLAT'].data

            log.info(f"Input: {header['FILENAME']}")
            log.info(f"Profiling {header['NORDERS']} order(s)")

            # extract rectified images for each order
            units = fits.Header({'XUNITS': 'cm-1'})
            rectimg = rectify(data, omask, wavecal, spatcal,
                              variance=var, poly_order=0, header=units)

            # make the spatial profile from the rectified images
            medprof, fitprof = \
                mkspatprof(rectimg, return_fit_profile=True,
                           ndeg=fit_order, bgsub=bg_sub, smooth_sigma=None,
                           robust=3.0)

            # extract matching rectified flat images for each order
            rectflat = rectify(flat, omask, wavecal, spatcal, poly_order=0)

            # keep the existing extensions, add rectified data for each order
            for order in sorted(rectimg.keys()):
                ordnum = f'{order:02d}'
                rectified = rectimg[order]
                rectified_flat = rectflat[order]
                if rectified is None or rectified_flat is None:
                    raise ValueError('Problem in rectification.')

                new_image = rectified['image']
                new_error = np.sqrt(rectified['variance'])
                new_flat = rectified_flat['image']

                # make an extension header with spectral WCS
                exthead = fits.Header()
                for card in rectified['header'].cards:
                    hdinsert(exthead, card.keyword, card.value, card.comment)

                # flux, error, flat
                hdinsert(exthead, 'BUNIT',
                         header.get('BUNIT', 'UNKNOWN'),
                         'Data units')
                hdul.append(fits.ImageHDU(new_image, exthead,
                                          name=f'FLUX_ORDER_{ordnum}'))
                hdul.append(fits.ImageHDU(new_error, exthead,
                                          name=f'ERROR_ORDER_{ordnum}'))
                hdinsert(exthead, 'BUNIT',
                         hdul['FLAT'].header.get('BUNIT', 'UNKNOWN'),
                         'Data units')
                hdul.append(fits.ImageHDU(new_flat, exthead,
                                          name=f'FLAT_ORDER_{ordnum}'))

                # append extra information from rectification
                boolmask = rectified['mask']
                mask = np.zeros(boolmask.shape, dtype=int)
                mask[~boolmask] = 1
                hdinsert(exthead, 'BUNIT', '', 'Data units')
                hdul.append(fits.ImageHDU(mask, exthead,
                                          name=f'BADMASK_ORDER_{ordnum}'))

                exthead = fits.Header()
                xunit = 'cm-1'
                hdinsert(exthead, 'BUNIT', xunit, 'Data units')
                hdul.append(fits.ImageHDU(rectified['wave'], exthead,
                                          name=f'WAVEPOS_ORDER_{ordnum}'))

                yunit = 'arcsec'
                hdinsert(exthead, 'BUNIT', yunit, 'Data units')
                hdul.append(fits.ImageHDU(rectified['spatial'], exthead,
                                          name=f'SLITPOS_ORDER_{ordnum}'))

                # add profiles to HDUList
                hdinsert(exthead, 'BUNIT', '', 'Data units')
                hdul.append(
                    fits.ImageHDU(fitprof[order], exthead,
                                  name=f'SPATIAL_MAP_ORDER_{ordnum}'))
                hdul.append(
                    fits.ImageHDU(medprof[order], exthead,
                                  name=f'SPATIAL_PROFILE_ORDER_{ordnum}'))

            # remove flat error and illumination -
            # no longer needed
            if 'FLAT_ERROR' in hdul:
                del hdul['FLAT_ERROR']
            if 'FLAT_ILLUMINATION' in hdul:
                del hdul['FLAT_ILLUMINATION']

            # save if desired
            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])
            if param.get_value('save'):
                self.write_output(hdul, outname)

            results.append(hdul)
            log.info('')

        self.input = results
        self.set_display_data()

    def locate_apertures(self):
        """
        Automatically find aperture centers.

        Calls `sofia_redux.spectroscopy.findapertures`.
        """
        from sofia_redux.spectroscopy.findapertures import find_apertures

        # get parameters
        param = self.get_parameter_set()
        method = param.get_value('method')
        num_aps = param.get_value('num_aps')
        input_position = param.get_value('input_position')
        fwhm_par = param.get_value('fwhm')
        exclude_orders = param.get_value('exclude_orders')

        if str(method).strip().lower() == 'fix to center':
            log.info('Fixing aperture to slit center.')
            positions = None
            fix_ap = True
            num_aps = 1
        elif str(method).strip().lower() == 'fix to input':
            log.info('Fixing aperture to input positions.')
            positions = parse_apertures(
                input_position, len(self.input))
            fix_ap = True
        else:
            log.info('Finding aperture positions from Gaussian fits.')
            if str(input_position).strip() == '':
                positions = None
            else:
                positions = parse_apertures(
                    input_position, len(self.input))
            fix_ap = False

        # parse orders, applying to all files
        exclude = list()
        if str(exclude_orders).strip() != '':
            try:
                exclude = [int(i) for i in exclude_orders.split(',')]
            except (ValueError, TypeError, IndexError):
                raise ValueError('Invalid order exclusion '
                                 'parameter.') from None

        log.info('')
        log.info('Apertures:')

        profile = dict()
        if positions is not None:
            guess = dict()
            npeaks = len(positions[0])
        else:
            guess = None
            npeaks = num_aps
        results = []
        fit_fwhm = []
        yunit = 'arcsec'
        for i, hdul in enumerate(self.input):
            for j in range(hdul[0].header['NORDERS']):
                if j + 1 in exclude:
                    log.info(f'Skipping order {j + 1:02d}')
                    continue
                ordnum = f'{j + 1:02d}'
                profile[j + 1] = [hdul[f'SLITPOS_ORDER_{ordnum}'].data,
                                  hdul[f'SPATIAL_PROFILE_ORDER_{ordnum}'].data]
                if positions is not None:
                    guess[j + 1] = positions[i]

            apertures = find_apertures(profile, npeaks=npeaks, positions=guess,
                                       fwhm=fwhm_par, fix=fix_ap,
                                       box_width=('stddev', 3))

            # update output name
            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])

            log.info('')
            log.info(f"  {hdul[0].header['FILENAME']}")
            for order in sorted(apertures.keys()):
                ordnum = f'{order:02d}'
                appos = []
                apsign = []
                apfwhm = []
                fwhm_list = []
                for ap in apertures[order]:
                    pos = ap['position']
                    sign = ap['sign']
                    fwhm = ap['fwhm']
                    appos.append(f'{pos:.3f}')
                    apsign.append(f'{sign:d}')
                    apfwhm.append(f'{fwhm:.3f}')
                    if fix_ap:
                        log.info('    Order {}: {:.3f} {} '
                                 '(sign: {})'.format(ordnum, pos, yunit, sign))
                    else:
                        log.info('    Order {}: {:.3f} {} '
                                 '(sign: {}, fit '
                                 'FWHM: {:.3f})'.format(ordnum, pos, yunit,
                                                        sign, fwhm))
                        fwhm_list.append(fwhm)
                fit_fwhm.extend(fwhm_list)

                # add apertures to primary header
                hdinsert(hdul[0].header, f'APPOSO{ordnum}', ','.join(appos),
                         comment=f'Aperture positions [{yunit}]')
                hdinsert(hdul[0].header, f'APSGNO{ordnum}', ','.join(apsign),
                         comment='Aperture signs')

                # add FWHM to primary header
                if not fix_ap:
                    comment = f'Fit aperture FWHM [{yunit}]'
                else:
                    comment = f'Assumed aperture FWHM [{yunit}]'
                hdinsert(hdul[0].header,
                         f'APFWHM{ordnum}', ','.join(apfwhm),
                         comment=comment)

                # also add to order flux extension header,
                # labeled as order 1
                exthead = hdul[f'FLUX_ORDER_{ordnum}'].header
                hdinsert(exthead, 'APPOSO01', ','.join(appos),
                         comment=f'Aperture positions [{yunit}]')
                hdinsert(exthead, 'APSGNO01', ','.join(apsign),
                         comment='Aperture signs')
                hdinsert(exthead, 'APFWHM01', ','.join(apfwhm),
                         comment=comment)

            log.info('')

            # save if desired
            if param.get_value('save'):
                self.write_output(hdul, outname)
            results.append(hdul)

        if len(fit_fwhm) > 0:
            log.info('')
            log.info(f'Mean fit FWHM: '
                     f'{np.mean(fit_fwhm):.2f} '
                     f'+/- {np.std(fit_fwhm):.2f} {yunit}')

        self.input = results
        self.set_display_data()

    def set_apertures(self):
        """
        Set aperture radii.

        Calls `sofia_redux.spectroscopy.getapertures` and
        `sofia_redux.spectroscopy.mkapmask`.
        """
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
        apstart_list = param.get_value('ap_start')
        apend_list = param.get_value('ap_end')

        manual_ap = (str(apstart_list).strip().lower() != ''
                     and str(apend_list).strip().lower() != '')
        if manual_ap:
            full_slit = False

        fix_apsign, fix_aprad, fix_psfrad, fix_bgr, fix_ap = \
            False, False, False, False, False
        if not full_slit:
            if str(apsign_list).strip().lower() != '':
                apsign_list = parse_apertures(apsign_list,
                                              len(self.input))
                fix_apsign = True
            if str(aprad_list).strip().lower() != '':
                aprad_list = parse_apertures(aprad_list,
                                             len(self.input))
                fix_aprad = True
            if str(psfrad_list).strip().lower() != '':
                psfrad_list = parse_apertures(psfrad_list,
                                              len(self.input))
                fix_psfrad = True
            if str(bgr_list).strip().lower() != '':
                if str(bgr_list).strip().lower() == 'none':
                    bgr_list = []
                    fix_bgr = True
                else:
                    bgr_list = parse_bg(bgr_list, len(self.input))
                    fix_bgr = True

            # override apertures if start/end set
            if manual_ap:
                apstart_list = parse_apertures(apstart_list,
                                               len(self.input))
                apend_list = parse_apertures(apend_list,
                                             len(self.input))
                fix_aprad = False
                fix_psfrad = False
                fix_ap = True

        results = []
        for i, hdul in enumerate(self.input):
            log.info('')
            header = hdul[0].header
            log.info(header['FILENAME'])

            # loop over orders
            for j in range(header['NORDERS']):
                ordnum = f'{j + 1:02d}'

                # check for excluded order
                appos_key = f'APPOSO{ordnum}'
                if appos_key not in header:
                    continue

                log.info('')
                log.info(f'Order {ordnum}:')

                # retrieve data from input
                space = hdul[f'SLITPOS_ORDER_{ordnum}'].data
                wave = hdul[f'WAVEPOS_ORDER_{ordnum}'].data
                profile = hdul[f'SPATIAL_PROFILE_ORDER_{ordnum}'].data
                appos = parse_apertures(header[f'APPOSO{ordnum}'], 1)[0]
                apfwhm = parse_apertures(header[f'APFWHM{ordnum}'], 1)[0]

                if full_slit:
                    # fix to full aperture, assume positive, no background
                    half_slit = (space.max() - space.min()) / 2
                    ap = {'position': half_slit,
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
                        apsign = parse_apertures(
                            header[f'APSGNO{ordnum}'], 1)[0]

                    aplist = []
                    for k, pos in enumerate(appos):
                        ap = {'position': pos,
                              'fwhm': apfwhm[k]}
                        if len(apsign) > k:
                            ap['sign'] = apsign[k]
                        else:  # pragma: no cover
                            ap['sign'] = apsign[-1]

                        if fix_ap:
                            # start and end explicitly set:
                            # update position to center of range and set
                            # radius to half of range
                            if len(apstart_list[i]) > k:
                                ap_start = apstart_list[i][k]
                            else:  # pragma: no cover
                                ap_start = aprad_list[i][-1]
                            if len(apend_list[i]) > k:
                                ap_end = apend_list[i][k]
                            else:  # pragma: no cover
                                ap_end = apend_list[i][-1]

                            half_aprad = (ap_end - ap_start) / 2
                            ap['position'] = half_aprad + ap_start
                            ap['psf_radius'] = half_aprad
                            ap['aperture_radius'] = half_aprad
                        else:
                            if fix_aprad:
                                if len(aprad_list[i]) > k:
                                    ap['aperture_radius'] = aprad_list[i][k]
                                else:  # pragma: no cover
                                    ap['aperture_radius'] = aprad_list[i][-1]
                            if fix_psfrad:
                                if len(psfrad_list[i]) > k:
                                    ap['psf_radius'] = psfrad_list[i][k]
                                else:  # pragma: no cover
                                    ap['psf_radius'] = psfrad_list[i][-1]
                        aplist.append(ap)

                    apertures = {j: aplist}
                    profiles = {j: np.vstack([space, profile])}

                    if fix_bgr:
                        aperture_regions = get_apertures(profiles, apertures,
                                                         get_bg=False,
                                                         refit_fwhm=refit)[j]
                        if len(bgr_list) > i:
                            aperture_regions['background'] = {
                                'regions': bgr_list[i]}
                        else:
                            aperture_regions['background'] = {'regions': []}
                    else:
                        aperture_regions = get_apertures(profiles, apertures,
                                                         refit_fwhm=refit)[j]

                # log aperture values
                apsign, aprad, appsfrad, header_pos = [], [], [], []
                for k, ap in enumerate(aperture_regions['apertures']):
                    pos = ap['position']
                    sign = '{:d}'.format(int(ap['sign']))
                    rad = '{:.3f}'.format(ap['aperture_radius'])
                    psfrad = '{:.3f}'.format(ap['psf_radius'])

                    apsign.append(sign)
                    aprad.append(rad)
                    appsfrad.append(psfrad)
                    header_pos.append(f'{pos:.3f}')

                    log.info('  Aperture {}:'.format(k))
                    log.info('           position: {:.3f}'.format(pos))
                    log.info('               sign: {}'.format(sign))
                    log.info('         PSF radius: {}'.format(psfrad))
                    log.info('    aperture radius: {}'.format(rad))
                    log.info('')

                    # also add the trace into the aperture, for use
                    # in making an aperture mask
                    # For EXES, this is always fixed to the aperture position
                    ap['trace'] = np.array([pos] * len(wave))

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
                hdinsert(header, f'APPOSO{ordnum}', ','.join(header_pos),
                         comment='Aperture positions [arcsec]')
                hdinsert(header, f'APSGNO{ordnum}', ','.join(apsign),
                         comment='Aperture signs')
                hdinsert(header, f'APRADO{ordnum}', ','.join(aprad),
                         comment='Aperture radii [arcsec]')
                hdinsert(header, f'PSFRAD{ordnum}', ','.join(appsfrad),
                         comment='Aperture PSF radii [arcsec]')
                hdinsert(header, f'BGR_O{ordnum}', ','.join(bgr),
                         comment='Aperture background regions [arcsec]')

                # also add to order flux extension header,
                # labeled as order 1
                exthead = hdul[f'FLUX_ORDER_{ordnum}'].header
                hdinsert(exthead, 'APPOSO01', ','.join(header_pos),
                         comment='Aperture positions [arcsec]')
                hdinsert(exthead, 'APSGNO01', ','.join(apsign),
                         comment='Aperture signs')
                hdinsert(exthead, 'APRADO01', ','.join(aprad),
                         comment='Aperture radii [arcsec]')
                hdinsert(exthead, 'PSFRAD01', ','.join(appsfrad),
                         comment='Aperture PSF radii [arcsec]')
                hdinsert(exthead, 'BGR_O01', ','.join(bgr),
                         comment='Aperture background regions [arcsec]')

                # make aperture mask and append to hdul
                if full_slit:
                    # use the whole slit
                    apmask = np.full((len(space), len(wave)), -1.0)
                    # but down weight the first and last row
                    apmask[0, :] = -0.5
                    apmask[-1, :] = -0.5
                else:
                    apmask = mkapmask(
                        space, wave, aperture_regions['apertures'],
                        aperture_regions['background']['regions'])
                exthead = hdul[f'BADMASK_ORDER_{ordnum}'].header.copy()

                # insert aperture mask after spatial profile for order
                idx = hdul.index_of(f'SPATIAL_PROFILE_ORDER_{ordnum}')
                hdul.insert(idx + 1,
                            fits.ImageHDU(
                                data=np.array(apmask),
                                header=exthead,
                                name=f'APERTURE_MASK_ORDER_{ordnum}'))

            # save if desired
            # update output name
            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])
            if param.get_value('save'):
                self.write_output(hdul, outname)
            results.append(hdul)

        self.input = results
        self.set_display_data()

    def subtract_background(self):
        """
        Subtract background along columns.

        Calls `sofia_redux.spectroscopy.extspec.col_subbg`.
        """
        from sofia_redux.spectroscopy.extspec import col_subbg

        # get parameters
        param = self.get_parameter_set()
        skip_bg = param.get_value('skip_bg')
        bg_fit_order = param.get_value('bg_fit_order')
        threshold = param.get_value('threshold')

        if skip_bg:
            log.info('No background subtraction performed.')
            return

        results = []
        for i, hdul in enumerate(self.input):
            header = hdul[0].header

            log.info('')
            log.info(header['FILENAME'])

            # loop over orders
            for j in range(header['NORDERS']):
                ordnum = f'{j + 1:02d}'
                # check for excluded order
                appos_key = f'APPOSO{ordnum}'
                if appos_key not in header:
                    continue

                log.info('')
                log.info(f'Order {ordnum}')

                # retrieve data from input
                image = hdul[f'FLUX_ORDER_{ordnum}'].data
                err = hdul[f'ERROR_ORDER_{ordnum}'].data
                mask = (hdul[f'BADMASK_ORDER_{ordnum}'].data < 1)
                space = hdul[f'SLITPOS_ORDER_{ordnum}'].data

                apmask = hdul[f'APERTURE_MASK_ORDER_{ordnum}'].data

                has_bg = np.any(np.isnan(apmask))
                if not has_bg:
                    log.info('No background regions defined.')
                else:
                    # correct each column for background identified in apmask
                    # image and error are corrected in place
                    nwave = image.shape[1]
                    for wavei in range(nwave):
                        with set_log_level('CRITICAL'):
                            bg_subtracted_col = col_subbg(
                                space, image[:, wavei], err[:, wavei]**2,
                                apmask[:, wavei], mask[:, wavei],
                                bg_fit_order, robust=threshold)
                        if bg_subtracted_col is not None:
                            # result is flux, variance, coefficients
                            image[:, wavei] = bg_subtracted_col[0]
                            err[:, wavei] = np.sqrt(bg_subtracted_col[1])

            # update output name
            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])

            # save if desired
            log.info('')
            if param.get_value('save'):
                self.write_output(hdul, outname)
            results.append(hdul)

        log.info('')
        self.input = results
        self.set_display_data()

    @staticmethod
    def _make_1d(hdul):
        """
        Make 1d spectrum file.

        Parameters
        ----------
        hdul : astropy.io.fits.HDUList
            File containing 2D and 1D spectral extensions.

        Returns
        -------
        spectrum : astropy.io.fits.HDUList
            New HDU list in Spextool style, containing 1D spectra.
        """
        header = hdul[0].header
        spechdr = header.copy()

        # add some spextool-required header keywords
        bunit = hdul[0].header.get('BUNIT', 'UNKNOWN').replace('/pixel', '')
        hdinsert(spechdr, 'XUNITS', 'cm-1', 'Spectral wavelength units')
        hdinsert(spechdr, 'YUNITS', bunit, 'Spectral flux units')

        try:
            del spechdr['BUNIT']
        except KeyError:  # pragma: no cover
            pass

        # loop over orders
        specset = []
        first = True
        max_length = 0
        orders = []
        for j in range(header['NORDERS']):
            ordnum = f'{j + 1:02d}'

            # check for excluded order
            appos_key = f'APPOSO{ordnum}'
            if appos_key not in header:
                continue

            # keep actually used orders
            orders.append(j + 1)

            suffix = f'_ORDER_{ordnum}'
            if (header['NORDERS'] == 1
                    and f'SPECTRAL_FLUX{suffix}' not in hdul):
                suffix = ''

            if hdul[f'SPECTRAL_FLUX{suffix}'].data.ndim > 1:
                naps = hdul[f'SPECTRAL_FLUX{suffix}'].data.shape[0]
            else:
                naps = 1
            if first:
                hdinsert(spechdr, 'NAPS', naps, 'Number of apertures')
                first = False

            wave = hdul[f'WAVEPOS{suffix}'].data
            disp = np.nanmean(wave[1:] - wave[:-1])
            hdinsert(spechdr, f'DISPO{ordnum}', disp,
                     'Dispersion [cm-1 pixel-1]')
            if len(wave) > max_length:
                max_length = len(wave)

            # transmission rows include additional data for various
            # molecular species -- use the first only,
            # which contains total transmission.
            trans = hdul[f'TRANSMISSION{suffix}'].data
            if trans.ndim > 1:
                trans = trans[0]

            for n in range(naps):
                if naps > 1:
                    speclist = [hdul[f'WAVEPOS{suffix}'].data,
                                hdul[f'SPECTRAL_FLUX{suffix}'].data[n],
                                hdul[f'SPECTRAL_ERROR{suffix}'].data[n],
                                trans,
                                hdul[f'RESPONSE{suffix}'].data[n]]
                else:
                    speclist = [hdul[f'WAVEPOS{suffix}'].data,
                                hdul[f'SPECTRAL_FLUX{suffix}'].data,
                                hdul[f'SPECTRAL_ERROR{suffix}'].data,
                                trans,
                                hdul[f'RESPONSE{suffix}'].data]

                specdata = np.vstack(speclist)
                specset.append(specdata)

        # update ORDERS and NORDERS in header to actually used values
        spechdr['ORDERS'] = ','.join([str(n) for n in orders])
        spechdr['NORDERS'] = len(orders)

        nspec = len(specset)
        spec_array = np.full((nspec, 5, max_length), np.nan)
        for i, data in enumerate(specset):
            spec_array[i, :, :data.shape[1]] = data

        # remove any empty dimensions
        spec_array = np.squeeze(spec_array)

        spec = fits.HDUList(fits.PrimaryHDU(data=spec_array,
                                            header=spechdr))
        return spec

    @staticmethod
    def _get_atran(header, atranfile, atrandir):
        """
        Get transmission data with wavenumber units.

        Calls `sofia_redux.instruments.exes.get_atran`.

        Parameters
        ----------
        header : astropy.io.fits.Header
            Header containing altitude and ZA information, for matching
            to model files.
        atranfile : str or None
            If provided, is used to directly specify the model file to
            use, rather than determining a best match from the transmission
            directory.
        atrandir : str or None
            If provided, is used as the source library for model files.
            If not provided, the default directory used is
            sofia_redux/instruments/exes/data/transmission.

        Returns
        -------
        atran : numpy.ndarray
            Array with shape (m, n), where n is the number of data
            points in the input model file and m is the number of
            rows containing transmission data. The first row is
            wavenumber; the second is total fractional atmospheric
            transmission at that wavenumber. Third and subsequent rows
            are transmission for other specific molecular species.
        """
        from sofia_redux.instruments.exes.get_atran import get_atran

        resolution = header['RP']
        atran = get_atran(header, resolution, filename=atranfile,
                          atran_dir=atrandir)
        if atran is None:
            if atrandir is None:
                msg = 'No matching transmission files.'
                log.warning(msg)
                raise ValueError(msg)
            else:
                log.info('')
                log.info(f'Model file not found in {atrandir}; '
                         f'trying default')
                # if not found, try the default directory
                atran = get_atran(header, resolution, filename=atranfile,
                                  atran_dir=None)
                if atran is None:
                    msg = 'No matching transmission files.'
                    log.warning(msg)
                    raise ValueError(msg)

        return atran

    def extract_spectra(self):
        """
        Extract 1D spectra from apertures.

        Calls `sofia_redux.spectroscopy.extspec`.
        """
        from sofia_redux.spectroscopy.extspec import extspec

        # get parameters
        param = self.get_parameter_set()
        use_profile = param.get_value('use_profile')
        fix_bad = param.get_value('fix_bad')
        threshold = param.get_value('threshold')
        optimal = 'optimal' in str(param.get_value('method')).lower()

        # get atran parameters
        atranfile = param.get_value('atranfile')
        atrandir = param.get_value('atrandir')
        if str(atranfile).strip() == '':
            atranfile = None
        if str(atrandir).strip() == '':
            atrandir = None
        else:
            # expand environment variables in path
            atrandir = os.path.expandvars(atrandir)
            if not os.path.exists(atrandir):
                atrandir = None

        results = []
        display_files = []
        for i, hdul in enumerate(self.input):
            log.info('')
            header = hdul[0].header
            log.info(header['FILENAME'])

            # get transmission data
            try:
                atran = self._get_atran(header, atranfile, atrandir)
            except ValueError:  # pragma: no cover
                atran = None

            # trim to wavecal range
            wavecal = hdul['WAVECAL'].data
            min_wave = np.nanmin(wavecal)
            max_wave = np.nanmax(wavecal)
            buffer = 0.5 * (max_wave - min_wave)
            if atran is not None:
                in_range = ((atran[0] > min_wave - buffer)
                            & (atran[0] < max_wave + buffer))
                atran = atran[:, in_range]

                # check for missing data
                if atran.size < 10:
                    log.warning('Transmission model range does not '
                                'match data.')
                    atran = None

            # attach empty spectrum if no appropriate data found
            if atran is None:
                atran = np.full((2, 1024), np.nan)
                atran[0] = np.linspace(min_wave - buffer,
                                       max_wave - buffer, 1024)

            # attach the full transmission spectrum to the file,
            # after the ORDER_MASK extension
            idx = hdul.index_of('ORDER_MASK')
            exthead = fits.Header()
            hdinsert(exthead, 'BUNIT', '', 'Data units')
            hdinsert(exthead, 'ATRNFILE', header.get('ATRNFILE', 'UNKNOWN'),
                     'Transmission file')
            hdinsert(exthead, 'XUNITS', 'cm-1', 'Wavecal units')
            hdinsert(exthead, 'YUNITS', '', 'Transmission units')
            hdul.insert(idx + 1,
                        fits.ImageHDU(atran, exthead,
                                      name='TRANSMISSION'))

            # loop over orders
            for j in range(header['NORDERS']):
                ordnum = f'{j + 1:02d}'

                # check for excluded order
                appos_key = f'APPOSO{ordnum}'
                if appos_key not in header:
                    continue

                log.info('')
                log.info(f'Order {ordnum}')

                # retrieve data from input
                image = hdul[f'FLUX_ORDER_{ordnum}'].data
                var = hdul[f'ERROR_ORDER_{ordnum}'].data ** 2
                mask = (hdul[f'BADMASK_ORDER_{ordnum}'].data < 1)
                wave = hdul[f'WAVEPOS_ORDER_{ordnum}'].data
                space = hdul[f'SLITPOS_ORDER_{ordnum}'].data
                profile = hdul[f'SPATIAL_PROFILE_ORDER_{ordnum}'].data
                spatmap = hdul[f'SPATIAL_MAP_ORDER_{ordnum}'].data
                apmask = hdul[f'APERTURE_MASK_ORDER_{ordnum}'].data

                apsign = parse_apertures(header[f'APSGNO{ordnum}'], 1)[0]
                rectimg = {j + 1: {'image': image, 'variance': var,
                                   'mask': mask, 'wave': wave,
                                   'spatial': space, 'header': header,
                                   'apmask': apmask, 'apsign': apsign}}

                if use_profile:
                    spatmap = None
                    profile = {j + 1: profile}
                else:
                    spatmap = {j + 1: spatmap}
                    profile = None

                spectra = extspec(rectimg,
                                  profile=profile, spatial_map=spatmap,
                                  optimal=optimal, fix_bad=fix_bad,
                                  sub_background=False,
                                  threshold=threshold)[j + 1]

                # update flux, error, and mask planes -- they may have had
                # bad pixels corrected in the extraction process
                hdul[f'FLUX_ORDER_{ordnum}'].data = rectimg[j + 1]['image']
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    hdul[f'ERROR_ORDER_{ordnum}'].data = np.sqrt(
                        rectimg[j + 1]['variance'])
                boolmask = rectimg[j + 1]['mask']
                mask = np.zeros(boolmask.shape, dtype=int)
                mask[~boolmask] = 1
                hdul[f'BADMASK_ORDER_{ordnum}'].data = mask

                # attach spectral flux and error to output file:
                # shape is n_ap x n_wave
                exthead = fits.Header()

                # insert after aperture mask
                idx = hdul.index_of(f'APERTURE_MASK_ORDER_{ordnum}')
                hdul.insert(idx + 1,
                            fits.ImageHDU(
                                spectra[:, 1, :], exthead,
                                name=f'SPECTRAL_FLUX_ORDER_{ordnum}'))
                hdul.insert(idx + 2,
                            fits.ImageHDU(
                                spectra[:, 2, :], exthead,
                                name=f'SPECTRAL_ERROR_ORDER_{ordnum}'))
                bunit = header.get('BUNIT', 'UNKNOWN').replace('/pixel', '')
                hdinsert(hdul[f'SPECTRAL_FLUX_ORDER_{ordnum}'].header,
                         'BUNIT', bunit, 'Data units')
                hdinsert(hdul[f'SPECTRAL_ERROR_ORDER_{ordnum}'].header,
                         'BUNIT', bunit, 'Data units')

                # attach an approximate transmission
                tdata = []
                for trans in atran[1:]:
                    outtrans = np.interp(hdul[f'WAVEPOS_ORDER_{ordnum}'].data,
                                         atran[0], trans,
                                         left=np.nan, right=np.nan)

                    tdata.append(outtrans)

                exthead = hdul[f'SPECTRAL_FLUX_ORDER_{ordnum}'].header.copy()
                hdul.insert(idx + 3,
                            fits.ImageHDU(np.array(tdata), exthead,
                                          name=f'TRANSMISSION_ORDER_{ordnum}'))
                hdinsert(hdul[f'TRANSMISSION_ORDER_{ordnum}'].header, 'BUNIT',
                         '', 'Data units')

                # extract and attach a reference flat spectrum

                # use the same aperture mask, but always use standard
                # extraction and do not fix bad pixels
                flat = hdul[f'FLAT_ORDER_{ordnum}'].data
                flat[flat == 0] = np.nan
                flatvar = np.full(flat.shape, np.nan)
                rectflat = {j + 1: {'image': flat, 'variance': flatvar,
                                    'wave': wave, 'spatial': space,
                                    'apmask': apmask}}
                flat_spec = extspec(rectflat, optimal=False, fix_bad=False,
                                    sub_background=False)[j + 1]

                # invert to get response (raw units / calibrated)
                resp_spec = 1 / flat_spec[:, 1, :]
                resp_spec[~np.isfinite(resp_spec)] = np.nan
                hdul.insert(idx + 4,
                            fits.ImageHDU(
                                resp_spec, exthead,
                                name=f'RESPONSE_ORDER_{ordnum}'))
                if bunit == 'Jy':
                    r_bunit = 'ct/Jy'
                else:
                    r_bunit = 'ct erg-1 s cm2 sr cm-1'
                hdinsert(hdul[f'RESPONSE_ORDER_{ordnum}'].header,
                         'BUNIT', r_bunit, 'Data units')

            # update output name
            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                log.info('')
                self.write_output(hdul, outname)

            if param.get_value('save_1d'):
                log.info('')
                log.info('Saving 1D spectra:')
                spec = self._make_1d(hdul)
                if self.sky_spectrum:
                    specname = outname.replace('SSM', 'SSP')
                    spec[0].header['PRODTYPE'] = 'sky_spectra_1d'
                else:
                    specname = outname.replace('SPM', 'SPC')
                    spec[0].header['PRODTYPE'] = 'spectra_1d'
                spec[0].header['FILENAME'] = os.path.basename(specname)
                self.write_output(spec, specname)
                display_files.append(spec)

            results.append(hdul)

        log.info('')
        self.input = results
        self.set_display_data(filenames=display_files + self.input)

    def combine_spectra(self):
        """
        Combine spectra.

        Calls `sofia_redux.toolkit.image.combine.combine_images`
        for coaddition. The combination method may be configured in
        parameters.
        """
        # get parameters
        param = self.get_parameter_set()
        method = param.get_value('method')
        weighted = param.get_value('weighted')
        robust = param.get_value('robust')
        sigma = param.get_value('threshold')
        maxiters = param.get_value('maxiters')
        combine_aps = param.get_value('combine_aps')

        hdr_list = list()
        data_list = list()
        var_list = list()
        exp_list = list()
        spectral_sets = dict()
        test_wave = dict()
        display_files = list()
        for i, hdul in enumerate(self.input):
            header = hdul[0].header
            hdr_list.append(header)
            data_list.append(hdul['FLUX'].data)
            var_list.append(hdul['ERROR'].data ** 2)
            exp_list.append(np.full_like(hdul['FLUX'].data,
                                         hdul[0].header.get('EXPTIME', 0.0)))

            for j in range(header['NORDERS']):
                ordnum = f'{j + 1:02d}'

                # check for excluded order
                appos_key = f'APPOSO{ordnum}'
                if appos_key not in header:
                    continue

                # check for same wavelength solution
                wavepos = hdul[f'WAVEPOS_ORDER_{ordnum}'].data

                if i == 0:
                    test_wave[ordnum] = wavepos

                    # initialize lists for order
                    spectral_sets[ordnum] = {'flux': list(),
                                             'var': list(),
                                             'specflux': list(),
                                             'specvar': list(),
                                             'response': list()}
                else:
                    if (ordnum not in test_wave
                            or not np.allclose(wavepos, test_wave[ordnum])):
                        msg = 'Mismatched wavenumbers or orders. ' \
                              'Spectra cannot be combined.'
                        log.error(msg)
                        raise ValueError(msg)

                # append images to lists
                spectral_sets[ordnum]['flux'].append(
                    hdul[f'FLUX_ORDER_{ordnum}'].data)
                spectral_sets[ordnum]['var'].append(
                    hdul[f'ERROR_ORDER_{ordnum}'].data ** 2)

                # gather spectra, accounting for possible multiple apertures
                n_ap = 1
                specflux = hdul[f'SPECTRAL_FLUX_ORDER_{ordnum}'].data
                specvar = hdul[f'SPECTRAL_ERROR_ORDER_{ordnum}'].data ** 2
                specresp = hdul[f'RESPONSE_ORDER_{ordnum}'].data

                if specflux.ndim > 1:
                    n_ap = specflux.shape[0]
                    if n_ap == 1:
                        specflux = specflux[0]
                        specvar = specvar[0]
                        specresp = specresp[0]
                if combine_aps and n_ap > 1:
                    for j in range(n_ap):
                        spectral_sets[ordnum]['specflux'].append(specflux[j])
                        spectral_sets[ordnum]['specvar'].append(specvar[j])
                        spectral_sets[ordnum]['response'].append(specresp[j])
                else:
                    spectral_sets[ordnum]['specflux'].append(specflux)
                    spectral_sets[ordnum]['specvar'].append(specvar)
                    spectral_sets[ordnum]['response'].append(specresp)

        # combine all the full data arrays
        if len(data_list) > 1:
            outdata, outvar = combine_images(
                data_list, variance=var_list,
                method=method, weighted=weighted,
                robust=robust, sigma=sigma, maxiters=maxiters)
        else:
            outdata, outvar = data_list[0], var_list[0]

        # merge all the headers
        outhdr = mergehdr(hdr_list)

        # make a header for the error image
        errhead = fits.Header()

        # store output data: final extensions are
        #   FLUX, ERROR, TRANSMISSION
        # and by order:
        #   WAVEPOS, FLUX, ERROR,
        #   SPECTRAL_FLUX, SPECTRAL_ERROR, TRANSMISSION, RESPONSE
        template = self.input[0]
        primary = fits.PrimaryHDU(data=outdata, header=outhdr)
        hdul = fits.HDUList(primary)
        hdinsert(errhead, 'BUNIT', outhdr.get('BUNIT', 'UNKNOWN'),
                 'Data units')
        hdul.append(fits.ImageHDU(data=np.sqrt(outvar),
                                  header=errhead,
                                  name='ERROR'))
        hdul.append(template['TRANSMISSION'])

        for ordnum in sorted(spectral_sets.keys()):
            # combine all the 2D arrays
            if len(spectral_sets[ordnum]['flux']) > 1:
                outdata, outvar = combine_images(
                    spectral_sets[ordnum]['flux'],
                    variance=spectral_sets[ordnum]['var'],
                    method=method, weighted=weighted,
                    robust=robust, sigma=sigma, maxiters=maxiters)
            else:
                outdata = spectral_sets[ordnum]['flux'][0]
                outvar = spectral_sets[ordnum]['var'][0]

            hdul.append(fits.ImageHDU(
                data=outdata,
                header=template[f'FLUX_ORDER_{ordnum}'].header,
                name=f'FLUX_ORDER_{ordnum}'))
            hdul.append(fits.ImageHDU(
                data=np.sqrt(outvar),
                header=template[f'ERROR_ORDER_{ordnum}'].header,
                name=f'ERROR_ORDER_{ordnum}'))

            # combine all the 1D spectra
            if len(spectral_sets[ordnum]['specflux']) > 1:
                outspec, outspecvar = combine_images(
                    spectral_sets[ordnum]['specflux'],
                    variance=spectral_sets[ordnum]['specvar'],
                    method=method, weighted=weighted, robust=robust,
                    sigma=sigma, maxiters=maxiters)
                outresp, _ = combine_images(spectral_sets[ordnum]['response'],
                                            method=method, robust=False)
            else:
                outspec = spectral_sets[ordnum]['specflux'][0]
                outspecvar = spectral_sets[ordnum]['specvar'][0]
                outresp = spectral_sets[ordnum]['response'][0]

            # wavepos, spectral flux and error, transmission, response
            # Note: wavepos and transmission are just copied
            # from the first file; the flux, error, and response
            # are from combined data
            hdul.append(fits.ImageHDU(
                data=template[f'WAVEPOS_ORDER_{ordnum}'].data,
                header=template[f'WAVEPOS_ORDER_{ordnum}'].header,
                name=f'WAVEPOS_ORDER_{ordnum}'))
            hdul.append(fits.ImageHDU(
                data=outspec,
                header=template[f'SPECTRAL_FLUX_ORDER_{ordnum}'].header,
                name=f'SPECTRAL_FLUX_ORDER_{ordnum}'))
            hdul.append(fits.ImageHDU(
                data=np.sqrt(outspecvar),
                header=template[f'SPECTRAL_ERROR_ORDER_{ordnum}'].header,
                name=f'SPECTRAL_ERROR_ORDER_{ordnum}'))
            hdul.append(fits.ImageHDU(
                data=template[f'TRANSMISSION_ORDER_{ordnum}'].data,
                header=template[f'TRANSMISSION_ORDER_{ordnum}'].header,
                name=f'TRANSMISSION_ORDER_{ordnum}'))
            hdul.append(fits.ImageHDU(
                data=outresp,
                header=template[f'RESPONSE_ORDER_{ordnum}'].header,
                name=f'RESPONSE_ORDER_{ordnum}'))

        outname = self.update_output(hdul, self.filenum,
                                     self.prodtypes[self.step_index])
        self.filenum = [self.get_filenum(outname)]

        # save if desired
        if param.get_value('save'):
            log.info('')
            # save 1D Spextool-style final products
            log.info('1D spectra:')
            spec = self._make_1d(hdul)
            if self.sky_spectrum:
                specname = outname.replace('_SCM_', '_SCS_')
                spec[0].header['PRODTYPE'] = 'sky_combined_spectrum_1d'
            else:
                specname = outname.replace('_COM_', '_CMB_')
                spec[0].header['PRODTYPE'] = 'combined_spectrum_1d'
            spec[0].header['FILENAME'] = os.path.basename(specname)
            self.write_output(spec, specname)
            display_files.append(spec)

            # also save full product
            log.info('')
            log.info('Full product (2D images and 1D spectra):')
            self.write_output(hdul, outname)

        self.input = [hdul]
        self.set_display_data(filenames=display_files + self.input)

    def refine_wavecal(self):
        """
        Refine wavecal by setting a new central wavenumber.

        User input should identify the order and pixel position of
        a known spectral feature as well as the calibrated wavenumber
        for that feature. This information is used to derive a new
        central wavenumber.

        Calls `sofia_redux.instruments.exes.wavecal` to update the
        wavelength calibration from the central wavenumber.
        """
        from sofia_redux.instruments.exes.wavecal import wavecal

        # get parameters
        param = self.get_parameter_set()
        interactive = param.get_value('interactive')
        identify_order = param.get_value('identify_order')
        identify_line = param.get_value('identify_line')
        identify_waveno = param.get_value('identify_waveno')

        # todo - add interactive line selection with Eye plot
        if interactive:  # pragma: no cover
            log.warning('Interactive wavecal not implemented')

        # check inputs
        test = [str(v).strip()
                for v in [identify_order, identify_line, identify_waveno]]
        if '' in test:
            log.info('No line identified, not refining wavecal.')
            return

        try:
            i_order = int(identify_order)
        except (TypeError, ValueError):
            i_order = -1

        # derive central wavenumber from identified line
        display_files = list()
        for i, hdul in enumerate(self.input):
            header = hdul[0].header

            # here, NORDERS must refer to total number of orders for
            # the data, not the number extracted
            norders = header['NORDERS']
            if i_order <= 0 or i_order > norders:
                msg = f'Invalid order number {identify_order}'
                log.error(msg)
                raise ValueError(msg)

            ordnum = f'{i_order:02d}'
            try:
                wavepos = hdul[f'WAVEPOS_ORDER_{ordnum}'].data
            except (IndexError, KeyError):
                msg = f'Invalid order number {identify_order}'
                log.error(msg)
                raise ValueError(msg)

            # old wavenumber value at pixel position
            old_wave = np.interp(identify_line, np.arange(wavepos.size),
                                 wavepos)

            # calculate new central wavenumber
            central_waveno = parse_central_wavenumber(header)
            hrr = header['HRR']
            hrdgr = header['HRDGR']
            instcfg = header['INSTCFG']
            cross_dispersed = ((instcfg == 'HIGH_MED')
                               or (instcfg == 'HIGH_LOW'))
            if cross_dispersed:
                dw = 0.5 / (np.sqrt(hrr ** 2 / (1 + hrr ** 2)) * hrdgr)
            else:
                dw = 1.0
            xorder = norders - i_order + 1

            wnoi = central_waveno + dw * (xorder - (norders + 1) / 2.0)
            wno0 = central_waveno + (identify_waveno
                                     - old_wave) * wnoi / old_wave

            log.info(f'Old central wavenumber: {central_waveno}')
            log.info(f'New central wavenumber: {wno0}')

            # update wavecal in existing orders
            header['WNO0'] = wno0
            for j in range(norders):
                ordnum = f'{j + 1:02d}'
                appos_key = f'APPOSO{ordnum}'
                if appos_key not in header:
                    continue

                wave_header = header.copy()
                new_wave = wavecal(wave_header, order=j + 1)
                hdul[f'WAVEPOS_ORDER_{ordnum}'].data = new_wave

                # update WCS in flux, error extensions
                for ext in [f'FLUX_ORDER_{ordnum}', f'ERROR_ORDER_{ordnum}']:
                    mid_wave = new_wave[new_wave.size // 2]
                    wave_scale = np.mean(new_wave[1:] - new_wave[:-1])
                    hdul[ext].header['CRVAL1'] = mid_wave
                    hdul[ext].header['CDELT1'] = wave_scale

                # update atran extension - old one no longer matches data
                atran = hdul['TRANSMISSION'].data
                tdata = []
                for trans in atran[1:]:
                    outtrans = np.interp(hdul[f'WAVEPOS_ORDER_{ordnum}'].data,
                                         atran[0], trans,
                                         left=np.nan, right=np.nan)

                    tdata.append(outtrans)
                hdul[f'TRANSMISSION_ORDER_{ordnum}'].data = np.array(tdata)

            # update output name
            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                log.info('')

                # save 1D Spextool-style products
                log.info('1D spectra:')
                spec = self._make_1d(hdul)
                if self.sky_spectrum:  # pragma: no cover
                    specname = outname.replace('_WRF_', '_SWS_')
                    spec[0].header['PRODTYPE'] = 'sky_wavecal_refined_1d'
                else:
                    specname = outname.replace('_WRF_', '_WRS_')
                    spec[0].header['PRODTYPE'] = 'wavecal_refined_1d'
                spec[0].header['FILENAME'] = os.path.basename(specname)
                self.write_output(spec, specname)
                display_files.append(spec)

                # also save full product
                log.info('')
                log.info('Full product (2D images and 1D spectra):')
                self.write_output(hdul, outname)

            # set override in load data parameters:
            # if reduction is reset, it will be run with the new calibration
            load_param = self.parameters.current[0]
            load_param.set_value('cent_wave', wno0)

        self.set_display_data(filenames=display_files + self.input)

    @staticmethod
    def _parse_regions(region_string):
        """
        Parse wavenumber regions from an input string.

        Wavenumber regions should be comma-separated ranges specified
        as start-stop, prepended by the order number. Orders should be
        separated by semicolons. For example,
        "1:780-785,786-787;2:801.5-802.3"
        specifies two regions in order 1 and 1 region in order 2.

        Parameters
        ----------
        region_string : str
            Input parameter string.

        Returns
        -------
        dict
            Keys are integer order numbers, values are lists of floating point
            region (start, stop) tuples.
        """
        bad_msg = ['Could not read wavenumber region '
                   f"parameter: '{region_string}'",
                   'Wavenumber regions should be comma-separated '
                   'ranges as start-stop, prepended by the order number. ',
                   'Orders should be separated by semi-colons. ',
                   'For example, "1:780-785,786-787;2:801.5-802.3" '
                   'specifies two regions in order 1 and 1 region in order 2.']

        regions = dict()
        order_pos = list(str(region_string).split(';'))
        for op in order_pos:
            order_set = list(op.split(':'))
            try:
                order, rg_set = order_set
                order = int(order)
            except (ValueError, TypeError):
                for msg in bad_msg:
                    log.error(msg)
                raise ValueError('Invalid wavenumber '
                                 'region parameter.') from None

            rg_set = list(rg_set.split(','))
            rg_list = list()
            for reg in rg_set:
                rg_range = reg.split('-')
                try:
                    start, stop = rg_range
                    rg_list.append((float(start), float(stop)))
                except (ValueError, TypeError):
                    for msg in bad_msg:
                        log.error(msg)
                    raise ValueError('Invalid wavenumber '
                                     'region parameter.') from None
            regions[order] = rg_list
        return regions

    @staticmethod
    def _trim_data(hdul, ordnum, nan_regions):
        """
        Set specified data to NaN.

        2D images and 1D spectra for the specified order are set
        to NaN between the wavenumber regions specifed.

        Parameters
        ----------
        hdul : astropy.io.fits.HDUList
            HDU list containing data to trim.
        ordnum : str
            Two character order number string, zero-padded.
        nan_regions : list of list of float
            List of regions to blank.  Each element should be a
            two-element list or tuple containing a start and end
            value for a region, in wavenumber.
        """
        # regions are specified as wavenumber ranges:
        # use wavepos to get areas to blank in image and spectral_flux,
        # in place
        wavepos = hdul[f'WAVEPOS_ORDER_{ordnum}'].data
        image = hdul[f'FLUX_ORDER_{ordnum}'].data
        error = hdul[f'ERROR_ORDER_{ordnum}'].data
        spec = hdul[f'SPECTRAL_FLUX_ORDER_{ordnum}'].data
        spec_err = hdul[f'SPECTRAL_ERROR_ORDER_{ordnum}'].data
        for region in nan_regions:
            idx = (wavepos >= region[0]) & (wavepos <= region[1])
            if np.any(idx):
                image[:, idx] = np.nan
                error[:, idx] = np.nan
                spec[idx] = np.nan
                spec_err[idx] = np.nan

    def merge_orders(self):
        """
        Merge all spectral orders.

        Calls `sofia_redux.spectroscopy.mergespec` to merge the
        1D spectra and `sofia_redux.toolkit.image.coadd` to combine
        the 2D spectral images.
        """
        from sofia_redux.spectroscopy.mergespec import mergespec
        from sofia_redux.toolkit.image.coadd import coadd

        # get parameters
        param = self.get_parameter_set()
        threshold = param.get_value('threshold')
        statistic = param.get_value('statistic')
        noise_test = param.get_value('noise_test')
        local_noise = param.get_value('local_noise')
        local_radius = param.get_value('local_radius')
        trim = param.get_value('trim')
        trim_regions = param.get_value('trim_regions')

        # todo: allow interactive trimming
        if trim:  # pragma: no cover
            log.warning('Interactive trimming not yet implemented.')
        if str(trim_regions).strip().lower() != '':
            nan_regions = self._parse_regions(trim_regions)
        else:
            nan_regions = None

        results = []
        for i, hdul in enumerate(self.input):
            log.info('')
            header = hdul[0].header
            log.info(header['FILENAME'])

            # get the atran data from the file
            atran = hdul['TRANSMISSION'].data

            # loop over orders, merging one at a time
            result = None
            for j in range(header['NORDERS']):
                ordnum = f'{j + 1:02d}'
                appos_key = f'APPOSO{ordnum}'
                if appos_key not in header:
                    continue

                # trim specified regions if necessary
                if nan_regions is not None and j + 1 in nan_regions:
                    log.info(f'Trimming order {ordnum} regions: '
                             f'{nan_regions[j + 1]}')
                    self._trim_data(hdul, ordnum, nan_regions[j + 1])

                # skip if spectrum is all nan
                spec = hdul[f'SPECTRAL_FLUX_ORDER_{ordnum}'].data
                if np.all(np.isnan(spec)):
                    log.warning(f'No good data in order {ordnum}')
                    continue

                if result is None:
                    # initialize output from first order
                    image = hdul[f'FLUX_ORDER_{ordnum}'].data
                    primary = fits.PrimaryHDU(image, header)
                    primary.header['EXTNAME'] = 'FLUX'
                    primary.header['NORDERS'] = 1

                    flux_header = hdul[f'FLUX_ORDER_{ordnum}'].header
                    for key in self.wcs_keys:
                        hdinsert(primary.header, key,
                                 flux_header[key], flux_header.comments[key])

                    result = fits.HDUList(primary)

                    for extname in ['ERROR', 'WAVEPOS',
                                    'SPECTRAL_FLUX', 'SPECTRAL_ERROR',
                                    'RESPONSE']:
                        hdu = hdul[f'{extname}_ORDER_{ordnum}']
                        hdu.header['EXTNAME'] = extname
                        result.append(hdu)

                else:
                    # retrieve data from input
                    image = hdul[f'FLUX_ORDER_{ordnum}'].data
                    image_header = hdul[f'FLUX_ORDER_{ordnum}'].header
                    var = hdul[f'ERROR_ORDER_{ordnum}'].data ** 2
                    wave = hdul[f'WAVEPOS_ORDER_{ordnum}'].data
                    flux = hdul[f'SPECTRAL_FLUX_ORDER_{ordnum}'].data
                    err = hdul[f'SPECTRAL_ERROR_ORDER_{ordnum}'].data
                    response = hdul[f'RESPONSE_ORDER_{ordnum}'].data

                    # check for apertures
                    if result['SPECTRAL_FLUX'].data.ndim > 1:
                        all_wave = result['WAVEPOS'].data.copy()
                        all_flux = result['SPECTRAL_FLUX'].data.copy()
                        all_err = result['SPECTRAL_ERROR'].data.copy()
                        all_resp = result['RESPONSE'].data.copy()
                        n_ap = all_flux.shape[0]
                        for n in range(n_ap):
                            # merge current spectral order with new
                            merged = [all_wave, all_flux[n], all_err[n]]
                            incoming = [wave, flux[n], err[n]]
                            new_spec = mergespec(merged, incoming,
                                                 s2n_threshold=threshold,
                                                 s2n_statistic=statistic,
                                                 noise_test=noise_test,
                                                 local_noise=local_noise,
                                                 local_radius=local_radius)

                            # do the same for response but just do
                            # straight mean in overlaps
                            merged = [all_wave, all_resp[n]]
                            incoming = [wave, response[n]]
                            new_resp = mergespec(merged, incoming)

                            # make sure response matches flux shape
                            outresp = np.interp(new_spec[0],
                                                new_resp[0], new_resp[1],
                                                left=np.nan, right=np.nan)

                            # update output
                            if n == 0:
                                result['WAVEPOS'].data = new_spec[0]
                                result['SPECTRAL_FLUX'].data = new_spec[1]
                                result['SPECTRAL_ERROR'].data = new_spec[2]
                                result['RESPONSE'].data = outresp
                            else:
                                result['SPECTRAL_FLUX'].data = np.vstack(
                                    [result['SPECTRAL_FLUX'].data,
                                     new_spec[1]])
                                result['SPECTRAL_ERROR'].data = np.vstack(
                                    [result['SPECTRAL_ERROR'].data,
                                     new_spec[2]])
                                result['RESPONSE'].data = np.vstack(
                                    [result['RESPONSE'].data,
                                     outresp])
                    else:
                        # merge current spectral order with new
                        merged = [result['WAVEPOS'].data,
                                  result['SPECTRAL_FLUX'].data,
                                  result['SPECTRAL_ERROR'].data]
                        incoming = [wave, flux, err]
                        new_spec = mergespec(merged, incoming,
                                             s2n_threshold=threshold,
                                             s2n_statistic=statistic,
                                             noise_test=noise_test,
                                             local_noise=local_noise,
                                             local_radius=local_radius)

                        # do the same for response but just do
                        # straight mean in overlaps
                        merged = [result['WAVEPOS'].data,
                                  result['RESPONSE'].data]
                        incoming = [wave, response]
                        new_resp = mergespec(merged, incoming)

                        # make sure response matches flux shape
                        outresp = np.interp(new_spec[0],
                                            new_resp[0], new_resp[1],
                                            left=np.nan, right=np.nan)

                        # update output
                        result['WAVEPOS'].data = new_spec[0]
                        result['SPECTRAL_FLUX'].data = new_spec[1]
                        result['SPECTRAL_ERROR'].data = new_spec[2]
                        result['RESPONSE'].data = outresp

                    # coadd images in spectral WCS
                    with set_log_level('WARNING'):
                        outhdr, merged_image, merged_var, _ = \
                            coadd([result['FLUX'].header, image_header],
                                  [result['FLUX'].data, image],
                                  [result['ERROR'].data ** 2, var],
                                  [np.ones_like(result['FLUX'].data),
                                   np.ones_like(image)],
                                  rotate=False, spectral=True)
                    result['FLUX'].header = outhdr
                    result['FLUX'].data = merged_image
                    result['ERROR'].data = np.sqrt(merged_var)

            # attach an approximate transmission for the full wavelength range
            tdata = []
            for trans in atran[1:]:
                outtrans = np.interp(result['WAVEPOS'].data,
                                     atran[0], trans,
                                     left=np.nan, right=np.nan)

                tdata.append(outtrans)

            # insert transmission before response
            idx = result.index_of('RESPONSE')
            exthead = result['SPECTRAL_FLUX'].header.copy()
            result.insert(idx, fits.ImageHDU(np.array(tdata), exthead,
                                             name='TRANSMISSION'))
            hdinsert(result['TRANSMISSION'].header, 'BUNIT', '', 'Data units')

            # set PROCSTAT to level 3, for final product only
            hdinsert(result[0].header, 'PROCSTAT', 'LEVEL_3')

            # update output name
            outname = self.update_output(result, self.filenum[i],
                                         self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                log.info('')
                self.write_output(result, outname)

                log.info('')
                log.info('Saving 1D spectra:')
                spec = self._make_1d(result)
                if self.sky_spectrum:
                    specname = outname.replace('SMM', 'SMD')
                    spec[0].header['PRODTYPE'] = 'sky_orders_merged_1d'
                else:
                    specname = outname.replace('MRM', 'MRD')
                    spec[0].header['PRODTYPE'] = 'orders_merged_1d'
                spec[0].header['FILENAME'] = os.path.basename(specname)
                self.write_output(spec, specname)

            results.append(result)

        log.info('')
        self.input = results
        self.set_display_data()

    def specmap(self):
        """
        Generate a quick-look spectral plot.

        Calls `sofia_redux.visualization.quicklook.make_spectral_plot`.

        The output from this step is identical to the input, so is
        not saved.  As a side effect, a PNG file is saved to disk to the
        same base name as the input file, with a '.png' extension.
        """
        from matplotlib.backends.backend_agg \
            import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        from sofia_redux.visualization.quicklook import make_spectral_plot

        # get parameters
        param = self.get_parameter_set()
        normalize = param.get_value('normalize')
        scale = param.get_value('scale')
        ignore_outer = param.get_value('ignore_outer')
        atran_plot = param.get_value('atran_plot')
        error_plot = param.get_value('error_plot')
        colormap = param.get_value('colormap')
        overplot_color = param.get_value('overplot_color')
        watermark = param.get_value('watermark')

        if scale[0] <= 0 and scale[1] >= 100:
            scale = None

        for i, hdul in enumerate(self.input):
            header = hdul[0].header

            labels = None
            aplot = None
            nspec = 1
            try:
                # merged order full product
                wave = hdul['WAVEPOS'].data
                xunit = hdul['WAVEPOS'].header.get('BUNIT', 'um')

                spec_flux = hdul['SPECTRAL_FLUX'].data
                spec_err = hdul['SPECTRAL_ERROR'].data
                yunit = hdul['SPECTRAL_FLUX'].header.get('BUNIT',
                                                         'UNKNOWN')

                # check for multiple rows in atran -- use the first only
                atran = hdul['TRANSMISSION'].data
                if atran.ndim > 1:
                    atran = atran[0]

                # check for multiple apertures in new-style data
                if spec_flux.ndim > 1 and wave.ndim == 1:
                    nspec = spec_flux.shape[0]
                    wave = np.tile(wave, (nspec, 1))
                    labels = [f'Spectrum {j + 1}' for j in range(nspec)]

            except KeyError:
                # 1D spectrum
                xunit = hdul[0].header.get('XUNITS', 'UNKNOWN')
                yunit = hdul[0].header.get('YUNITS', 'UNKNOWN')

                # check for multiple orders/apertures
                if hdul[0].data.ndim > 2:
                    # multi-order or multi-ap
                    wave = hdul[0].data[:, 0]
                    nspec = hdul[0].data.shape[0]
                    labels = [f'Spectrum {j + 1}' for j in range(nspec)]
                    spec_flux = hdul[0].data[:, 1]
                    spec_err = hdul[0].data[:, 2]
                    try:
                        atran = hdul[0].data[:, 3]
                    except IndexError:  # pragma: no cover
                        # may be missing for old data
                        atran = np.full_like(spec_flux, np.nan)
                else:
                    # single order/ap
                    wave = hdul[0].data[0]
                    spec_flux = hdul[0].data[1]
                    spec_err = hdul[0].data[2]
                    try:
                        atran = hdul[0].data[3]
                    except IndexError:  # pragma: no cover
                        atran = np.full_like(spec_flux, np.nan)

            if nspec > 1:
                if normalize:
                    norm = np.nanmedian(spec_flux, axis=1)[:, None]
                    spec_flux = spec_flux / norm
                    spec_err = spec_err / norm
                    yunit = 'normalized'

                if ignore_outer > 0:
                    # set the outer N% of frames to NaN
                    wstart = int(ignore_outer * wave.shape[1])
                    wend = int((1 - ignore_outer) * wave.shape[1])
                    spec_flux[:, :wstart] = np.nan
                    spec_flux[:, wend:] = np.nan
                    atran[:wstart] = np.nan
                    atran[wend:] = np.nan
                    log.debug(f'Plotting between w={wstart} and w={wend}')
            else:

                if normalize:
                    norm = np.nanmedian(spec_flux)
                    spec_flux = spec_flux / norm
                    spec_err = spec_err / norm
                    yunit = 'normalized'
                if ignore_outer > 0:
                    wstart = int(ignore_outer * len(wave))
                    wend = int((1 - ignore_outer) * len(wave))
                    spec_flux[:wstart] = np.nan
                    spec_flux[wend:] = np.nan
                    atran[:wstart] = np.nan
                    atran[wend:] = np.nan
                    log.debug(f'Plotting between w={wstart} and w={wend}')

            # set text for title in plot
            obj = header.get('OBJECT', 'UNKNOWN')
            mode = header.get('INSTCFG', 'UNKNOWN')
            basename = os.path.basename(header.get('FILENAME', 'UNKNOWN'))
            title = f'Object: {obj}, Mode: {mode}\nFilename: {basename}'

            # make the figure for the spectral plot
            fig = Figure(figsize=(8, 5))
            FigureCanvas(fig)
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title(title)

            if atran_plot and not np.all(np.isnan(atran)):
                if wave.ndim > 1:
                    aplot = [wave.T, atran.T]
                else:
                    aplot = [wave, atran]

            # plot spectral flux
            spec_err = None if not error_plot else spec_err
            make_spectral_plot(ax, wave, spec_flux, spectral_error=spec_err,
                               scale=scale, labels=labels, colormap=colormap,
                               xunit=xunit, yunit=yunit,
                               title=title, overplot=aplot,
                               overplot_label='Atmospheric Transmission',
                               overplot_color=overplot_color,
                               watermark=watermark)

            # output filename for image
            fname = os.path.splitext(basename)[0] + '.png'
            outname = os.path.join(self.output_directory, fname)

            fig.savefig(outname, dpi=300)
            log.info(f'Saved image to {outname}')
