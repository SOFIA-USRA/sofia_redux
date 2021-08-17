# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FLITECAM Reduction pipeline steps"""

import os
import re

from astropy import log

from sofia_redux.pipeline.sofia.sofia_exception import SOFIAImportError
try:
    import sofia_redux.instruments.flitecam
except ImportError:
    raise SOFIAImportError('FLITECAM modules not installed')

from sofia_redux.pipeline.reduction import Reduction
from sofia_redux.pipeline.sofia.forcast_reduction \
    import FORCASTReduction
from sofia_redux.pipeline.sofia.parameters.flitecam_parameters \
    import FLITECAMParameters
from sofia_redux.toolkit.utilities.fits import hdinsert


class FLITECAMReduction(FORCASTReduction):
    """
    FLITECAM reduction steps.

    Primary image reduction algorithms are defined in the flitecam
    package (`sofia_redux.instruments.flitecam`).  Calibration-related
    algorithms are pulled from the `sofia_redux.calibration` package, and
    some utilities come from the `sofia_redux.toolkit` package.  This
    reduction object requires that all three packages be installed.

    This reduction object defines a method for each pipeline
    step, that calls the appropriate algorithm from its source
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
    cal_conf : dict-like
        Flux calibration and atmospheric correction configuration,
        as returned from the pipecal `pipecal_config` function.
    """
    def __init__(self):
        """Initialize the reduction object."""
        super().__init__()

        # descriptive attributes
        self.name = 'Redux'
        self.instrument = 'FLITECAM'
        self.mode = 'any'
        self.data_keys = ['File Name', 'OBJECT', 'OBSTYPE',
                          'AOR_ID', 'MISSN-ID', 'DATE-OBS',
                          'INSTCFG', 'INSTMODE',
                          'SPECTEL1', 'SPECTEL2',
                          'SLIT', 'ALTI_STA', 'ZA_START',
                          'EXPTIME', 'DTHINDEX', 'NODBEAM']

        self.pipe_name = "FLITECAM_REDUX"
        self.pipe_version = \
            sofia_redux.instruments.flitecam.__version__.replace('.', '_')

        # associations: this will be instantiated on load
        self.parameters = None

        # product type definitions for FLITECAM steps
        self.prodtype_map = {
            'check_header': 'raw',
            'correct_linearity': 'linearized',
        }
        self.prodnames = {
            'raw': 'RAW',
            'linearized': 'LNZ',
        }

        # invert the map for quick lookup of step from type
        self.step_map = {v: k for k, v in self.prodtype_map.items()}

        # this will be populated when the recipe is set
        self.prodtypes = []

        # default recipe and step names
        self.recipe = ['check_header', 'correct_linearity']
        self.processing_steps = {
            'check_header': 'Check Headers',
            'correct_linearity': 'Correct Nonlinearity',
        }

        # reduction information
        self.output_directory = os.getcwd()

        # reference file and obsmode configuration
        self.calres = {}

        # photometric flux calibration information
        self.cal_conf = None

    def getfilenum(self, filename):
        r"""
        Get the file number from a file name.

        Returns UNKNOWN if file number can't be parsed.

        Parameters
        ----------
        filename : str
            File name to parse.  Raw filenames (not starting with
            F[\d]+) are assumed to be \\*[_-][filenum].?.fits.
            Processed filenames are assumed to be \\*_[filenum].fits.

        Returns
        -------
        str or list
            File number(s), formatted to 4 digits.
        """
        basename = os.path.basename(filename)
        try:
            if re.match(r'^F\d+', basename):
                numstr = re.split('[_.]', basename)[-2]
            elif basename.endswith('.a.fits'):
                numstr = re.split('[_.-]', basename)[-3]
            else:
                numstr = re.split('[_.-]', basename)[-2]
            try:
                filenum = '{0:04d}'.format(int(numstr))
            except ValueError:
                filenum = ['{0:04d}'.format(int(n))
                           for n in numstr.split('-')]
        except (ValueError, IndexError):
            filenum = 'UNKNOWN'
        return filenum

    def getfilename(self, header, update=True, prodtype='RAW',
                    filenum='UNKNOWN'):
        """
        Create an output filename from an input header.

        Parameters
        ----------
        header : fits.Header
            Header to create filename from.
        update : bool, optional
            If set, FILENAME key will be added or updated
            in the header.
        prodtype : str, optional
            Three letter product type designator.
        filenum : str or list, optional
            List of file numbers to concatenate for filename.

        Returns
        -------
        str
           The output name.
        """
        # get flight number
        missn = header.get('MISSN-ID', 'UNKNOWN')
        flight = missn.split('_')[-1].lstrip('F')
        try:
            flight = 'F{0:04d}'.format(int(flight))
        except ValueError:
            flight = 'UNKNOWN'

        # get AOR_ID
        aorid = header.get('AOR_ID', 'UNKNOWN').replace('_', '')

        # get filter
        spectel = header.get('SPECTEL1', 'UNKNOWN').replace('_', '')

        # get config
        instcfg = header.get('INSTCFG', 'UNKNOWN').strip().upper()
        if instcfg in ['SPECTROSCOPY', 'GRISM']:
            inst = 'FC_GRI'
        else:
            inst = 'FC_IMA'

        # concatenate file numbers
        filenumstr = self._catfilenum(filenum)

        # join outname
        outname = '_'.join([flight, inst, aorid, spectel,
                            prodtype, filenumstr]).upper()
        outname = outname + '.fits'

        if update:
            hdinsert(header, 'FILENAME', outname, comment='File name')

        return outname

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
        from sofia_redux.calibration.pipecal_config import pipecal_config
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

        # read in the pipecal configuration
        if self.calres['name'] == 'IMA':
            self.cal_conf = pipecal_config(self.basehead)
            log.debug('Full pipecal configuration:')
            for key, value in self.cal_conf.items():
                log.debug('  {}: {}'.format(key, value))

        if param_class is None:  # pragma: no cover
            # this option is not currently used
            self.parameters = FLITECAMParameters(
                config=self.calres,
                pipecal_config=self.cal_conf)
        else:
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

    def _load_raw_data(self, datafile):
        """
        Robustly read raw FLITECAM data.

        Parameters
        ----------
        datafile : str
            Input filename.

        Returns
        -------
        fits.HDUList
            Flux and error data, as a FITS HDU list.
        """
        from sofia_redux.instruments.flitecam.readfits import readfits
        hdul = readfits(datafile)
        return hdul

    def check_header(self):
        """
        Check input headers.

        Calls sofia_redux.instruments.flitecam.hdcheck.hdcheck to
        compare header keywords to requirements.  Halts reduction if
        the abort parameter is True and the headers do not meet
        requirements.
        """
        from sofia_redux.instruments.flitecam.hdcheck import hdcheck

        # get parameters
        param = self.get_parameter_set()

        # check headers loaded into self.input
        valid = hdcheck(self.input, kwfile=self.calres['kwfile'])
        if not valid and param.get_value('abort'):
            msg = 'Invalid headers.'
            log.error(msg)
            self.error = msg
            self.input = []
            return

        # if data passed validation,
        # read fits files into memory for processing
        self.load_fits()

    def correct_linearity(self):
        """
        Correct flux for nonlinearity.

        Also calculates the error on the flux, after linearity
        correction.
        """
        from sofia_redux.instruments.flitecam.lincor import lincor

        # get parameters
        param = self.get_parameter_set()
        saturation = param.get_value('saturation')

        linfile = param.get_value('linfile')
        if os.path.isfile(linfile):
            log.info(f'Using linearity file {linfile}')
        else:
            raise ValueError('No linearity file provided.')

        if str(saturation).strip() == '':
            saturation = None

        outdata = []
        for i, hdul in enumerate(self.input):
            log.info('')
            log.info(f"Input: {hdul[0].header.get('FILENAME', 'UNKNOWN')}")

            result = lincor(hdul, linfile, saturation=saturation)

            outname = self.update_output(
                result, self.filenum[i], self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                self.write_output(result, outname)

            outdata.append(result)

        self.input = outdata

        # set display data to input
        self.set_display_data()
