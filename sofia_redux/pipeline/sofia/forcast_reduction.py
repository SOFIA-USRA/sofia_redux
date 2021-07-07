# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FORCAST Reduction pipeline steps"""

import os
import re
import time
import warnings

import numpy as np
from astropy import log
from astropy.io import fits
from astropy.wcs import WCS

from sofia_redux.pipeline.sofia.sofia_exception import SOFIAImportError
try:
    import sofia_redux.instruments.forcast
except ImportError:
    raise SOFIAImportError('FORCAST modules not installed')

import sofia_redux.instruments.forcast.configuration as dripconfig
from sofia_redux.instruments.forcast.getcalpath import getcalpath
from sofia_redux.instruments.forcast.readfits import readfits
from sofia_redux.pipeline.reduction import Reduction
from sofia_redux.pipeline.gui.qad_viewer import QADViewer
from sofia_redux.pipeline.sofia.parameters.forcast_parameters \
    import FORCASTParameters
from sofia_redux.toolkit.utilities.fits import hdinsert, getheader


class FORCASTReduction(Reduction):
    """
    FORCAST reduction steps.

    Primary image reduction algorithms are defined in the DRIP
    package (`sofia_redux.instruments.forcast`).  Calibration-related
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
    slit_image_recipe : list
        Alternate processing recipe to use when input data is
        marked as a slit image (SLIT!=NONE).  Calibration and
        coaddition steps are skipped.
    mosaic_recipe : list
        Alternate processing recipe to use when input is telluric-
        corrected or calibrated.  Only calibration, registration,
        and coaddition are applied.
    basehead : `astropy.io.fits.header.Header`
        Header for the first raw input file loaded, used for calibration
        configuration.
    calres : dict-like
        Reduction mode and auxiliary file configuration mapping,
        as returned from the sofia_redux.instruments.forcast
        `getcalpath` function.
    cal_conf : dict-like
        Flux calibration and atmospheric correction configuration,
        as returned from the pipecal `pipecal_config` function.
    """
    def __init__(self):
        """Initialize the reduction object."""
        super().__init__()

        # descriptive attributes
        self.name = 'DRIP'
        self.instrument = 'FORCAST'
        self.mode = ''
        self.data_keys = ['File Name', 'OBJECT', 'OBSTYPE',
                          'AOR_ID', 'MISSN-ID', 'DATE-OBS',
                          'INSTCFG', 'INSTMODE', 'SKYMODE', 'DETCHAN',
                          'DICHROIC', 'SPECTEL1', 'SPECTEL2',
                          'SLIT', 'ALTI_STA', 'ZA_START',
                          'EXPTIME', 'DTHINDEX', 'NODBEAM']

        self.pipe_name = "FORCAST_REDUX"
        self.pipe_version = \
            sofia_redux.instruments.forcast.__version__.replace('.', '_')

        # associations: this will be instantiated on load
        self.parameters = None

        # product type definitions for FORCAST steps
        self.prodtype_map = {'checkhead': 'raw',
                             'clean': 'cleaned',
                             'droop': 'drooped',
                             'nonlin': 'linearized',
                             'stack': 'stacked'}
        self.prodnames = {'raw': 'RAW',
                          'cleaned': 'CLN',
                          'drooped': 'DRP',
                          'linearized': 'LNZ',
                          'stacked': 'STK'}

        # invert the map for quick lookup of step from type
        self.step_map = {v: k for k, v in self.prodtype_map.items()}

        # this will be populated when the recipe is set
        self.prodtypes = []

        # default recipe and step names
        self.recipe = ['checkhead', 'clean', 'droop', 'nonlin',
                       'stack']
        self.processing_steps = {
            'checkhead': 'Check Headers',
            'clean': 'Clean Images',
            'droop': 'Correct Droop',
            'nonlin': 'Correct Nonlinearity',
            'stack': 'Stack Chops/Nods',
        }

        # reduction information
        self.check_input = True
        self.output_directory = os.getcwd()
        self.basehead = None
        self.calres = None
        self.filenum = []

        # store some 1D spectrum types, for display and
        # historical accommodation
        self.spec1d_prodtype = ['spectra_1d', 'merged_spectrum_1d',
                                'calibrated_spectrum_1d',
                                'combined_spectrum', 'combspec',
                                'calspec', 'mrgspec', 'mrgordspec']

    def __setattr__(self, name, value):
        """Check if recipe is being set. If so, also set prodtypes."""
        if name == 'recipe':
            # set the prodtype list
            try:
                self.prodtypes = [self.prodtype_map[step] for step in value]
            except AttributeError:
                self.prodtypes = []
        super().__setattr__(name, value)

    def getfilenum(self, filename):
        """
        Get the file number from a file name.

        Returns UNKNOWN if file number can't be parsed.

        Parameters
        ----------
        filename : str
            File name to parse.  Assumed to be \\*_[filenum].fits.

        Returns
        -------
        str or list
            File number(s), formatted to 4 digits.
        """
        try:
            numstr = re.split('[_.]', os.path.basename(filename))[-2]
            try:
                filenum = '{0:04d}'.format(int(numstr))
            except ValueError:
                filenum = ['{0:04d}'.format(int(n))
                           for n in numstr.split('-')]
        except (ValueError, IndexError):
            filenum = 'UNKNOWN'
        return filenum

    def _catfilenum(self, filenum):
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
        # concatenate file numbers
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
            filenums.sort()
            filenumstr = '-'.join([filenums[0], filenums[-1]])
        else:
            filenumstr = filenums[0]

        return filenumstr

    def getfilename(self, header, update=True, prodtype='RAW',
                    filenum='UNKNOWN'):
        """
        Create an output filename from an input header.

        Requires calibration data to be loaded (self.calres).

        Parameters
        ----------
        header : astropy.io.fits.header.Header
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
        if self.calres is None:
            raise ValueError('Cannot generate name without '
                             'loaded calibration paths.')

        # get flight number
        missn = header.get('MISSN-ID', 'UNKNOWN')
        flight = missn.split('_')[-1].lstrip('F')
        try:
            flight = 'F{0:04d}'.format(int(flight))
        except ValueError:
            flight = 'UNKNOWN'

        # get AOR_ID
        aorid = header.get('AOR_ID', 'UNKNOWN')
        aorid = aorid.replace('_', '')

        # get config from calres
        spectel = self.calres['spectel'].replace('_', '')
        if self.calres['gmode'] != -1:
            inst = 'FO_GRI'
        else:
            inst = 'FO_IMA'

        filenumstr = self._catfilenum(filenum)

        outname = '_'.join([flight, inst, aorid, spectel,
                            prodtype, filenumstr]).upper()
        outname = outname + '.fits'

        if update:
            hdinsert(header, 'FILENAME', outname,
                     comment='File name')

        return outname

    def set_display_data(self, raw=False, filenames=None):
        """
        Store display data for QAD viewer.

        Parameters
        ----------
        raw : bool
            If True, display data is taken from self.rawfiles.
            If False, display data is taken from self.input
        filenames : list of str, optional
            If provided and `raw` is False, file names will be
            passed to QADViewer instead of self.input.
        """
        self.display_data = {}
        if raw:
            data_list = self.raw_files
        elif filenames is not None:
            data_list = filenames
        else:
            data_list = self.input
        self.display_data['QADViewer'] = data_list

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
        header : `astropy.io.fits.header.Header`
            Header to update.
        """
        # update procstat if necessary
        procstat = header.get('PROCSTAT', 'UNKNOWN').upper()
        if procstat in ['UNKNOWN', 'LEVEL_0', 'LEVEL_1']:
            hdinsert(header, 'PROCSTAT', 'LEVEL_2',
                     comment='Processing status')

        # copy AOR_ID, OBS_ID, and MISSN-ID to ASSC* keys
        # if not yet present
        aorid = header.get('AOR_ID', 'UNKNOWN').upper()
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
        hdul : `astropy.io.fits.HDUList`
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
        # update header with product type
        hdinsert(hdul[0].header, 'PRODTYPE',
                 prodtype, comment='Product type')

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

        outname = self.getfilename(hdul[0].header, update=True,
                                   prodtype=self.prodnames[prodtype],
                                   filenum=filenum)
        return outname

    def write_output(self, hdul, outname):
        """
        Write an output FITS file to disk.

        Outname is joined to self.output_directory, before writing.

        Parameters
        ----------
        hdul : `astropy.io.fits.HDUList`
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
            Parameters to instantiate, if not FORCASTParameters.
        """
        # call the parent method to initialize
        # reduction variables
        Reduction.load(self, data)

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
        prodtype = self.basehead.get('PRODTYPE', 'UNKNOWN')

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

        if param_class is None:  # pragma: no cover
            self.parameters = FORCASTParameters(
                drip_cal_config=self.calres,
                drip_config=dripconfig.configuration)
        else:
            self.parameters = param_class(
                drip_cal_config=self.calres,
                drip_config=dripconfig.configuration)

        # load the files in immediately
        self.load_fits(intermediate=intermediate)

    def _load_raw_data(self, datafile):
        """
        Read raw FORCAST data and calculate errors.

        Data is rearranged into FLUX and ERROR extensions.

        Parameters
        ----------
        datafile : str
            Input filename.

        Returns
        -------
        fits.HDUList
            Flux and error data, as a FITS HDU list.
        """
        # use the sofia_redux.instruments.forcast function
        # to read the data and calculate the error
        hdul = readfits(datafile, fitshdul=True, stddev=True)

        # add the config files to the header
        header = hdul[0].header
        conffile = self.calres['conffile'].split(
            self.calres['pathcal'])[-1]
        hdinsert(header, 'CONFFILE', conffile,
                 comment='DRIP config file')
        return hdul

    def load_fits(self, intermediate=False):
        """
        Load FITS data into memory.

        Handles raw data, as well as intermediate data.  Intermediate
        data may have been produced by the current pipeline version
        (multiple HDUs expected), or from the v1 IDL pipeline
        (single HDU expected).

        Loaded data are stored in the input attribute.

        Parameters
        ----------
        intermediate : bool
            If False, the
            `sofia_redux.instruments.forcast.readfits.readfits`
            function will be used to read in the data and calculate
            the associated error images.  If True, the data will
            just be read in from disk.
        """
        self.input = []
        self.filenum = []
        filenames = []
        for i, datafile in enumerate(self.raw_files):
            if not intermediate:
                # load raw data and rearrange into appropriate
                # extensions
                hdul = self._load_raw_data(datafile)
            else:
                # otherwise, just read the data
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

                prodtype = str(hdul[0].header.get('PRODTYPE', 'UNKNOWN'))
                if len(hdul) == 1:
                    if prodtype.strip().lower() in self.spec1d_prodtype or \
                            prodtype.strip().lower() == 'response_spectrum':
                        # 1D spectra:
                        # leave as is and pass filenames to display instead
                        filenames.append(datafile)
                    else:
                        # for legacy data
                        data = hdul[0].data
                        header = hdul[0].header
                        hdinsert(header, 'EXTNAME', 'FLUX',
                                 comment='extension name')
                        primary = fits.PrimaryHDU(data=data[0, :, :],
                                                  header=header)

                        # make a basic extension header from
                        # the WCS in the primary
                        wcs = WCS(header)
                        ehead = wcs.to_header(relax=True)
                        err = fits.ImageHDU(data=np.sqrt(data[1, :, :]),
                                            header=ehead, name='ERROR')
                        if data.shape[0] > 2:
                            # same for exposure map
                            expmap = fits.ImageHDU(
                                data=data[2, :, :],
                                header=ehead, name='EXPOSURE')
                            hdul = fits.HDUList([primary, err, expmap])
                        else:
                            hdul = fits.HDUList([primary, err])

            # get the file number from the filename
            self.filenum.append(self.getfilenum(datafile))

            # update required SOFIA keys if necessary
            self.update_sofia_keys(hdul[0].header)

            self.input.append(hdul)

        # store the data in display variables
        self.set_display_data(raw=(not intermediate), filenames=filenames)

    def register_viewers(self):
        """Return a new QADViewer."""
        viewers = [QADViewer()]
        return viewers

    def checkhead(self):
        """
        Check input headers.

        Calls sofia_redux.instruments.forcast.hdcheck.hdcheck to
        compare header keywords to requirements.  Halts reduction if
        the abort parameter is True and the headers do not meet
        requirements.
        """
        from sofia_redux.instruments.forcast.hdcheck import hdcheck

        # get parameters
        param = self.get_parameter_set()

        # check headers in files on disk
        valid = hdcheck(self.raw_files, kwfile=self.calres['kwfile'])
        if not valid and param.get_value('abort'):
            msg = 'Invalid headers.'
            log.error(msg)
            self.error = msg
            self.input = []
            return

        # if passed validation,
        # read fits files into memory for processing
        self.load_fits()

    def clean(self):
        """
        Clean bad pixels from image data.

        Calls `sofia_redux.instruments.forcast.check_readout_shift`
        to determine if the 16-pixel readout error is present,
        and corrects for it if desired.

        Calls `sofia_redux.instruments.forcast.clean` to clean bad
        pixels identified in a bad pixel mask.  Bad pixels may be either
        interpolated over or propagated as NaN values.
        """

        # import from sofia_redux.instruments.forcast
        from sofia_redux.instruments.forcast.clean import clean
        from sofia_redux.instruments.forcast.check_readout_shift \
            import check_readout_shift

        # get parameters
        param = self.get_parameter_set()
        propagate_nan = not param.get_value('interpolate')

        # readout shift parameters
        shift_files = param.get_value('shiftfile')
        do_files = [False] * len(self.input)
        if shift_files.strip() != '':
            if shift_files.lower() == 'all':
                do_files = [True] * len(self.input)
            else:
                try:
                    file_nums = [int(f) for f in shift_files.split(';')]
                except ValueError:
                    msg = "Shifted files must be semicolon-separated " \
                          "integers or the word " \
                          "'all'.  Input value: '{}'.".format(shift_files)
                    log.error(msg)
                    self.error = msg
                    return

                for fn in file_nums:
                    try:
                        do_files[fn - 1] = True
                    except IndexError:
                        msg = "Shifted file value '{}' out of range " \
                              "for loaded data".format(fn)
                        log.error(msg)
                        self.error = msg
                        return
            autoshift = False
        else:
            autoshift = param.get_value('autoshift')

        badfile = param.get_value('badfile')
        if os.path.isfile(badfile):
            log.info('Using bad pixel file {}'.format(badfile))
            log.info('')
            badmap_int = readfits(badfile)
            badmap = (badmap_int > 0)
            badfile = badfile.split(self.calres['pathcal'])[-1]
        else:
            log.warning('No bad pixel file provided.')
            badfile = 'None'
            badmap = None

        outdata = []
        for i, hdul in enumerate(self.input):
            header = hdul[0].header
            data = hdul['FLUX'].data
            err = hdul['ERROR'].data

            # check for readout shift
            if autoshift:
                do_files[i] = check_readout_shift(data, header)

            # do readout shift if necessary
            if do_files[i]:
                log.info('Shifting file {} ({}) by 16 pixels in x'.format(
                    i + 1, os.path.basename(self.raw_files[i])))

                data = np.roll(data, -16, axis=-1)
                err = np.roll(err, -16, axis=-1)

            # clean bad pixels
            result = clean(data, badmap, header,
                           variance=err**2,
                           propagate_nan=propagate_nan)
            if result is None:
                msg = "Problem in sofia_redux.instruments.forcast.clean."
                log.error(msg)
                self.error = msg
                return
            else:
                outimg, outvar = result

            # add badfile to header
            hdinsert(header, 'BDPXFILE', badfile,
                     comment='Bad pixel mask')

            # update hdul with result
            # (header may be updated in place, but it's
            # safer to make sure it's properly stored)
            hdul[0].header = header
            hdul['FLUX'].data = outimg
            hdul['ERROR'].data = np.sqrt(outvar)

            outname = self.update_output(
                hdul, self.filenum[i], self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                self.write_output(hdul, outname)

            outdata.append(hdul)

        self.input = outdata

        # set display data to input
        self.set_display_data()

    def droop(self):
        """
        Correct for 'droop' detector response effect.

        Calls `sofia_redux.instruments.forcast.droop.droop`.
        The droop correction may be tuned with the fracdroop parameter.
        """

        # import from sofia_redux.instruments.forcast
        from sofia_redux.instruments.forcast.droop import droop

        # get parameters
        param = self.get_parameter_set()
        droop_frac = param.get_value('fracdroop')

        outdata = []
        for i, hdul in enumerate(self.input):
            header = hdul[0].header
            data = hdul['FLUX'].data
            err = hdul['ERROR'].data

            # droop correct
            result = droop(data, header, frac=droop_frac,
                           variance=err**2)

            if result is None:
                msg = "Problem in sofia_redux.instruments.forcast.droop."
                log.error(msg)
                self.error = msg
                return
            else:
                outimg, outvar = result

            # update hdul with result
            hdul[0].header = header
            hdul['FLUX'].data = outimg
            hdul['ERROR'].data = np.sqrt(outvar)

            outname = self.update_output(
                hdul, self.filenum[i], self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                self.write_output(hdul, outname)

            outdata.append(hdul)

        self.input = outdata
        self.set_display_data()

    def nonlin(self):
        """
        Correct for detector nonlinearity.

        Calls `sofia_redux.instruments.forcast.imgnonlin.imgnonlin`.
        The image section that determines the background levels for the
        correction may be specified in parameters.
        """

        # import from sofia_redux.instruments.forcast
        from sofia_redux.instruments.forcast.background import background
        from sofia_redux.instruments.forcast.imgnonlin import imgnonlin

        # get parameters
        param = self.get_parameter_set()
        secctr = param.get_value('secctr')
        secsize = param.get_value('secsize')

        # read section parameter
        try:
            cx, cy = [int(n) for n in secctr.split(',')]
            sx, sy = [int(n) for n in secsize.split(',')]
            data_section = (cx, cy, sx, sy)
        except (ValueError, IndexError):
            msg = "Invalid background section specified."
            log.error(msg)
            self.error = msg
            return

        outdata = []
        for i, hdul in enumerate(self.input):
            header = hdul[0].header
            data = hdul['FLUX'].data
            err = hdul['ERROR'].data

            # calculate median background values
            bglevel = background(data, data_section, header, 'median')

            # use background level to correct for nonlinearity
            result = imgnonlin(data, header,
                               siglev=bglevel, variance=err**2)

            if result is None:
                # just continue with input data, in this case
                outimg = data
                outerr = err
            else:
                outimg, outvar = result
                outerr = np.sqrt(outvar)

            # update hdul with result
            hdul[0].header = header
            hdul['FLUX'].data = outimg
            hdul['ERROR'].data = outerr

            outname = self.update_output(
                hdul, self.filenum[i], self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                self.write_output(hdul, outname)

            outdata.append(hdul)

        self.input = outdata
        self.set_display_data()

    def stack(self):
        """
        Stack chop/nod frames.

        Calls `sofia_redux.instruments.forcast.stack.stack`.
        Images are chop- and nod-subtracted and cleaned of
        "jailbar" artifacts.
        """

        # import from sofia_redux.instruments.forcast
        from sofia_redux.instruments.forcast.stack import stack

        # get parameters
        param = self.get_parameter_set()
        add_frames = param.get_value('add_frames')
        jbclean = param.get_value('jbclean')
        bgscale = param.get_value('bgscale')
        bgsub = param.get_value('bgsub')
        secctr = param.get_value('secctr')
        secsize = param.get_value('secsize')
        bgstat = param.get_value('bgstat')

        # read section parameter
        try:
            cx, cy = [int(n) for n in secctr.split(',')]
            sx, sy = [int(n) for n in secsize.split(',')]
            data_section = (cx, cy, sx, sy)
        except (ValueError, IndexError):
            msg = "Invalid background section specified."
            log.error(msg)
            self.error = msg
            return

        # check for sky frame parameter
        if add_frames:
            log.info("All frames added, regardless of INSTMODE.")
            bgscale = False
            bgsub = False

        # set parameter values in config
        dripconfig.configuration['bgscale'] = bgscale
        dripconfig.configuration['bgsub'] = bgsub
        dripconfig.configuration['nlinsection'] = data_section
        if jbclean:
            dripconfig.configuration['jbclean'] = 'median'
        else:
            dripconfig.configuration['jbclean'] = 'none'

        outdata = []
        for i, hdul in enumerate(self.input):
            header = hdul[0].header
            data = hdul['FLUX'].data
            err = hdul['ERROR'].data

            if add_frames:
                header['INSTMODE'] = 'STARE'
                header['SKYMODE'] = 'STARE'

            # subtract chops and nods
            result = stack(data, header, variance=err**2, stat=bgstat)

            if result is None:
                msg = "Problem in drip.stack."
                log.error(msg)
                self.error = msg
                return
            else:
                outimg, outvar = result

            # update BUNIT keywords
            hdinsert(header, 'BUNIT', 'Me/s', comment='Data units')
            hdinsert(hdul[1].header, 'BUNIT', 'Me/s', comment='Data units')

            # add exposure keywords
            if self.calres['cnmode'] == 'NMC':
                nexp = 2
            else:
                nexp = 1
            detitime = header.get('DETITIME', 0.0)
            hdinsert(header, 'NEXP', nexp,
                     comment='Approximate number of exposures')
            hdinsert(header, 'EXPTIME', nexp * detitime / 2.0,
                     comment='Nominal on-source integration time [s]')

            # update hdul with result
            hdul[0].header = header
            hdul['FLUX'].data = outimg
            hdul['ERROR'].data = np.sqrt(outvar)

            outname = self.update_output(
                hdul, self.filenum[i], self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                self.write_output(hdul, outname)

            outdata.append(hdul)

        self.input = outdata
        self.set_display_data()
