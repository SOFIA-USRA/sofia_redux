# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FORCAST Imaging Reduction pipeline steps"""

import os
import warnings

import numpy as np
from astropy import log
from astropy.io import fits
from astropy.wcs import WCS

from sofia_redux.pipeline.sofia.sofia_exception import SOFIAImportError
try:
    import sofia_redux.instruments.forcast
    assert sofia_redux.instruments.forcast
except ImportError:
    raise SOFIAImportError('FORCAST modules not installed')

from sofia_redux.calibration.pipecal_config import pipecal_config
from sofia_redux.calibration.pipecal_error import PipeCalError
import sofia_redux.instruments.forcast.configuration as dripconfig
from sofia_redux.instruments.forcast.getcalpath import getcalpath
from sofia_redux.instruments.forcast.getpar import getpar
from sofia_redux.pipeline.reduction import Reduction
from sofia_redux.pipeline.sofia.forcast_reduction import FORCASTReduction
from sofia_redux.pipeline.sofia.parameters.forcast_imaging_parameters \
    import FORCASTImagingParameters
from sofia_redux.toolkit.utilities.fits import hdinsert, getheader

# these imports are not used here, but are needed to avoid
# a numba error on linux systems
from sofia_redux.toolkit.resampling import tree
assert tree


class FORCASTImagingReduction(FORCASTReduction):
    """
    FORCAST imaging reduction steps.

    Primary image reduction algorithms are defined in the DRIP
    package (`sofia_redux.instruments.forcast`).  Calibration-related
    algorithms are pulled from the `sofia_redux.calibration` package,
    and some utilities come from the `sofia_redux.toolkit` package.
    This reduction object requires that all three packages be installed.

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
        as returned from the sofia_redux.instruments.forcast `getcalpath`
        function.
    cal_conf : dict-like
        Flux calibration and atmospheric correction configuration,
        as returned from the sofia_redux.calibration `pipecal_config`
        function.
    """
    def __init__(self):
        """Initialize the reduction object."""
        super().__init__()

        # descriptive attributes specific to imaging
        self.mode = 'Imaging'

        # product type definitions for imaging steps
        self.prodtype_map.update(
            {'undistort': 'undistorted',
             'merge': 'merged',
             'register': 'registered',
             'tellcor': 'telluric_corrected',
             'coadd': 'coadded',
             'fluxcal': 'calibrated',
             'mosaic': 'mosaic',
             'imgmap': 'imgmap'})
        self.prodnames.update(
            {'undistorted': 'UND',
             'merged': 'MRG',
             'registered': 'REG',
             'telluric_corrected': 'TEL',
             'coadded': 'COA',
             'calibrated': 'CAL',
             'mosaic': 'MOS',
             'imgmap': 'IMP'})

        # invert the map for quick lookup of step from type
        self.step_map = {v: k for k, v in self.prodtype_map.items()}

        # default recipe and step names
        self.recipe = ['checkhead', 'clean', 'droop', 'nonlin',
                       'stack', 'undistort', 'merge', 'register',
                       'tellcor', 'coadd', 'fluxcal', 'imgmap']
        self.processing_steps.update(
            {'undistort': 'Correct Distortion',
             'merge': 'Merge Chops/Nods',
             'register': 'Register Images',
             'tellcor': 'Telluric Correct',
             'coadd': 'Coadd',
             'fluxcal': 'Flux Calibrate',
             'mosaic': 'Mosaic',
             'imgmap': 'Make Image Map'})

        # also define a couple alternate recipes for
        # special situations
        self.default_recipe = self.recipe.copy()
        self.slit_image_recipe = ['checkhead', 'clean', 'droop', 'nonlin',
                                  'stack', 'undistort', 'merge']
        self.mosaic_recipe = ['fluxcal', 'register', 'coadd', 'imgmap']

        # photometric flux calibration information
        self.cal_conf = None

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
        - Use the base header to load a calibration configuration
          (`sofia_redux.calibration.pipecal_config`).
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

        # check for non-standard recipes first
        if self.calres['slit'] not in ['UNKNOWN', 'NONE', '']:
            # slit image
            log.info('Slit image detected; using alternate recipe')
            self.recipe = self.slit_image_recipe
        elif prodtype in ['telluric_corrected', 'calibrated']:
            # cal files to mosaic
            log.info('TEL or CAL files detected; using mosaic recipe')
            self.recipe = self.mosaic_recipe
        else:
            self.recipe = self.default_recipe

        # get remaining recipe
        if prodtype in self.prodtypes:
            pidx = self.prodtypes.index(prodtype)
            self.recipe = self.recipe[pidx + 1:]
            if len(self.recipe) == 0:
                raise ValueError("No steps to run for "
                                 "prodtype '{}'.".format(prodtype))
            intermediate = True
        elif prodtype.upper() != 'UNKNOWN':
            intermediate = True

        # read in the pipecal configuration
        self.cal_conf = pipecal_config(self.basehead)
        log.debug('Full pipecal configuration:')
        for key, value in self.cal_conf.items():
            log.debug('  {}: {}'.format(key, value))

        if param_class is None:
            self.parameters = FORCASTImagingParameters(
                drip_cal_config=self.calres,
                drip_config=dripconfig.configuration,
                pipecal_config=self.cal_conf)
        else:  # pragma: no cover
            # this option is not currently used
            self.parameters = param_class(
                drip_cal_config=self.calres,
                drip_config=dripconfig.configuration,
                pipecal_config=self.cal_conf)

        # if not starting from raw data, load the files in
        # immediately
        if intermediate:
            self.load_fits(intermediate=True)
        else:
            # just load headers
            self.input = []
            for datafile in self.raw_files:
                self.input.append(fits.getheader(datafile))

    def reorganize_c2nc2(self):
        """
        Fix old-style C2NC2 files to newer data organization.

        Early science flights took C2NC2 data in C2 mode,
        with nods separated into separate files.  The nod pattern
        was: A B A A B A A B.  Later flights used the same nod
        pattern, but repackaged B nods with the A nods into 5
        files: AB BA AB BA AB.  This function performs the same
        function for old-style data, so that all further steps
        may be run in the same way as for the new-style data.
        """
        from sofia_redux.instruments.forcast.hdmerge import hdmerge

        log.info('Reorganizing old-style C2NC2 to new-style A/B files')

        # assume pattern is ABAABAAB
        last_hd = None
        last_nod = None
        last_idx = None
        last_fn = None
        new_filenum = []
        new_input = []
        for i, hdul in enumerate(self.input):
            idx = int(hdul[0].header['DTHINDEX'])
            fn = self.filenum[i]

            if idx in [2, 5, 8]:
                nodbeam = 'B'
            else:
                nodbeam = 'A'

            if (last_nod is not None
                    and last_nod != nodbeam
                    and abs(idx - last_idx) == 1):
                # neighbor A and B found

                # keep file numbers
                new_filenum.append([last_fn, fn])

                # stack data
                new_data = np.vstack([last_hd[0].data, hdul[0].data])
                new_err = np.vstack([last_hd[1].data, hdul[1].data])

                hdr_list = [last_hd[0].header, hdul[0].header]
                if nodbeam == 'A':
                    ref_hdr = hdr_list[1]
                else:
                    ref_hdr = hdr_list[0]
                new_hdr = hdmerge(hdr_list, reference_header=ref_hdr)
                new_hdr['NODBEAM'] = last_nod
                log.debug(new_hdr['ASSC_OBS'])

                new_hd = last_hd.copy()
                new_hd[0].data = new_data
                new_hd[1].data = new_err
                new_hd[0].header = new_hdr
                new_input.append(new_hd)

            last_hd = hdul
            last_nod = nodbeam
            last_idx = idx
            last_fn = fn

        self.input = new_input
        self.filenum = new_filenum

    def filter_shift(self):
        """
        For early data, shift reference pixels for filter offsets.

        Header keywords CRPIX1 and CRPIX2 for all files in
        self.input are updated with specified offset values.

        Pixel offsets by filter are listed in the
        sofia_redux.instruments.forcast/data/filtershift.txt file.
        After 2015, filter offsets were applied by the instrument
        software, so this step is not required.
        """
        log.info('Shifting CRPIX for filter offsets.')
        px = self.calres['pixshiftx']
        py = self.calres['pixshifty']
        log.info('Pixel offsets (x,y): {}, {}'.format(px, py))
        for hdul in self.input:
            for hdu in hdul:
                hdu.header['CRPIX1'] += px
                hdu.header['CRPIX2'] += py

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
            sofia_redux.instruments.forcast.readfits.readfits` function will
            be used to read in the data and calculate the
            associated error images.  If True, the data will
            just be read in from disk.
        """
        self.input = []
        self.filenum = []
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

                # for legacy data
                if len(hdul) == 1:
                    data = hdul[0].data
                    header = hdul[0].header
                    hdinsert(header, 'EXTNAME', 'FLUX',
                             comment='extension name')
                    primary = fits.PrimaryHDU(data=data[0, :, :],
                                              header=header)

                    # fix a couple known bad old WCS keywords
                    bad_keys = ['XPIXELSZ', 'YPIXELSZ']
                    for bad_key in bad_keys:
                        try:
                            del primary.header[bad_key]
                        except KeyError:
                            pass

                    # make a basic extension header from
                    # the WCS in the primary
                    try:
                        wcs = WCS(primary.header)
                    except (ValueError, IndexError,
                            KeyError, MemoryError):  # pragma: no cover
                        log.warning('Bad WCS in input')
                        ehead = fits.Header()
                    else:
                        ehead = wcs.to_header(relax=True)
                    hdinsert(ehead, 'BUNIT',
                             header.get('BUNIT', 'UNKNOWN'),
                             comment='Data units')

                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        err_data = np.sqrt(data[1, :, :])
                    err = fits.ImageHDU(data=err_data,
                                        header=ehead, name='ERROR')
                    if data.shape[0] > 2:
                        # same for exposure map
                        hdinsert(ehead, 'BUNIT', 's', comment='Data units')
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

        # rearrange old-style C2NC2 data if necessary
        if not intermediate and self.calres['cnmode'] == 'C2NC2':
            self.reorganize_c2nc2()

        # update CRPIX for filter shifts if necessary
        if not intermediate \
            and (self.calres.get('pixshiftx', 0) != 0
                 or self.calres.get('pixshifty', 0) != 0):
            self.filter_shift()

        # store the data in display variables
        self.set_display_data(raw=True)

    def undistort(self):
        """
        Correct for optical distortion.

        Calls `sofia_redux.instruments.forcast.undistort.undistort`.
        The transformation algorithm may be specified in parameters.
        Detailed pinhole model parameters should be specified in the
        DRIP config file.
        """

        # import from sofia_redux.instruments.forcast
        from sofia_redux.instruments.forcast.undistort import undistort

        # get parameters
        param = self.get_parameter_set()

        # set pinfile from calconf
        pinfile = param.get_value('pinfile')
        if os.path.isfile(pinfile):
            pinname = pinfile.split(self.calres['pathcal'])[-1]
            log.info('Using pinhole file {}'.format(pinname))
            log.info('')
        else:
            msg = 'No pinhole file provided.'
            log.error(msg)
            self.error = msg
            return

        outdata = []
        if param.get_value('save'):
            display_files = []
        else:
            display_files = None
        for i, hdul in enumerate(self.input):
            header = hdul[0].header
            data = hdul['FLUX'].data
            err = hdul['ERROR'].data

            # correct for optical distortion
            result = undistort(
                data, header, variance=err**2,
                transform_type=param.get_value('transform_type'),
                extrapolate=param.get_value('extrapolate'),
                pinhole=pinfile)
            if result is None:
                msg = "Problem in sofia_redux.instruments.forcast.undistort."
                log.error(msg)
                self.error = msg
                return
            else:
                outimg, outvar = result

            # add pinfile to header
            hdinsert(header, 'PINFILE', pinname,
                     comment='Pinhole mask file')

            # update hdul with result
            hdul[0].header = header
            hdul['FLUX'].data = outimg
            hdul['ERROR'].data = np.sqrt(outvar)

            outname = self.update_output(
                hdul, self.filenum[i], self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                outname = self.write_output(hdul, outname)
                display_files.append(outname)

            outdata.append(hdul)

        self.input = outdata
        self.set_display_data(filenames=display_files)

    def merge(self):
        """
        Merge on-array chops/nods.

        Calls `sofia_redux.instruments.forcast.merge.merge`.
        Merging algorithm may be specified in parameters.
        """

        # import from sofia_redux.instruments.forcast
        from sofia_redux.instruments.forcast.merge import merge

        # get parameters
        param = self.get_parameter_set()

        cormerge_idx = param['cormerge']['option_index']
        dripconfig.configuration['cormerge'] = \
            self.parameters.merge_opt[cormerge_idx]

        outdata = []
        if param.get_value('save'):
            display_files = []
        else:
            display_files = None
        for i, hdul in enumerate(self.input):
            header = hdul[0].header
            data = hdul['FLUX'].data
            err = hdul['ERROR'].data

            # merge chops and nods if desired
            # also normalize fluxes for the number of on-source
            # exposures, and rotate image to standard coordinates
            # (North up, East left)
            normmap = np.zeros_like(data)
            result = merge(data, header, variance=err**2,
                           normmap=normmap, strip_border=True)
            if result is None:
                msg = "Problem in sofia_redux.instruments.forcast.merge."
                log.error(msg)
                self.error = msg
                return
            else:
                outimg, outvar = result

            # update integration time
            nexp = np.nanmax(normmap)
            itime = header['DETITIME'] / 2.0
            max_exptime = nexp * itime
            hdinsert(header, 'EXPTIME', max_exptime,
                     comment='Nominal on-source integration time [s]')
            hdinsert(header, 'NEXP', nexp,
                     comment='Approximate number of exposures')

            # change normmap to an exposure time map
            normmap *= itime

            # update hdul with result
            hdul[0].header = header
            hdul['FLUX'].data = outimg

            # update WCS for error extension
            wcs = WCS(header)
            ehead = wcs.to_header(relax=True)
            hdinsert(ehead, 'BUNIT',
                     header.get('BUNIT', 'UNKNOWN'),
                     comment='Data units')
            hdul[1] = fits.ImageHDU(data=np.sqrt(outvar), header=ehead,
                                    name='ERROR')

            # append normmap
            nhead = ehead.copy()
            nhead['BUNIT'] = 's'
            nmap_hdu = fits.ImageHDU(data=normmap, header=nhead,
                                     name='EXPOSURE')
            hdul.append(nmap_hdu)

            outname = self.update_output(
                hdul, self.filenum[i], self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                outname = self.write_output(hdul, outname)
                display_files.append(outname)

            outdata.append(hdul)

        self.input = outdata
        self.set_display_data(filenames=display_files)

    def register(self):
        """
        Register frames to a reference coordinate system.

        Calls `sofia_redux.instruments.forcast.register_datasets.get_shifts`.
        Registration algorithm may be specified in parameters.
        """
        # import from sofia_redux.instruments.forcast
        from sofia_redux.instruments.forcast.register_datasets \
            import get_shifts, resize_datasets

        # get parameters
        param = self.get_parameter_set()
        corcoadd_idx = param['corcoadd']['option_index']
        corcoadd = self.parameters.merge_opt[corcoadd_idx]

        # set a couple parameter values in config
        dripconfig.configuration['xyshift'] = param.get_value('xyshift')
        dripconfig.configuration['mfwhm'] = param.get_value('mfwhm')

        offsets = param.get_value('offsets')
        if str(offsets).strip().lower() not in ['', 'none']:
            log.debug('Offsets: {}'.format(offsets))
            overrides = list(offsets.split(';'))
            if len(overrides) != len(self.input):
                msg = "Number of offsets (separated by ';') " \
                      "does not match number of images."
                log.error(msg)
                self.error = msg
                return
            for i, ofs in enumerate(overrides):
                try:
                    ofx, ofy = ofs.split(',')
                    overrides[i] = (-1 * float(ofx), -1 * float(ofy))
                except (ValueError, IndexError):
                    msg = "Must provide valid x,y offsets for " \
                          "each image."
                    log.error(msg)
                    self.error = msg
                    return
            corcoadd = 'OVERRIDE'
        else:
            overrides = None
        log.debug('Overrides: {}'.format(overrides))

        log.info('Algorithm: {}'.format(corcoadd))

        if corcoadd == 'WCS':
            shifts = [(0., 0.)] * len(self.input)
        else:
            # organize data into datasets for registration
            datasets = []
            dcrpix = []
            for hdul in self.input:
                header = hdul[0].header
                data = hdul['FLUX'].data
                err = hdul['ERROR'].data
                nmap = hdul['EXPOSURE'].data

                datasets.append((data, header, err ** 2, nmap))
                dcrpix.append([header.get('DCRPIX1', 0.0),
                               header.get('DCRPIX2', 0.0)])
            dcrpix = np.array(dcrpix)

            # resize to the same size -- necessary for cross-correlation
            if corcoadd == 'XCOR' or corcoadd == 'CENTROID':
                datasets = resize_datasets(datasets)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                shifts = get_shifts(datasets, user_shifts=overrides,
                                    algorithm=corcoadd, do_wcs_shift=True)

            # check for errors, correct for relative image changes
            for idx, s in enumerate(shifts):
                if s is None or not np.all(np.isfinite(s)):
                    log.warning("Failed to register dataset %i; "
                                "setting shift to 0." % idx)
                    shifts[idx] = (0., 0.)
                elif corcoadd.lower() == 'header':
                    # DCRPIX1,2 records change to CRPIX values
                    # from undistort and merge -- if it has changed
                    # differently from the reference image, the change
                    # needs to be added into header shifts, which
                    # are calculated from dither values only
                    log.debug('Original shift: {}'.format(shifts[idx]))
                    log.debug('Delta CRPIX: {}'.format(dcrpix[idx]))
                    log.debug('Reference delta CRPIX: {}'.format(dcrpix[0]))
                    shifts[idx] += dcrpix[idx] - dcrpix[0]
                    log.debug('New shift: {}'.format(shifts[idx]))

        outdata = []
        disp_shifts = []
        if param.get_value('save'):
            display_files = []
        else:
            display_files = None
        for i, hdul in enumerate(self.input):

            shift = shifts[i]
            shiftstr = '{:.2f},{:.2f}'.format(*shift)
            disp_shifts.append(shiftstr)

            # update the header with the shifts
            for hdu in hdul:
                hdu.header['CRPIX1'] += shift[0]
                hdu.header['CRPIX2'] += shift[1]

            # add some history messages
            header = hdul[0].header
            hdinsert(header, 'HISTORY',
                     'Register: Method: {}'.format(corcoadd))
            hdinsert(header, 'HISTORY',
                     'Register: WCS shifts applied: {}'.format(shiftstr))

            outname = self.update_output(
                hdul, self.filenum[i], self.prodtypes[self.step_index])
            outdata.append(hdul)

            # save if desired
            if param.get_value('save'):
                outname = self.write_output(hdul, outname)
                display_files.append(outname)

        log.info('CRPIX shifts used:')
        log.info(';'.join(disp_shifts))

        self.input = outdata
        self.set_display_data(filenames=display_files)

    def tellcor(self):
        """
        Correct for atmospheric absorption.

        Calls `sofia_redux.calibration.pipecal_config.pipecal_config`
        and `sofia_redux.calibration.pipecal_util.apply_tellcor`.
        For standards, photometry is performed with
        `sofia_redux.calibration.pipecal_util.run_photometry`.
        """
        from sofia_redux.calibration.pipecal_util \
            import apply_tellcor, run_photometry
        from sofia_redux.calibration.pipecal_config import pipecal_config

        # get parameters
        param = self.get_parameter_set()

        outdata = []
        if param.get_value('save'):
            display_files = []
        else:
            display_files = None
        for i, hdul in enumerate(self.input):
            header = hdul[0].header
            data = hdul['FLUX'].data
            err = hdul['ERROR'].data

            # use pipecal to telluric-correct

            # allow config to vary by file
            # (in case data spans multiple flight series)
            config = pipecal_config(header)
            outimg, outvar = apply_tellcor(
                data, header, config, variance=err**2)

            # run photometry for standards
            if self.calres['obstype'] == 'STANDARD_FLUX':
                log.info('')
                try:
                    run_photometry(outimg, header, outvar, config,
                                   allow_badfit=True)
                except PipeCalError:  # pragma: no cover
                    log.warning('Photometry failed.')
                log.info('')

            # update hdul with result
            hdul[0].header = header
            hdul['FLUX'].data = outimg
            hdul['ERROR'].data = np.sqrt(outvar)

            outname = self.update_output(
                hdul, self.filenum[i], self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                outname = self.write_output(hdul, outname)
                display_files.append(outname)

            outdata.append(hdul)

        self.input = outdata
        self.set_display_data(filenames=display_files)

    def coadd(self):
        """
        Combine registered images.

        Calls `sofia_redux.toolkit.image.combine.combine_images` for
        image coaddition.  For standards, photometry is run on
        the coadded image with
        `sofia_redux.calibration.pipecal_util.run_photometry`.
        Input headers are merged with
        `sofia_redux.instruments.forcast.hdmerge.hdmerge`.

        The combination method may be configured in parameters,
        or coadd may be skipped entirely if desired.  In this case,
        a COA file is written to disk for each input file.
        """
        from sofia_redux.calibration.pipecal_util import run_photometry
        from sofia_redux.instruments.forcast.hdmerge import hdmerge
        from sofia_redux.instruments.forcast.coadd import coadd

        # get parameters
        param = self.get_parameter_set()
        do_coadd = not param.get_value('skip_coadd')

        if not do_coadd:
            # just write COA files to disk for each input
            if param.get_value('save'):
                display_files = []
            else:
                display_files = None
            for i, hdul in enumerate(self.input):
                outname = self.update_output(
                    hdul, self.filenum[i],
                    self.prodtypes[self.step_index])
                if param.get_value('save'):
                    outname = self.write_output(hdul, outname)
                    display_files.append(outname)
            self.set_display_data(filenames=display_files)
            return

        reference = param.get_value('reference')
        method = param.get_value('method')
        weighted = param.get_value('weighted')
        robust = param.get_value('robust')
        sigma = param.get_value('threshold')
        maxiters = param.get_value('maxiters')
        smoothing = param.get_value('smoothing')

        if 'target' in str(reference).lower():
            log.info('Correcting for target motion, if necessary.')
            reference = 'target'
        else:
            log.info('Using first image as reference WCS.')
            reference = 'first'

        hdr_list = []
        data_list = []
        var_list = []
        exp_list = []
        for hdul in self.input:
            hdr_list.append(hdul[0].header)
            data_list.append(hdul['FLUX'].data)
            var_list.append(hdul['ERROR'].data**2)
            exp_list.append(hdul['EXPOSURE'].data)

        outhdr, outdata, outvar, expmap = coadd(
            hdr_list, data_list, var_list, exp_list,
            method=method, weighted=weighted,
            robust=robust, sigma=sigma, maxiters=maxiters,
            smoothing=smoothing, reference=reference)

        # output header
        outhdr = hdmerge(hdr_list, reference_header=outhdr)
        extwcs = WCS(outhdr).to_header(relax=True)

        # add some history messages for combination parameters
        hdinsert(outhdr, 'HISTORY',
                 'Coadd: Method: {}'.format(method))
        if method == 'mean':
            hdinsert(outhdr, 'HISTORY',
                     'Coadd: Weighted: {}'.format(weighted))
        hdinsert(outhdr, 'HISTORY',
                 'Coadd: Robust: {}'.format(robust))
        if robust:
            hdinsert(outhdr, 'HISTORY',
                     'Coadd: Threshold: {}'.format(sigma))
            hdinsert(outhdr, 'HISTORY',
                     'Coadd: Max. Iters: {}'.format(maxiters))

        # update integration time from map
        exptime = np.nanmax(expmap)
        hdinsert(outhdr, 'EXPTIME', exptime,
                 comment='Nominal on-source integration time [s]')

        # re-run photometry for standards
        # use the basehead calibration config
        if self.calres['obstype'] == 'STANDARD_FLUX':
            log.info('')
            try:
                run_photometry(outdata, outhdr, outvar, self.cal_conf,
                               allow_badfit=True)
            except PipeCalError:  # pragma: no cover
                log.warning('Photometry failed.')
            log.info('')

        # check if data should be marked as a level 4 mosaic
        # (i.e. input is already calibrated)
        procstat = getpar(outhdr, 'PROCSTAT',
                          default='UNKNOWN', dtype=str)
        procstat = procstat.strip().upper()
        if procstat == 'LEVEL_3':
            hdinsert(outhdr, 'PROCSTAT', 'LEVEL_4',
                     comment='Processing status')
            ptype = 'mosaic'
        else:
            ptype = self.prodtypes[self.step_index]

        # store output data
        hdul = fits.HDUList()
        # copy only the expected extensions
        expected = ['FLUX', 'ERROR', 'EXPOSURE']
        for hdu in self.input[0]:
            if hdu.header.get('EXTNAME', 'UNKNOWN').upper() in expected:
                hdul.append(hdu)
        hdul[0].header = outhdr
        hdul['FLUX'].data = outdata

        hdinsert(extwcs, 'BUNIT', outhdr.get('BUNIT', 'UNKNOWN'),
                 comment='Data units')
        hdul['ERROR'] = fits.ImageHDU(data=np.sqrt(outvar), header=extwcs,
                                      name='ERROR')
        hdinsert(extwcs, 'BUNIT', 's', comment='Data units')
        hdul['EXPOSURE'] = fits.ImageHDU(data=expmap, header=extwcs,
                                         name='EXPOSURE')

        outname = self.update_output(hdul, self.filenum, ptype)
        self.input = [hdul]
        self.filenum = [self._catfilenum(self.filenum)]

        # save if desired
        if param.get_value('save'):
            outname = self.write_output(hdul, outname)
            self.set_display_data(filenames=[outname])
        else:
            self.set_display_data()

    def fluxcal(self):
        """
        Calibrate flux to physical units.

        Calls `sofia_redux.calibration.pipecal_util.apply_fluxcal`.  For
        standards, photometry may optionally be re-run,
        using `sofia_redux.calibration.pipecal_util.run_photometry`.  The
        pipecal config is determined individually for each
        file, so that different calibration factors may be
        applied to each file if necessary.
        """
        from sofia_redux.calibration.pipecal_util \
            import apply_fluxcal, run_photometry
        from sofia_redux.calibration.pipecal_config import pipecal_config

        # get parameters
        param = self.get_parameter_set()
        do_phot = param.get_value('rerun_phot')
        srcpos = param.get_value('srcpos')
        fitsize = param.get_value('fitsize')
        fwhm = param.get_value('fwhm')
        profile = param.get_value('profile')

        # args for photometry
        kwargs = {'fitsize': fitsize,
                  'fwhm': fwhm,
                  'profile': profile}

        # read srcpos parameter
        if do_phot and str(srcpos).strip().lower() not in ['', 'none']:
            try:
                cx, cy = [float(n) for n in srcpos.split(',')]
                kwargs['srcpos'] = [cx, cy]
            except (ValueError, IndexError):
                msg = "Invalid source position specified."
                log.error(msg)
                self.error = msg
                return

        outdata = []
        if param.get_value('save'):
            display_files = []
        else:
            display_files = None
        for i, hdul in enumerate(self.input):
            header = hdul[0].header
            data = hdul['FLUX'].data
            err = hdul['ERROR'].data
            variance = err**2

            # allow config to vary
            config = pipecal_config(header)

            # re-run photometry first if desired
            if do_phot and self.calres['obstype'] == 'STANDARD_FLUX':
                run_photometry(data, header, variance, config,
                               **kwargs)

            # use pipecal to calibrate
            outimg, outvar = apply_fluxcal(
                data, header, config, variance=variance,
                write_history=True)

            # set beam keywords in header
            cdelt = np.abs(header.get('CDELT1', 0.768 / 3600))
            beam = config.get('fwhm', 5.0) * cdelt
            hdinsert(header, 'BMAJ', beam,
                     comment='Beam major axis (deg)')
            hdinsert(header, 'BMIN', beam,
                     comment='Beam minor axis (deg)')
            hdinsert(header, 'BPA', 0.0,
                     comment='Beam angle (deg)')

            # if data is not calibrated (cal factor could not be found),
            # then continue
            if '3' not in header['PROCSTAT']:
                outdata.append(hdul)
                continue

            # print source flux after calibration
            if self.calres['obstype'] == 'STANDARD_FLUX':
                try:
                    flux = header['STAPFLX'] / header['CALFCTR']
                    flux_err = header['STAPFLXE'] / header['CALFCTR']
                    log.info('')
                    log.info('After calibration:')
                    log.info('Source Flux: '
                             '{:.2f} +/- {:.2f} Jy'.format(flux, flux_err))
                except (KeyError, ValueError):
                    pass
                else:
                    try:
                        modlflx = header['MODLFLX']
                        modlflxe = header['MODLFLXE']
                        log.info(
                            'Model Flux: '
                            '{:.3f} +/- {:.3f} Jy'.format(modlflx, modlflxe))
                        log.info(
                            'Percent difference from model: '
                            '{:.1f}%'.format(100 * (flux - modlflx) / modlflx))
                    except KeyError:
                        pass
                    log.info('')

            # update hdul with result
            hdul[0].header = header
            hdul['FLUX'].data = outimg

            hdinsert(hdul[1].header, 'BUNIT',
                     header.get('BUNIT', 'UNKNOWN'),
                     comment='Data units')
            hdul[1].data = np.sqrt(outvar)

            outname = self.update_output(
                hdul, self.filenum[i], self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                outname = self.write_output(hdul, outname)
                display_files.append(outname)

            outdata.append(hdul)

        self.input = outdata
        self.set_display_data(filenames=display_files)

    def imgmap(self):
        """
        Generate a quick-look image map.

        Calls `sofia_redux.visualization.quicklook.make_image`.

        The output from this step is identical to the input, so is
        not saved.  As a side effect, a PNG file is saved to disk to the
        same base name as the input file, with a '.png' extension.
        """
        from sofia_redux.visualization.quicklook import make_image

        # get parameters
        param = self.get_parameter_set()
        colormap = param.get_value('colormap')
        scale = param.get_value('scale')
        n_contour = param.get_value('n_contour')
        contour_color = param.get_value('contour_color')
        fill_contours = param.get_value('fill_contours')
        grid = param.get_value('grid')
        beam = param.get_value('beam')
        watermark = param.get_value('watermark')
        crop_border = param.get_value('crop_border')

        for i, hdul in enumerate(self.input):
            header = hdul[0].header

            # set text for title and subtitle
            obj = header.get('OBJECT', 'UNKNOWN')
            spectel = self.calres['spectel']
            basename = os.path.basename(header.get('FILENAME', 'UNKNOWN'))
            title = f'Object: {obj}, Filter: {spectel}'
            subtitle = f'Filename: {basename}'

            # add a default beam to the header if standard keywords
            # are not present
            if beam and 'BMAJ' not in header:
                cdelt = np.abs(header.get('CDELT1', 0.768 / 3600))
                beam = 5.0 * cdelt
                hdinsert(header, 'BMAJ', beam,
                         comment='Beam major axis (deg)')
                hdinsert(header, 'BMIN', beam,
                         comment='Beam minor axis (deg)')
                hdinsert(header, 'BPA', 0.0,
                         comment='Beam angle (deg)')

            # set crop to trim NaN border if needed
            crop_region = None
            if crop_border:
                img = hdul['FLUX'].data
                badmask = np.isnan(img)
                badmask[img == 0] = True
                strip_rows = np.all(badmask, axis=1)
                strip_cols = np.all(badmask, axis=0)
                xl = np.argmax(~strip_cols)
                xu = np.argmax(np.flip(~strip_cols))
                yl = np.argmax(~strip_rows)
                yu = np.argmax(np.flip(~strip_rows))
                xu, yu = len(strip_cols) - xu, len(strip_rows) - yu
                crop_region = [(xu - xl) / 2 + xl, (yu - yl) / 2 + yl,
                               (xu - xl) / 2, (yu - yl) / 2]

            # make the image figure
            fig = make_image(hdul, colormap=colormap,
                             scale=scale, n_contour=n_contour,
                             contour_color=contour_color,
                             fill_contours=fill_contours, title=title,
                             subtitle=subtitle, grid=grid, beam=beam,
                             watermark=watermark, crop_region=crop_region,
                             crop_unit='pixel')

            # output filename for image
            fname = os.path.splitext(basename)[0] + '.png'
            outname = os.path.join(self.output_directory, fname)

            fig.savefig(outname, dpi=300)
            log.info(f'Saved image to {outname}')
