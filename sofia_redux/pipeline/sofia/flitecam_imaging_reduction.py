# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FLITECAM Imaging Reduction pipeline steps"""

import os

from astropy import log
import numpy as np

from sofia_redux.pipeline.sofia.sofia_exception import SOFIAImportError
try:
    import sofia_redux.instruments.flitecam
    assert sofia_redux.instruments.flitecam
except ImportError:
    raise SOFIAImportError('FLITECAM modules not installed')
try:
    import sofia_redux.instruments.forcast.configuration as dripconfig
except ImportError:
    raise SOFIAImportError('FORCAST modules not installed')

from sofia_redux.pipeline.sofia.flitecam_reduction \
    import FLITECAMReduction
from sofia_redux.pipeline.sofia.parameters.flitecam_imaging_parameters \
    import FLITECAMImagingParameters
from sofia_redux.pipeline.sofia.forcast_imaging_reduction \
    import FORCASTImagingReduction
from sofia_redux.toolkit.utilities.fits import hdinsert, gethdul


class FLITECAMImagingReduction(FLITECAMReduction, FORCASTImagingReduction):
    """
    FLITECAM imaging reduction steps.

    Primary image reduction algorithms are defined in the flitecam
    package (`sofia_redux.instruments.flitecam`).  Calibration-related
    algorithms are pulled from the `sofia_redux.calibration` package, and
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
        self.mode = 'imaging'

        # product type definitions for FLITECAM steps
        self.prodtype_map.update({
            'clip_image': 'clipped',
            'make_flat': 'flat',
            'correct_gain': 'gain_corrected',
            'subtract_sky': 'background_subtracted',
            'register': 'registered',
            'tellcor': 'telluric_corrected',
            'coadd': 'coadded',
            'fluxcal': 'calibrated',
            'imgmap': 'imgmap'
        })
        self.prodnames.update({
            'clipped': 'CLP',
            'flat': 'FLT',
            'gain_corrected': 'GCR',
            'background_subtracted': 'BGS',
            'registered': 'REG',
            'telluric_corrected': 'TEL',
            'coadded': 'COA',
            'calibrated': 'CAL',
            'imgmap': 'IMP',
        })

        # invert the map for quick lookup of step from type
        self.step_map = {v: k for k, v in self.prodtype_map.items()}

        # default recipe and step names
        self.recipe = ['check_header', 'correct_linearity',
                       'clip_image', 'make_flat', 'correct_gain',
                       'subtract_sky', 'register', 'tellcor',
                       'coadd', 'fluxcal', 'imgmap']
        self.processing_steps.update({
            'clip_image': 'Clip Image',
            'make_flat': 'Make Flat',
            'correct_gain': 'Correct Gain',
            'subtract_sky': 'Subtract Sky',
            'register': 'Register Images',
            'tellcor': 'Telluric Correct',
            'coadd': 'Combine Images',
            'fluxcal': 'Flux Calibrate',
            'imgmap': 'Make Image Map',
        })

        # load some default FORCAST configurations, for registration step
        # Actual values will be provided in parameters.
        dripconfig.load()

    def load(self, data, param_class=None):
        """Call parent load, with imaging parameters."""
        FLITECAMReduction.load(self, data,
                               param_class=FLITECAMImagingParameters)

    def clip_image(self):
        """Clip image to useful portion of detector."""
        from sofia_redux.instruments.flitecam.clipimg import clipimg
        from sofia_redux.instruments.flitecam.expmap import expmap
        from sofia_redux.instruments.flitecam.maskbp import maskbp

        # get parameters
        param = self.get_parameter_set()
        datasec = param.get_value('datasec')
        skip_clean = param.get_value('skip_clean')

        outdata = []
        for i, hdul in enumerate(self.input):
            log.info('')
            log.info(f"Input: {hdul[0].header['FILENAME']}")

            # clip data
            result = clipimg(hdul, datasec)

            # mask bad pixels
            if skip_clean:
                log.info('Skipping bad pixel identification')
            else:
                result = maskbp(result, cval=np.nan)

            # append exposure map
            result = expmap(result)

            outname = self.update_output(
                result, self.filenum[i], self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                self.write_output(result, outname)

            outdata.append(result)

        self.input = outdata

        # set display data to input
        self.set_display_data()

    def make_flat(self):
        """
        Make a flat field from input data.

        The procedure is:

        - Check for a previously made flat. If present, use it.
        - If sky files are loaded, use those to make the flat.
          Correct the sky files with others to verify flat.
        - Otherwise, make a flat out of all input except the current
          file. If less than 3 files are loaded, no flat will be
          generated.

        """
        from sofia_redux.instruments.flitecam.mkflat import mkflat
        from sofia_redux.instruments.flitecam.split_input import split_input

        # get parameters
        param = self.get_parameter_set()
        flatfile = param.get_value('flatfile')
        skip_flat = param.get_value('skip_flat')

        # check input options
        if skip_flat:
            log.info('Skipping flat generation. Data will not be '
                     'gain-corrected.')
            return

        # sort data, looking for sky files and data to correct
        manifest = split_input(self.input)

        flat = None
        if os.path.isfile(flatfile):
            flat = gethdul(flatfile)
            if flat is None or 'FLAT' not in flat:
                raise ValueError(f'Bad flat file: {flatfile}')
            log.info(f'Using previously generated flat file {flatfile}.')
        else:
            if len(manifest['sky']) > 0:
                # sky files present, make the flat from them
                flat_input = manifest['sky']
                log.info('Using sky files to make flat:')
                for f in flat_input:
                    log.info(f"  {f[0].header.get('FILENAME', 'UNKNOWN')}")

                flat = mkflat(flat_input)
                log.info('')

        # warn if insufficient data to generate a flat
        if flat is None and len(self.input) < 3:
            log.warning('Too few files to generate flat. '
                        'Data will not be gain corrected.')

        outdata = []
        for i, hdul in enumerate(self.input):
            # if not already generated, make the flat from all input
            # except the current file
            if flat is None:
                # this procedure requires at least 2 files other
                # than the current one
                if len(self.input) < 3:
                    this_flat = None
                else:
                    flat_input = self.input[:i] + self.input[i + 1:]
                    log.info('')
                    log.info(f"Using remaining source files to make flat for "
                             f"{hdul[0].header.get('FILENAME', 'UNKNOWN')}:")
                    for f in flat_input:
                        log.info(f"  {f[0].header.get('FILENAME', 'UNKNOWN')}")
                    this_flat = mkflat(flat_input)
            else:
                this_flat = flat

            # attach flat to hdul
            if this_flat is not None:
                for extname in ['FLAT', 'FLAT_ERROR', 'FLAT_BADMASK']:
                    # allow flat error and badmask not to exist,
                    # in externally provided flats
                    if extname in this_flat:
                        hdul.append(this_flat[extname])

            outdata.append(hdul)

        # loop through output again to update names and headers
        # and save to disk
        log.info('')
        for i, hdul in enumerate(outdata):
            outname = self.update_output(
                hdul, self.filenum[i], self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                self.write_output(hdul, outname)
        self.input = outdata

        # set display data to input
        self.set_display_data()

    def correct_gain(self):
        """Correct gain variations."""
        from sofia_redux.instruments.flitecam.gaincor import gaincor
        param = self.get_parameter_set()
        outdata = []
        for i, hdul in enumerate(self.input):
            if 'FLAT' in hdul:
                # correct data and propagate errors
                hdul = gaincor(hdul)
            else:
                log.warning('No FLAT extension present; not correcting data.')

            outname = self.update_output(
                hdul, self.filenum[i], self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                self.write_output(hdul, outname)

            outdata.append(hdul)

        self.input = outdata

        # set display data to input
        self.set_display_data()

    def subtract_sky(self):
        """Subtract sky background level."""
        from sofia_redux.instruments.flitecam.backsub import backsub
        from sofia_redux.instruments.flitecam.split_input import split_input
        from sofia_redux.instruments.forcast.peakfind import peakfind

        param = self.get_parameter_set()

        skyfile = param.get_value('skyfile')
        skip_sky = param.get_value('skip_sky')
        method = param.get_value('sky_method')

        if 'median' in str(method).lower():
            method = 'median'
        else:
            method = 'flatnorm'

        # check input options
        if skip_sky:
            log.info('Skipping sky subtraction. Data will not be '
                     'background corrected.')
            log.info('Sky files will be propagated to the following steps.')
            return

        sky = None
        if os.path.isfile(skyfile):
            sky = gethdul(skyfile)
            if sky is None or 'FLUX' not in sky:
                raise ValueError(f'Bad sky file: {skyfile}')
            log.info(f'Using previously generated sky file {skyfile}.')

        # get sky files to drop from input, if present
        manifest = split_input(self.input)
        propagate = manifest['object'] + manifest['standard']

        # special case: only sky files were loaded. In that
        # case go ahead and correct them, for a check of the flat.
        if len(propagate) == 0 and len(manifest['sky']) > 0:
            log.warning('Only sky files are present; propagating them '
                        'to following steps.')
            propagate = manifest['sky']
        elif len(manifest['sky']) > 0:
            log.info('Dropping sky files from further propagation.')

        outdata = []
        outfilenum = []
        for i, hdul in enumerate(self.input):
            # skip the file if it's not a science file
            if hdul not in propagate:
                continue

            # keep the file number for future steps
            outfilenum.append(self.filenum[i])

            # now correct the data and propagate errors
            hdul = backsub(hdul, bgfile=sky, method=method)

            # for standards, mark the likely source position
            # near CRPIX1/2
            if hdul in manifest['standard']:
                sx = hdul[0].header['CRPIX1'] - 1
                sy = hdul[0].header['CRPIX2'] - 1
                try:
                    shape = hdul['FLUX'].data.shape
                    wdw = int(np.min([100, sx, sy,
                                      shape[1] - sx,
                                      shape[0] - sy]))
                    xmin, xmax = int(sx - wdw), int(sx + wdw)
                    ymin, ymax = int(sy - wdw), int(sy + wdw)
                    stamp = hdul['FLUX'].data[ymin:ymax, xmin:xmax]

                    peak = peakfind(stamp, npeaks=1, silent=True,
                                    positive=True, coordinates=True,
                                    fwhm=6.0, smooth=False)
                    if peak is not None and len(peak) == 1:
                        px = peak[0][0] + xmin
                        py = peak[0][1] + ymin
                        if (0 < px < shape[1]
                                and 0 < py < shape[0]):
                            sx, sy = px, py

                except (IndexError, ValueError,
                        TypeError, RuntimeError) as err:  # pragma: no cover
                    log.debug(f'Peakfind error: {str(err)}')
                hdinsert(hdul[0].header, 'SRCPOSX', sx,
                         'Source x-position')
                hdinsert(hdul[0].header, 'SRCPOSY', sy,
                         'Source y-position')

            outname = self.update_output(
                hdul, self.filenum[i], self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                self.write_output(hdul, outname)

            outdata.append(hdul)

        self.input = outdata
        self.filenum = outfilenum

        # set display data to input
        self.set_display_data()
