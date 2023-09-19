# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Scanning polarimetry image reconstruction pipeline step."""

import os
import warnings

from astropy import log
from astropy.io import fits

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.stepmiparent import StepMIParent
from sofia_redux.instruments.hawc.steps.stepscanmap import StepScanMap
from sofia_redux.scan.reduction.reduction import Reduction

__all__ = ['StepScanPolMerge']


class StepScanPolMerge(StepMIParent):
    """
    Reconstruct Stokes images from scanning polarimetry data.

    This step requires that scanning polarimetry data are taken with
    four HWP angles, one per input file.

    Output from this step is a DataFits with the following image
    extensions: STOKES I, ERROR I, STOKES Q, ERROR Q, STOKES U,
    ERROR U, COVAR Q I, COVAR U I, COVAR Q U, BAD PIXEL MASK.
    Additionally, binary table HDUs (index 10+) are added for each
    input scan, which contain original header information for each scan,
    per-scan reduction summary data, and a binary table of scan data
    (e.g. pixel gains, weights, per-pixel spectral filter profile, pixel
    noise spectra etc.).

    Output stokes maps are corrected for instrumental polarization
    and rotated by sky angle, prior to being merged.

    NOTE:
    In HAWC DRP v3.3.0, the functionality in this step was created to
    replace the existing scanmappol, scanstokes, ip, and rotate steps
    with a more direct reconstruction of the Stokes I, Q, and U maps.
    This older steps are preserved for comparison and default reductions.
    This mode may be invoked via the 'direct_scanpol' pipeline mode,
    defined in the default pipeline configuration file
    (hawc/data/config/pipeconf.cfg). The older scanpol method is left
    as the default reduction mode; this method is considered experimental
    and requires further testing before use.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'scanpolmerge', and are named with
        the step abbreviation 'SPR'.

        Parameters defined for this step are:

        use_frames : str
            Frames to use from the reduction. Specify a particular
            range, as '400:-400', or '400:1000'.
        grid : float
            Output pixel scale.  If not set, default values from scan map
            configuration will be used.
        deep : bool
            If set, faint point-like emission is prioritized.
        faint : bool
            If set, faint emission (point-like or extended) is prioritized.
        extended : bool
            If set, extended emission is prioritized.
            This may increase noise on large angular scales.
        options : str
            Additional options to pass to the scan map algorithm.
        """
        # Name of the pipeline reduction step
        self.name = 'scanpolmerge'
        self.description = 'Construct Stokes Maps'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'spr'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['save_intermediate', False,
                               'Save individual scanmap frames'])
        self.paramlist.append(['use_frames', '',
                               "Frames to use from reduction. "
                               "Specify a particular range, as "
                               "'400:-400', or '400:1000'."])
        self.paramlist.append(['grid', '',
                               "Output pixel scale, if not default. "
                               "Specify in arcsec."])
        self.paramlist.append(['deep', False,
                               'Attempt to recover faint point-like '
                               'emission'])
        self.paramlist.append(['faint', False,
                               'Attempt to recover faint emission '
                               '(point-like or extended)'])
        self.paramlist.append(['extended', False,
                               'Attempt to recover extended emission '
                               '(may increase noise)'])
        self.paramlist.append(['options', '',
                               'Additional options for scan reconstruction'])

    def run(self):
        """
        Run the data reduction algorithm.

        This step is run as a multi-in single-out (MISO) step:
        self.datain should be a list of DataFits, and output
        is a single DataFits, stored in self.dataout.

        The process is:

        1. Assemble the scan map options from input parameters.
        2. Call the iterative scan map reconstructor.
           It will create Stokes I, Q, and U maps, corrected for
           instrumental polarization and rotated by the sky angle.

        """
        # collect input options in dict
        kwargs = {}
        options = {}
        if not self.getarg('save_intermediate'):
            kwargs['write'] = {'source': False}

        # output path
        outpath = os.path.dirname(self.datain[0].filename)
        kwargs['outpath'] = outpath

        # specify polmap
        kwargs['polmap'] = True
        kwargs['scanpol'] = True

        # add additional top-level parameters
        for arg in ['deep', 'faint', 'extended']:
            if self.getarg(arg):
                kwargs[arg] = True

        # add frame clipping if necessary
        use_frames = str(self.getarg('use_frames')).strip()
        use_frames = StepScanMap.check_use_frames(self.datain, use_frames)
        if use_frames != '':
            kwargs['frames'] = use_frames

        # set the output pixel scale if supplied
        try:
            grid = float(str(self.getarg('grid')).strip())
        except (ValueError, TypeError):
            grid = None
        if grid is not None:
            kwargs['grid'] = grid

        # add additional options from parameters at end,
        # so they can override any defaults set by the above
        additional = str(self.getarg('options')).strip()
        if additional != '':
            all_val = additional.split()
            for val in all_val:
                try:
                    k, v = val.split('=')
                except (IndexError, ValueError, TypeError):
                    pass
                else:
                    options[k] = v
        kwargs['options'] = options

        # get input file names
        infiles = []
        for datain in self.datain:
            if os.path.exists(datain.filename):
                infiles.append(datain.filename)
            else:
                rawname = datain.rawname
                if os.path.exists(rawname):
                    infiles.append(rawname)

        # log input
        log.debug(f'All provided options: {kwargs}')
        log.debug(f'Input files: {infiles}')

        # run the reduction
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            reduction = Reduction('hawc_plus')
            output_hdul = reduction.run(infiles, **kwargs)
        log.info('')

        # read output file(s)
        if output_hdul is not None and isinstance(output_hdul, fits.HDUList):
            # single output file
            df = DataFits(config=self.config)
            df.filename = self.datain[-1].filename
            df.load(hdul=output_hdul)

            # store the output in dataout
            self.dataout = df
        else:
            if output_hdul is None:
                log.error('No output created.')
                raise ValueError('No output created.')
            else:
                log.error('Unexpected output for scan pol mode. '
                          'Check INSTCFG.')
                raise ValueError("Expected output not found.")

        # Check for Stokes extensions
        if 'STOKES Q' not in self.dataout.imgnames:
            log.warning('Found 1 HWP; processing as imaging data.')
            self.dataout.header['NHWP'] = 1

        # Copy and save headers
        scnhead = self.dataout.header.copy()
        self.dataout.header = self.datain[0].header.copy()
        StepScanMap.merge_scan_hdr(self.dataout, scnhead, self.datain)

        # Update SOFIA mandated keywords (since this is first pipe step)
        obsid = 'P_' + self.datain[0].getheadval('OBS_ID')
        self.dataout.setheadval('OBS_ID', obsid)
        self.dataout.setheadval('PIPELINE', 'HAWC_DRP')

    def runend(self, data):
        """
        Clean up after a pipeline step.

        Override the default method to use a different product name/type
        for single HWP files.

        Parameters
        ----------
        data : DataFits
            Output data to update.
        """
        try:
            if data.header.get('NHWP', 1) == 1:
                self.name = 'scanmap'
                self.procname = 'smp'
            StepMIParent.runend(self, data)
        finally:
            self.name = 'scanpolmerge'
            self.procname = 'spr'
