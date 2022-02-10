# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Scan image reconstruction pipeline step."""

import os
import warnings

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.stepmiparent import StepMIParent
from sofia_redux.instruments.hawc.stepmoparent import StepMOParent
from sofia_redux.scan.reduction.reduction import Reduction
from sofia_redux.toolkit.utilities.fits import hdinsert


__all__ = ['StepScanMap']


class StepScanMap(StepMOParent):
    """
    Reconstruct an image from scanning data.

    Input data for this step are raw HAWC data FITS files.

    Output from this step is typically a single output DataFits,
    with 4 image planes (HDUs): SIGNAL, EXPOSURE, NOISE and S/N.
    Additionally, binary table HDUs (index 4+) are added for each input scan
    (controlled by the write.scandata option), which contains original
    header information for each scan, per-scan reduction summary data,
    and a binary table of scan data (e.g. pixel gains, weights,
    per-pixel spectral filter profile, pixel noise spectra etc.).
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'scanmap', and are named with
        the step abbreviation 'SMP'.

        Parameters defined for this step are:

        noout : bool
            If set, FITS output is not expected and will not be
            loaded. This may be set for off-nominal reductions, like
            skydip observations.
        subarray : str
            Specify the subarrays to use in the reduction. Default is
            all available ('R0,T0,R1').
        use_frames : str
            Frames to use from the reduction. Specify a particular
            range, as '400:-400' or '400:1000'
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
        self.name = 'scanmap'
        self.description = 'Construct Scan Map'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'smp'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['noout', False,
                               'No FITS output is expected'])
        self.paramlist.append(['subarray', 'R0,T0,R1',
                               "Subarrays to use in reduction "
                               "('' for default)"])
        self.paramlist.append(['use_frames', '',
                               "Frames to use from reduction. "
                               "Specify a particular range, as "
                               "'400:-400', or '400:1000'."])
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

    @staticmethod
    def merge_scan_hdr(df, scnhead, datain):
        """
        Merge headers from scan output files.

        The header in the `df` is updated in place.

        Parameters
        ----------
        df : DataFits
            The output data to update.
        scnhead : fits.Header
            The header from the scan output file.
        datain : list of DataFits
            The list of input data to merge into the output data.
        """
        # Get card lists and add scan keywords
        othercards = scnhead.cards
        for cardi, card in enumerate(othercards):
            kwd = card.keyword

            # skip comments
            if kwd == 'COMMENT':
                continue

            # otherwise add key normally
            try:
                if len(card.comment) > 0:
                    df.setheadval(kwd, card.value, card.comment)
                else:
                    df.setheadval(kwd, card.value)
            except (KeyError, ValueError, TypeError):
                log.warning("Unable to add FITS keyword %s=%s" %
                            (kwd, card.value))

        # set ASSC_AOR and ASSC_MSN value in input header, before merging
        try:
            df.setheadval('ASSC_AOR',
                          df.getheadval('AOR_ID'),
                          'Associated AORs')
        except KeyError:
            pass
        try:
            df.setheadval('ASSC_MSN',
                          df.getheadval('MISSN-ID'),
                          'Associated Mission IDs')
        except KeyError:
            pass

        # update header keywords from datain list
        for i, df_in in enumerate(datain):
            try:
                df_in.setheadval('ASSC_AOR', df_in.getheadval('AOR_ID'),
                                 'Associated AORs')
            except KeyError:
                pass
            try:
                df_in.setheadval('ASSC_MSN', df_in.getheadval('MISSN-ID'),
                                 'Associated Mission IDs')
            except KeyError:
                pass

            # remove any previous history for less clutter
            df_in.delheadval('HISTORY')

            df.mergehead(df_in)

        # Remove an engineering keyword if necessary
        df.delheadval('XPADDING')

    @staticmethod
    def check_use_frames(datain, use_frames):
        if use_frames == '':
            return use_frames
        try:
            nframes = np.array([float(df.header.get('EXPTIME'))
                                * float(df.header.get('SMPLFREQ'))
                                for df in datain])
        except (ValueError, TypeError):
            log.warning('Could not read EXPTIME or SMPLFREQ from header.')
            return use_frames
        else:
            try:
                ranges = use_frames.split(',')
                for r in ranges:
                    s = r.split(':')
                    if len(s) > 1:
                        if s[0].strip() not in ['', '*']:
                            nframes -= abs(int(s[0]))
                        if s[1].strip() not in ['', '*']:
                            nframes -= abs(int(s[1]))
                    elif s[0].strip() not in ['', '*']:
                        nframes -= abs(int(s[0]))
                if np.any(nframes < 10):
                    log.warning('Frame range parameter is out of '
                                'bounds for this scan set. Turning off '
                                'frame clipping.')
                    use_frames = ''
            except (ValueError, TypeError):
                log.warning('Bad use_frames parameter. Turning off '
                            'frame clipping.')
                use_frames = ''
        return use_frames

    def run(self):
        """
        Run the data reduction algorithm.

        This step is run as a multi-in multi-out (MIMO) step:
        self.datain should be a list of DataFits, and output
        will also be a list of DataFits, stored in self.dataout.

        The process is:

        1. Assemble the scan map options from input parameters.
        2. Call the iterative scan map reconstructor.
        3. Retrieve output and update headers as necessary.

        """
        # collect input options in dict
        kwargs = {'write': {'source': False}}
        options = {}

        # output path
        outpath = os.path.dirname(self.datain[0].filename)
        kwargs['outpath'] = outpath

        # output basename
        outname = os.path.basename(self.datain[0].filenamebegin
                                   + self.procname.upper()
                                   + self.datain[0].filenameend)
        kwargs['name'] = outname

        # get options
        subarr = str(self.getarg('subarray')).strip()
        if subarr != '':
            kwargs['subarray'] = subarr
            kwargs['lock'] = 'subarray'

        # add additional top-level parameters
        for arg in ['deep', 'faint', 'extended']:
            if self.getarg(arg):
                kwargs[arg] = True

        # add frame clipping if necessary
        use_frames = str(self.getarg('use_frames')).strip()
        use_frames = self.check_use_frames(self.datain, use_frames)
        if use_frames != '':
            kwargs['frames'] = use_frames

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

        # If no output requested
        if self.getarg('noout'):
            log.debug('Not loading output - copy from input')

            # copy input
            self.dataout = self.datain

            # SOFIA keywords are not updated since this output is not
            # intended to be saved
            return

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
                log.error('Unexpected output for scan imaging mode. '
                          'Check INSTCFG.')
                raise ValueError("Expected output not found.")

        # Copy and save headers
        scnhead = self.dataout.header.copy()
        self.dataout.header = self.datain[0].header.copy()
        self.merge_scan_hdr(self.dataout, scnhead, self.datain)

        # Update SOFIA mandated keywords (since this is first pipe step)
        obsid = 'P_' + self.datain[0].getheadval('OBS_ID')
        self.dataout.setheadval('OBS_ID', obsid)
        self.dataout.setheadval('PIPELINE', 'HAWC_DRP')

        # Set BUNIT correctly in S/N and exposure extensions

        if 'S/N' in self.dataout.imgnames and \
                'EXPOSURE' in self.dataout.imgnames:
            self.dataout.setheadval('BUNIT', '',
                                    comment='Data units',
                                    dataname='S/N')
            self.dataout.setheadval('BUNIT', 's',
                                    comment='Data units',
                                    dataname='EXPOSURE')

        # Set output list for MIMO compatibility
        self.dataout = [self.dataout]

    def runend(self, data):
        """
        Clean up after a pipeline step.

        Override the default method to call MISO-style header
        updating, in the case where a single file is returned.

        Parameters
        ----------
        data : list of DataFits or DataText
            Output data to update.
        """
        # if one output file, use the MISO update header;
        # if multiple, use MIMO
        if len(data) == 1:
            log.debug('Updating headers for single-output format')
            self.iomode = 'MISO'
            StepMIParent.runend(self, data[0])
        else:
            log.debug('Updating headers for multiple-output format')
            self.iomode = 'MIMO'
            StepMOParent.runend(self, data)
