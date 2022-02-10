# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Scan flat field generation pipeline step."""

import os
import warnings

from astropy import log

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.stepparent import StepParent
from sofia_redux.instruments.hawc.steps.stepscanmap import StepScanMap
from sofia_redux.scan.reduction.reduction import Reduction


__all__ = ['StepScanMapFlat']


class StepScanMapFlat(StepParent):
    """
    Generate a flat field from scanning data.

    This step calls scan map with special parameters to produce a flat
    field from a scanning observation of a bright source. Additional
    options for the scan map algorithm may be passed in the
    'options' parameter.

    Input data for this step is raw HAWC data FITS files.

    Output from this step is a single output DataFits,
    with 4 image planes (HDUs): R ARRAY GAIN, T ARRAY GAIN, R BAD PIXEL MASK,
    and T BAD PIXEL MASK.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'scanmapflat', and are named with
        the step abbreviation 'SFL'.

        Parameters defined for this step are:

        options : str
            Command-line options to pass to scan map.
        use_frames : str
            Frames to use from the reduction. Specify a particular
            range, as '400:-400' or '400:1000'.
        """
        # Name of the pipeline reduction step
        self.name = 'scanmapflat'
        self.description = 'Make Scan Flat'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'sfl'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['scanmappath', '.',
                               'Path to the scan map installation'])
        self.paramlist.append(['options', '',
                               'Command line options for scan map'])
        self.paramlist.append(['use_frames', '',
                               "Frames to use from reduction. "
                               "Specify a particular range, as "
                               "'400:-400', or '400:1000'."])

    # The following comment is for PyCharm:
    #   allow scanmap command list to be assembled
    #   without complaining -- it's more readable that way.
    #
    # noinspection PyListCreation
    def run(self):
        """
        Run the data reduction algorithm.

        This step is run as a single-in single-out (SISO) step:
        self.datain should be a DataFits, and output
        will also be a DataFits, stored in self.dataout.

        The process is:

        1. Assemble the scan map command from input parameters.
        2. Call scan map as a subprocess.
        3. Read scan map output from disk and update headers as
           necessary.

        """
        # collect input options in dict
        kwargs = {}
        options = {}

        outpath = os.path.dirname(self.datain.filename)
        kwargs['outpath'] = outpath

        # output basename
        outname = os.path.basename(self.datain.filenamebegin
                                   + self.procname.upper()
                                   + self.datain.filenameend)
        kwargs['name'] = outname

        # Add all subarrays
        kwargs['subarray'] = 'R0,T0,R1'
        kwargs['lock'] = 'subarray'

        # Add frame clipping if necessary
        use_frames = str(self.getarg('use_frames')).strip()
        use_frames = StepScanMap.check_use_frames([self.datain], use_frames)
        if use_frames != '':
            kwargs['frames'] = use_frames

        # Add flat field and output options
        kwargs['source'] = {'flatfield': True}
        kwargs['write'] = {'source': False,
                           'flatfield': {'value': True,
                                         'name': outname}}
        flatname = os.path.join(outpath, outname)

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

        # get input file name
        if os.path.exists(self.datain.filename):
            infile = self.datain.filename
        else:
            rawname = self.datain.rawname
            if os.path.exists(rawname):
                infile = rawname

        # log input
        log.debug(f'All provided options: {kwargs}')
        log.debug(f'Input file: {infile}')

        # run the reduction
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            reduction = Reduction('hawc_plus')
            reduction.run(infile, **kwargs)

        # check for appropriate output
        if not os.path.isfile(flatname):
            log.error(f"Unable to open scan map output "
                      f"file = {outname}")
            raise ValueError("No scan map output found.")

        # load flat output file
        df = DataFits(config=self.config)
        df.load(flatname)
        os.remove(flatname)
        df.filename = self.datain.filename

        # Store the output in dataout
        self.dataout = df

        # Copy and save headers
        scnhead = self.dataout.header.copy()
        self.dataout.header = self.datain.header.copy()
        StepScanMap.merge_scan_hdr(self.dataout, scnhead, [self.datain])

        # Update SOFIA mandated keywords (since this is first pipe step)
        obsid = 'P_' + self.datain.getheadval('OBS_ID')
        self.dataout.setheadval('OBS_ID', obsid)
        self.dataout.setheadval('PIPELINE', 'HAWC_DRP')
