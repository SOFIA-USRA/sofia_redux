# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Flux jump correction pipeline step."""

import os

from astropy import log
import numpy as np

from sofia_redux.instruments.hawc.stepparent import StepParent
from sofia_redux.instruments.hawc.datafits import DataFits

__all__ = ['StepFluxjump']


class StepFluxjump(StepParent):
    """
    Correct for flux jumps in raw data.

    This pipeline step corrects for a detector effect that introduces
    discontinuous changes in flux values (flux jumps). Jumps are
    detected in the data, then a jump map is used to shift all data
    following the jump to correct values.

    Input to this step is raw HAWC data files. This step should
    be called before `sofia_redux.instruments.hawc.steps.StepPrepare`.
    It uses the 'SQ1Feedback' and 'FluxJumps' columns in the data table.
    Output from this step has the same format as the input; only flux
    values in the SQ1FeedbackColumn are modified.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'fluxjump', and are named with
        the step abbreviation 'FJP'.

        Parameters defined for this step are:

        jumpmap : str or float
            Path to a file name, specifying the jump gap map
            in FITS format. Alternatively, a single value can be
            specified to apply to all pixels. If all jump map
            values are zero, no jump correction will be performed.
        """
        # Name of the pipeline reduction step
        self.name = 'fluxjump'
        self.description = 'Fix Flux Jumps'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'fjp'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['jumpmap', '4600.0',
                               'filepathname specifying the jump gap map, '
                               'alternatively a number for the gap '
                               'to be used for all pixels'])

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object. The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Read the jump map from the parameters.
        2. Identify pixel samples with flux jumps
           (FluxJump data at that pixel is < -32 or > 32).
        3. Fix flux jump data for any pixels that wrap around
           from 64 to -64, or from -64 to 64 (compared
           to the previous sample).
        4. Multiply flux jump data by the jump map.
        5. Add the jump data * map to the raw data array.
        6. Store the result in the SQ1Feedback column.
        """
        # Preparation

        # Get raw data size
        detsize = self.datain.table['SQ1Feedback'].shape[1:]

        # Load jump gap map
        jumpmap = os.path.expandvars(str(self.getarg('jumpmap')))
        if os.path.exists(jumpmap):
            # Load pipedata object and get image
            jumpdat = DataFits(jumpmap).image
        else:
            # make it an int
            try:
                jumpmap = int(float(jumpmap))
                jumpdat = np.ones(detsize, dtype=np.int32) * jumpmap
            except (ValueError, TypeError):
                msg = 'Bad fluxjump value: {}'.format(jumpmap)
                log.error(msg)
                raise ValueError
        log.debug('Got fluxjump data = %s' % jumpmap)

        if np.allclose(jumpdat, 0):
            log.info('No correction to apply.')
            self.dataout = self.datain
            return

        # Get Data
        fjdata = self.datain.table['FluxJumps']
        rawdata = self.datain.table['SQ1Feedback']

        # Correct fluxjumps wraparounds

        # select only pixels which have fj values >32 and <-32
        (rowinds, colinds) = np.where((np.amin(fjdata, axis=0) < -32)
                                      & (np.amax(fjdata, axis=0) > 32))
        log.debug('%d pixels need wraparound check' % len(rowinds))

        # Loop through pixels which need check
        for i in range(len(rowinds)):
            ri, ci = rowinds[i], colinds[i]
            # Fix wraparounds in fluxjump trace for that pixel
            for j in range(1, fjdata.shape[0]):
                if fjdata[j - 1, ri, ci] < -32 and fjdata[j, ri, ci] > 32:
                    fjdata[j, ri, ci] -= 128
                if fjdata[j - 1, ri, ci] > 32 and fjdata[j, ri, ci] < -32:
                    fjdata[j, ri, ci] += 128

        # Do flux jump correction

        # Shift fluxjumps by one sample in time
        # (to have one sample delayed response to FJ)
        fjdata[1:, ...] = fjdata[:-1, ...]

        # Multiply with jump map
        fjdata *= jumpdat

        # Correct raw data with flux jumps
        rawdata += fjdata

        # Put data back
        self.dataout = self.datain
        self.dataout.table['SQ1Feedback'] = rawdata
