# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Pipeline step that processes multiple input objects."""

import os
import re

from astropy import log

from sofia_redux.instruments.hawc.dataparent import DataParent
from sofia_redux.instruments.hawc.stepparent import StepParent

__all__ = ['StepMIParent']


class StepMIParent(StepParent):
    """
    Pipeline step parent class for multiple input files.

    This class defines a pipeline step that expects a list of
    input data objects, and produces a single output data object
    (multiple-in, single-out (MISO) mode).
    """
    def __init__(self):
        # call superclass constructor (calls setup)
        super().__init__()

        # Change datain
        self.datain = [DataParent()]

        # set iomode
        self.iomode = 'MISO'

        # add a filenum list, for output filenames
        self.nfiles = 0
        self.filenum = []

    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        See `sofia_redux.instruments.hawc.stepparent.StepParent.setup`
        for more information.
        """
        # Name of the pipeline reduction step
        self.name = 'parentmi'
        self.description = 'Multi-in Step Parent'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'unk'

        # Clear Parameter list
        self.paramlist = []

    def run(self):
        """Run the data reduction algorithm."""
        # Return the first datain element
        self.dataout = self.datain[0]

    def runstart(self, data, arglist):
        """
        Initialize the pipeline step.

        Checks the input data list, storing file numbers extracted
        from the input file names, then calls the parent runstart
        method.

        Parameters
        ----------
        data : DataFits or DataText
            Input data to validate.
        arglist : dict
            Parameters to pass to the step.
        """
        # Keep a list of input file numbers for output filename
        self.filenum = []

        # Check input data - should be a list/tuple with PipeData objects
        if isinstance(data, (list, tuple)):
            for d in data:
                if not isinstance(d, DataParent):
                    msg = 'Invalid input data type: Pipe Data ' \
                          'object is required'
                    log.error(msg)
                    raise TypeError('Runstart: ' + msg)
                # try to read numerical file number from input name
                if '-' in str(d.filenum):
                    fn = str(d.filenum).split('-')
                else:
                    fn = [d.filenum]
                for f in fn:
                    try:
                        # test if it is a valid number
                        int(f)
                        # append the string version if it is
                        self.filenum.append(f)
                    except (ValueError, TypeError):
                        pass
        else:
            msg = 'Invalid input data type: List object is required'
            log.error(msg)
            raise TypeError('Runstart: ' + msg)

        # Call parent runstart
        super().runstart(data[0], arglist)

    def updateheader(self, data):
        """
        Update the header for a data object.

        This function calls the parent updateheader method, then
        additionally updates the output file name with the input
        file numbers, stored in self.filenum.

        Parameters
        ----------
        data : DataFits or DataText
            Output data to update.
        """
        # Call parent updateheader
        super().updateheader(data)

        # Update file name with PipeStepName and input filenumbers
        # if available and MISO. Otherwise, use the version set by the parent.
        if self.iomode == 'MISO' and len(self.filenum) > 1:
            fn = sorted(self.filenum)
            filenums = fn[0] + '-' + fn[-1]
            outdir, basename = os.path.split(data.filename)
            match = re.search(self.config['data']['filenum'], basename)
            if match is not None:
                # regex may contain multiple possible matches --
                # for middle or end of filename
                for i, g in enumerate(match.groups()):
                    if g is not None:
                        fbegin = basename[:match.start(i + 1)]
                        fend = basename[match.end(i + 1):]
                        data.filename = os.path.join(outdir,
                                                     fbegin + filenums + fend)
                        break
