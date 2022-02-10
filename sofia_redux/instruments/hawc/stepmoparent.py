# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Pipeline step that processes multiple inputs and
produces multiple outputs."""

from astropy import log

from sofia_redux.instruments.hawc.dataparent import DataParent
from sofia_redux.instruments.hawc.stepmiparent import StepMIParent

__all__ = ['StepMOParent']


class StepMOParent(StepMIParent):
    """
    Pipeline step parent class for multiple output files.

    This class defines a pipeline step that expects a list of
    input data objects, and produces a list of output data object
    (multiple-in, multiple-out (MIMO) mode).
    """
    def __init__(self):
        # call superclass constructor (calls setup)
        super().__init__()

        # Change dataout
        self.dataout = [DataParent()]

        # set iomode
        self.iomode = 'MIMO'

    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        See `sofia_redux.instruments.hawc.stepparent.StepParent.setup`
        for more information.
        """
        # Name of the pipeline reduction step
        self.name = 'parentmo'
        self.description = 'Multi-out Step Parent'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'unk'

        # Clear Parameter list
        self.paramlist = []

    def run(self):
        """Run the data reduction algorithm."""
        # Return the first datain element
        self.dataout = self.datain

    def runend(self, data):
        """
        Clean up after a pipeline step.

        This method should be called after calling self.run.

        Sends a final log message, updates the header in
        each data object in the output list, and clears
        input parameter arguments.

        Parameters
        ----------
        data : list of DataFits or DataText
            Output data to update.
        """
        # update header (status and history)
        for d in data:
            self.updateheader(d)

        # clear input arguments
        self.arglist = {}
        log.info('Finished Reduction: Pipe Step %s' % self.name)
