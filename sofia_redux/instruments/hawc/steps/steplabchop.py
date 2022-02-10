# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Diagnostic lab chop pipeline step."""

import numpy as np

from sofia_redux.instruments.hawc.stepparent import StepParent

__all__ = ['StepLabChop']


class StepLabChop(StepParent):
    """
    Produce diagnostic data for lab chopping.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'labchop', and are named with
        the step abbreviation 'RED'.

        There are currently no parameters defined for this step.
        """
        # Name of the pipeline reduction step
        self.name = 'labchop'
        self.description = "Reduce Lab Chops"

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'red'

        # Clear Parameter list
        self.paramlist = []

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object.  The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Calculate median modulus and phase
        2. Save to output
        """
        # copy input to output
        self.dataout = self.datain.copy()

        # R array, T array, R array Imag and T array Imag medians
        rphase = np.median(self.datain.table['R array'], axis=0)
        rquad = np.median(self.datain.table['R array Imag'], axis=0)
        tphase = np.median(self.datain.table['T array'], axis=0)
        tquad = np.median(self.datain.table['T array Imag'], axis=0)

        rown = rphase.shape[0]
        coln = rphase.shape[1] + tphase.shape[1]
        allphase = np.zeros((rown, coln))
        allphase[:, :rphase.shape[1]] = rphase
        allphase[:, rphase.shape[1]:] = tphase
        allquad = np.zeros((rown, coln))
        allquad[:, :rphase.shape[1]] = rquad
        allquad[:, rphase.shape[1]:] = tquad

        # Modulus and angle for all pixels
        allmodulus = np.sqrt(allphase * allphase + allquad * allquad)
        allangle = np.arctan2(allquad, allphase) / np.pi * 180.

        # If no data beyond column 96 cut it
        if coln > 96:
            if allmodulus[:, 96:].max() == 0:
                allmodulus = allmodulus[:, :96]
                allangle = allangle[:, :96]

        # Remove the instrumental configuration HDU
        if 'CONFIGURATION' in self.dataout.imgnames:
            self.dataout.imagedel('CONFIGURATION')

        # Add images to output file
        self.dataout.imageset(allmodulus, 'Modulus')
        allangle[allangle < 0.0] += 360.
        self.dataout.imageset(allangle, 'Phase')
