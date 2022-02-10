# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Diagnostic polarization plot pipeline step."""

import os

from astropy import log
import numpy as np
from matplotlib.backends.backend_agg \
    import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


from sofia_redux.instruments.hawc.stepparent import StepParent

__all__ = ['StepLabPolPlots']


class StepLabPolPlots(StepParent):
    """
    Produce diagnostic plots for lab-generated polarization data.

    This step makes a two panel figure showing percentage polarization
    on the left and polarization angle on the right. Histograms are
    shown below.

    This step is intended to be run in place of StepPolMap.  All other
    steps in the standard chop/nod polarimetry pipeline should be run
    prior to this step.  The input data expected is the output of
    the `sofia_redux.instruments.hawc.steps.StepPolVec` pipeline step,
    containing Percent Pol and Pol Angle extensions.  Output data for this
    step is identical to the input data.  PNG plots are produced as a side
    effect and are saved to disk with the same basename and directory
    as the input data.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'labpolplots', and are named with
        the step abbreviation 'PLT'.

        Parameters defined for this step are:

        region : list of int
            Region box given as [xmin, xmax, ymin, ymax].
        polrange : list of float
            Limit to only use percent polarization in this range,
            given as [pol_min, pol_max].
        anglerange : list of float
            Limit to only use polarization angles in this range,
            given as [angle_min, angle_max], in degrees.

        """
        # Name of the pipeline reduction step
        self.name = 'labpolplots'
        self.description = "Make Lab Pol Plots"

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'plt'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['region', [5, 28, 13, 34],
                               'Region box given as xmin,xmax,ymin,ymax'])
        self.paramlist.append(['polrange', [0, 5],
                               'Limit to only use percent pol '
                               'in the given range'])
        self.paramlist.append(['anglerange', [-45, 45],
                               'Limit to only use pol angle in '
                               'the given range'])

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object.  The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Read percent polarization and angle from the input.
        2. Restrict data to range specified by parameters.
        3. Plot the data and save to disk.
        """
        # copy input to output
        self.dataout = self.datain

        # Get input parameters
        reg = list(self.getarg('region'))
        polrange = self.getarg('polrange')
        anglerange = self.getarg('anglerange')

        # Get the data from the input file
        pol = self.datain.imageget('Percent Pol')
        theta = self.datain.imageget('Pol Angle')

        # Extract and calculate data statistics based on the region
        polstats = pol[reg[0] - 1:reg[1], reg[2] - 1:reg[3]]
        thetastats = theta[reg[0] - 1:reg[1], reg[2] - 1:reg[3]]
        mask = np.where((polstats >= polrange[0]) & (polstats <= polrange[1]))
        polstats = polstats[mask]
        mask = np.where((thetastats >= anglerange[0])
                        & (thetastats <= anglerange[1]))
        thetastats = thetastats[mask]
        polmedian = np.median(polstats)
        thetamedian = np.median(thetastats)

        outfile = self.datain.filenamebegin + self.procname.upper()
        outfile += self.datain.filenameend.replace('.fits', '.png')

        # Make top left subplot, Polarization Percent
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(2, 2, 1)
        img = ax.imshow(pol, origin='lower', cmap='rainbow',
                        interpolation='nearest',
                        aspect='equal', vmin=0., vmax=100.0)
        ax.plot([reg[0] - 1.5, reg[1] - 0.5, reg[1] - 0.5,
                 reg[0] - 1.5, reg[0] - 1.5],
                [reg[2] - 1.5, reg[2] - 1.5, reg[3] - 0.5,
                 reg[3] - 0.5, reg[2] - 1.5], 'w--', lw=2)
        fig.colorbar(img, orientation='vertical', pad=0.01)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_xlim([-0.5, 31.5])
        ax.set_ylim([-0.5, 40.5])

        # Make top right subplot, Polarization Angle
        ax = fig.add_subplot(2, 2, 2)
        img = ax.imshow(theta, origin='lower', cmap='rainbow',
                        interpolation='nearest',
                        aspect='equal', vmin=anglerange[0],
                        vmax=anglerange[1])
        ax.plot([reg[0] - 1.5, reg[1] - 0.5, reg[1] - 0.5,
                 reg[0] - 1.5, reg[0] - 1.5],
                [reg[2] - 1.5, reg[2] - 1.5, reg[3] - 0.5,
                 reg[3] - 0.5, reg[2] - 1.5], 'w--', lw=2)
        fig.colorbar(img, orientation='vertical', pad=0.01)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_xlim([-0.5, 31.5])
        ax.set_ylim([-0.5, 40.5])

        # Make bottom left subplot, Polarization Percent Histogram
        ax = fig.add_subplot(2, 2, 3)
        ax.hist(polstats.flatten())
        limits = ax.get_ylim()
        ax.plot([polmedian, polmedian], limits, 'r--')
        ax.text(0.9, 0.9,
                'Median = %.1f %%' % polmedian,
                horizontalalignment='right',
                transform=ax.transAxes)
        ax.set_xlabel('Polarization (%)')
        ax.set_ylabel('Number')

        # Make bottom right subplot, Polarization Angle Histogram
        ax = fig.add_subplot(2, 2, 4)
        ax.hist(thetastats.flatten())
        limits = ax.get_ylim()
        ax.plot([thetamedian, thetamedian], limits, 'r--')
        ax.text(0.9, 0.9,
                r'Median = $%.1f^\circ$' % thetamedian,
                horizontalalignment='right',
                transform=ax.transAxes)
        ax.set_xlabel('Pol. Angle (deg)')
        ax.set_ylabel('Number')

        # Finish plots
        fig.subplots_adjust(hspace=0.05)
        title = os.path.basename(outfile)
        fig.suptitle(title, y=0.95)
        fig.savefig(outfile)
        fig.clear()

        self.auxout = [outfile]
        log.info('Saved result %s' % outfile)
