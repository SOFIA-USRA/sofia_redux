# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Skydip plots pipeline step."""

import os

from astropy import log
from matplotlib.backends.backend_agg \
    import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from sofia_redux.instruments.hawc.datatext import DataText
from sofia_redux.instruments.hawc.stepmoparent import StepMOParent

__all__ = ['StepSkydip']


class StepSkydip(StepMOParent):
    """
    Produce diagnostic plots from skydip data.

    This step uses demodulated data taken in sky dip mode to produce
    a plot of averaged raw data vs. elevation. If a skydip fit was
    produced by the scan map algorithm, it is also stored as a plot.

    This step should be run as the final step in the skydip recipe.
    Pipeline steps for this mode should be run in this order:

        - `sofia_redux.instruments.hawc.steps.StepCheckhead`
        - `sofia_redux.instruments.hawc.steps.StepScanMap`
        - `sofia_redux.instruments.hawc.steps.StepFluxjump`
        - `sofia_redux.instruments.hawc.steps.StepPrepare`
        - `sofia_redux.instruments.hawc.steps.StepDemodulate`
        - `sofia_redux.instruments.hawc.steps.StepDmdPlot`
        - `sofia_redux.instruments.hawc.steps.StepDmdCut`
        - `sofia_redux.instruments.hawc.steps.StepSkydip`

    This step produces two PNG images, saved to the same directory
    and base name as the input data. The output data is otherwise
    identical to the input data.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'skydip', and are named with
        the step abbreviation 'SDP'.

        Parameters defined for this step are:

        indata : str
            Column name for the data to be displayed (vs. elevation).
            Either 'R array AVG' or 'T array AVG' is recommended.
        """
        # Name of the pipeline reduction step
        self.name = 'skydip'
        self.description = 'Make Skydip Plots'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'sdp'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['indata', 'R array AVG',
                               'Input Data: Column name for data '
                               'to be used'])

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is multi-in, multi-out (MIMO),
        self.datain must be a list of DataFits objects. The output
        is also a list of DataFits objects, stored in self.dataout.

        The process is:

        1. Plot elevation vs. averaged raw data.
        2. Read in a scan map skydip fit ('tmp*.dat' in the same directory
           as the input data).
        3. Plot scan map fit data.
        """
        # Make image from demodulated data

        # Input is copied to output, unmodified
        self.dataout = self.datain

        # Get elevation and averaged median signal
        # for each chop, for each file.
        elev = np.zeros((0, ))
        signal = np.zeros((0, ))

        # Loop through each file and collect data
        for din in self.datain:
            elev = np.append(elev, din.table['Elevation']
                             - din.table['Elevation Error'] / 3600.)
            dat = din.table[self.getarg('indata')]
            while len(dat.shape) > 1:
                dat = np.median(dat, axis=1)
            signal = np.append(signal, dat)

        # Make output data list
        self.auxout = []

        # Remove high and low points
        sigmin = np.nanpercentile(signal, 2)
        sigmax = np.nanpercentile(signal, 98)
        with np.errstate(invalid='ignore'):
            elev = elev[np.where(signal > sigmin)]
            signal = signal[np.where(signal > sigmin)]
            elev = elev[np.where(signal < sigmax)]
            signal = signal[np.where(signal < sigmax)]

        # Plot data
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot()
        ax.plot(1.0 / np.sin(np.pi * elev / 180.), signal, 'rd')
        ax.set_xlabel('1 / sin(Elevation)')
        ax.set_ylabel('Median Signal (%s)' % self.getarg('indata'))
        pngname = self.datain[-1].filename.replace('.fits', '_skydiplot.png')
        fig.savefig(pngname)
        self.auxout.append(pngname)
        log.info('Saved result %s' % pngname)

        # Make image from scanmap data

        # Search for output .dat file
        datfile = os.path.basename(self.datain[0].filenamebegin
                                   + 'SMP'
                                   + self.datain[0].filenameend
                                   + '.dat')
        if not os.path.isfile(datfile):
            return

        # Load data
        dt = DataText(config=self.config)
        dt.load(datfile)

        # Get data
        elev = []
        obs = []
        model = []
        for dat in dt.data:
            # Don't record data with no observations
            if '...' in dat:
                continue
            spl = dat.split('\t')
            elev.append(float(spl[0]))
            obs.append(float(spl[1]))
            model.append(float(spl[2]))
        elev = np.array(elev)
        obs = np.array(obs)
        model = np.array(model)

        # Get taulabel and remove bad characters
        taulabel = 'tau = ' + str(dt.getheadval('tau')).replace('+-', r'$\pm$')
        taulabel = ''.join([c for c in taulabel if ord(c) < 128])

        # Make Plot
        fname = os.path.split(self.datain[0].filenamebegin)[1]
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot()
        ax.plot(elev, model, 'k-', linewidth=2,
                label='Model: %s' % taulabel)
        ax.plot(elev, obs, 'rs-', linewidth=1, markersize=3,
                label='Skydip %s' % fname)

        # label axis
        ax.set_xlabel('Elevation (deg)')
        ax.set_ylabel('Mean Pixel Response (counts)')
        fig.legend()

        # Save image and add to output
        pngname = self.datain[-1].filename.replace('.fits', '_skymodel.png')
        fig.savefig(pngname)
        self.auxout.append(pngname)
        log.info('Saved result %s' % pngname)

        # Remove the temporary scanmap file
        try:
            log.debug(f"Removing {datfile}")
            os.remove(datfile)
        except OSError:
            pass
