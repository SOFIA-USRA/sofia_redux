# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Diagnostic noise plot pipeline step."""

import os

from astropy import log
import numpy as np
from matplotlib.backends.backend_agg \
    import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


from sofia_redux.instruments.hawc.stepparent import StepParent

__all__ = ['StepNoisePlots']


class StepNoisePlots(StepParent):
    """
    Produce diagnostic plots for lab-generated noise data.

    Input is the power spectrum image created from noise data,
    by the StepNoiseFFT step.  Output is a number of diagnostic
    images.  The FITS input data is not modified, except for the
    addition of a table containing binned median noise values
    by pixel.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'noiseplots', and are named with
        the step abbreviation 'NPL'.

        No parameters are defined for this step.
        """
        # Name of the pipeline reduction step
        self.name = 'noiseplots'
        self.description = "Make Noise Plots"

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'npl'

        # Clear Parameter list
        self.paramlist = []

    def _median_plot(self, frequ, med, top90, xlim, ylim, outfile, allmed):
        # make figure
        fig = Figure(figsize=(6, 4), tight_layout=True)
        FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)
        ax.loglog()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.plot(frequ, med, label='Median Noise')
        ax.plot(frequ, top90, label='90% Level')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Noise level (A/sqrtHz)')
        ax.legend()
        ax.set_title('Median 8-12Hz =%.3fnA/sqrtHz' % (allmed * 1e9),
                     size=10)

        # save median plot
        outfile += self.datain.filenameend.replace('.fits', '_med.png')
        title = os.path.basename(outfile)
        fig.suptitle(title, y=0.95)
        fig.savefig(outfile, dpi=200)
        fig.clear()

        self.auxout.append(outfile)
        log.info('Saved result %s' % outfile)

    def _fft_image(self, fft, ylim, rown, coln, bini, binf, outfile):
        # make figure
        fig = Figure(figsize=(6, 9))
        FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)
        img = ax.imshow(np.log10(fft.T), vmin=np.log10(ylim[0]),
                        vmax=np.log10(ylim[1]),
                        interpolation='nearest',
                        aspect='auto', origin='lower')
        ax.set_yticks(np.arange(rown) * coln)
        ax.set_yticklabels([f'R{n}' for n in range(rown)],
                           verticalalignment='bottom')

        ax.set_xticks(bini)
        ax.set_xticklabels(binf[:-1])
        ax.set_xlabel('Frequency (Hz)')

        ax.set_title('Log10(A/sqrt(Hz))')
        fig.colorbar(img, orientation='vertical')

        # save plot
        outfile += self.datain.filenameend.replace('.fits', '_specmap.png')
        title = os.path.basename(outfile)
        fig.suptitle(title)
        fig.savefig(outfile)
        fig.clear()

        self.auxout.append(outfile)
        log.info('Saved result %s' % outfile)

    def _image_8_12(self, meds8_12, ylim, rown, coln, allmed8_12, outfile):
        fig = Figure(figsize=(coln // 8 + 1, rown // 8))
        FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)
        meds8_12.shape = (rown, coln)
        img = ax.imshow(np.log10(meds8_12), vmin=np.log10(ylim[0]),
                        vmax=np.log10(ylim[1]),
                        interpolation='nearest',
                        aspect='auto', origin='lower')

        ax.text(0, -5,
                'Median all pixels = %.3fnA/sqrtHz' % (allmed8_12 * 1e9))
        ax.set_title('Median 8-12Hz - Log10(A/sqrt(Hz))')
        fig.colorbar(img, orientation='vertical')

        # save plot
        outfile += self.datain.filenameend.replace('.fits', '_8-12Hz.png')
        title = os.path.basename(outfile)
        fig.suptitle(title)
        fig.savefig(outfile)
        fig.clear()

        self.auxout.append(outfile)
        log.info('Saved result %s' % outfile)

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object.  The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Bin the FFT data from the previous step.
        2. Generate plots from the binned FFT data.

        """
        # copy input to output
        self.dataout = self.datain.copy()

        # get the data from the input file
        fft_data = self.datain.image

        # data shape: pixels x frequencies
        rown = 41
        pixn = fft_data.shape[0]
        freqn = fft_data.shape[1]
        coln = pixn // rown
        f0 = float(self.datain.getheadval('CRVAL1'))
        df = float(self.datain.getheadval('CDELT1'))

        # check if data needs to be cut, for missing 4th array
        fft_data.shape = (rown, coln, freqn)
        if fft_data[:, 3 * coln // 4:, :].max() == 0:
            fft_data = fft_data[:, :3 * coln // 4, :].copy()
            coln = 3 * coln // 4
            pixn = coln * rown
        fft_data.shape = (pixn, freqn)

        # make frequencies array
        linfrequ = f0 + df * np.arange(freqn)

        # set bin parameters
        nbins = 512
        fmin = 10.0 * df
        fmax = linfrequ.max()
        frange = [np.log(fmin), np.log(fmax)]

        # get number of elements in each bin and edges
        binhist, binedge = np.histogram(np.log(linfrequ),
                                        bins=nbins, range=frange)
        binedge = np.exp(binedge)

        # get number of elements in good bins and bin centers
        binfrequ = [0.5 * (binedge[i] + binedge[i + 1])
                    for i in range(nbins) if binhist[i] > 0]
        binfrequ = np.array(binfrequ)

        binn = [binhist[i] for i in range(nbins) if binhist[i] > 0]
        binn = np.array(binn)
        nfrequs = len(binn)
        frequ = binfrequ

        # prepare binned data
        fftlin = fft_data[:, -binn.sum():]
        ind0 = 0
        fft = np.zeros((nfrequs, pixn))
        for i in range(nfrequs):
            fft[i, :] = np.mean(fftlin[:, ind0:ind0 + binn[i]], axis=1)
            ind0 += binn[i]

        # make table averages
        binf = np.array([1.0, 3.0, 10.0, 30.0, 100.0, 300.0])

        # remove empty bins
        while binf[0] < frequ[0]:
            binf = binf[1:]
        binn = len(binf) - 1

        # fill bin data
        binvals = np.zeros((binn, pixn))
        bini = np.zeros(binn)
        for i in range(binn):
            try:
                indmin = np.min(np.where(frequ >= binf[i]))
                indmax = np.max(np.where(frequ <= binf[i + 1]))
            except ValueError:  # pragma: no cover
                binvals[i] = np.nan
                bini[i] = len(frequ)
            else:
                binvals[i] = np.median(fft[indmin:indmax, :], axis=0)
                bini[i] = indmin

        # pixel values (names and indices - electronic indexing)
        pixnames = list(range(pixn))
        pixrows = np.zeros(pixn)
        pixcols = np.zeros(pixn)

        # loop over rows
        for ri in range(rown):
            # loop over pixels in row
            for ci in range(coln):
                # Get indices
                pixrows[coln * ri + ci] = ri
                pixcols[coln * ri + ci] = ci
                # Get pixel name
                pixnames[coln * ri + ci] = f'R{ri}C{ci}'

        # prepare table
        self.dataout.tableaddcol('Pixel', pixnames)
        self.dataout.tableaddcol('Row Ind', pixrows)
        self.dataout.tableaddcol('Col Ind', pixcols)

        # spectral values
        for i in range(binn):
            if binf[i] < 1.0:  # pragma: no cover
                name = "%.1f-%.1f Hz nA/sqrtHz" % (binf[i], binf[i + 1])
            else:
                name = "%.0f-%.0f Hz nA/sqrtHz" % (binf[i], binf[i + 1])
            self.dataout.tableaddcol(name, 1e9 * binvals[i, :])

        # get plot ranges
        fmin = frequ[0]
        fmax = frequ[-1]
        pmin = 1e-9
        pmax = 1e-9

        # make median and 90% Arrays
        med = np.zeros(nfrequs)
        top90 = np.zeros(nfrequs)
        for i in range(nfrequs):
            sort = np.sort(fft[i, :])
            med[i] = sort[pixn // 2]
            top90[i] = sort[pixn - pixn // 10]
        while pmin > med.min():
            pmin /= 10.0
        while pmax < top90.max():  # pragma: no cover
            pmax *= 10.0

        # get 8-12Hz medians (for label and next plot)
        indlist = [i for i in range(len(frequ)) if 8.0 < frequ[i] < 12.0]
        meds8_12 = np.median(fft[indlist, :], axis=0)
        allmed8_12 = np.median(meds8_12)

        # output basename for plots
        outfile = self.datain.filenamebegin + self.procname.upper()

        # make median plot
        self._median_plot(frequ, med, top90,
                          [fmin, fmax], [pmin, pmax],
                          outfile, allmed8_12)

        # make FFT image plot
        self._fft_image(fft, [pmin, pmax], rown, coln,
                        bini, binf, outfile)

        # make 8-12 Hz image plot
        self._image_8_12(meds8_12, [pmin, pmax], rown, coln,
                         allmed8_12, outfile)
