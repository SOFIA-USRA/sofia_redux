# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Data splitting pipeline step."""

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.instruments.hawc.stepparent import StepParent
from sofia_redux.instruments.hawc.datafits import DataFits

__all__ = ['StepSplit']


class StepSplit(StepParent):
    """
    Split the data by nod, HWP angle, and by additive and difference signals.

    This pipeline step takes demodulated R and T data and splits the
    data by HWP angle and Left/Right nod position. Then, it subtracts
    the T array from the R array, storing R-T data for each HWP and nod
    position. It also adds the T array to the R array, storing R+T
    data for each HWP and nod position. It also identifies and flags
    for later use any widow pixels: pixels that are good in R, but bad
    in T, or vice versa.

    The Nod Index and HWP Index columns in the demodulated data table
    are used to identify nod and HWP transition points for splitting.
    Variances in R and T are propagated to variance arrays for R+T and
    R-T, and are also stored directly for each nod/HWP combination,
    for later use in computing covariances.

    Input data for this step must contain R Array and T Array images
    with associated variances, and a DEMODULATED DATA table, containing
    Nod Index and HWP Index columns. This step is typically run after
    `sofia_redux.instruments.hawc.steps.StepShift`.

    For each HWP angle *M* and Nod *N* this step creates six
    images: DATA R-T HWP M NOD N, DATA R+T HWP M NOD N, VAR
    R-T HWP M NOD N, VAR R+T HWP M NOD N, VAR R HWP M NOD N, and
    VAR T HWP M NOD N. In addition, it creates a table containing the
    rows corresponding to a given HWP and Nod, named TABLE HWP M NOD N.
    Finally, it adds a single bad pixel mask image.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'split', and are named with
        the step abbreviation 'SPL'.

        Parameters defined for this step are:

        nod_tol : float
            Nod tolerance, as the percent difference allowed in number
            of chop cycles between 1st and 2nd left nods, and between
            left and right nods.
        rtarrays : str
            Use both R and T arrays ('RT'), or only R ('R') or only T ('T').
        """
        # Name of the pipeline reduction step
        self.name = 'split'
        self.description = 'Split By Nod/HWP'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'spl'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['nod_tol', 30.0,
                               'Nod tolerance, as the percent difference '
                               'allowed in number of chop cycles between '
                               '1st and 2nd left, and between left and right'])
        self.paramlist.append(['rtarrays', 'RT',
                               'Use both R and T arrays (RT), '
                               'or only R (R) or only T (T)'])

    def _subtable(self, table, mask):
        """
        Create a new table with only certain rows specified by a mask.

        Parameters
        ----------
        table : fits.FITS_rec
            The table to pull from.
        mask : array-like
            The row indices to keep.

        Returns
        -------
        BinTableHDU
            The subtable.
        """

        names = table.names
        formats = table.columns.formats
        dims = table.columns.dims
        units = table.columns.units
        cols = []

        for n, f, d, u in zip(names, formats, dims, units):
            cols.append(fits.Column(name=n, format=f, dim=d, unit=u,
                                    array=table.field(n)[mask]))
        tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
        return tbhdu

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object. The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Find transition points between nod beams and HWP angles.
        2. For each set of nod and angle combination, compute R+T
           and R-T.
        3. Identify widow pixels and multiply their fluxes by 2 to
           account for the missing data.
        4. Store the data in separate extensions for each combination.

        Raises
        ------
        ValueError
            If zero chop cycles are found in any of the nod positions.
        """
        rdata = self.datain.imageget('R array')
        tdata = self.datain.imageget('T array')
        rvar = self.datain.imageget('R array VAR')
        tvar = self.datain.imageget('T array VAR')
        table = self.datain.tableget('demodulated data')
        rbadmask = self.datain.imageget('R Bad Pixel Mask').astype(np.int32)
        tbadmask = self.datain.imageget('T Bad Pixel Mask').astype(np.int32)

        hwpidx = table['HWP Index']
        nodidx = table['Nod Index']
        nhwp = self.datain.getheadval('nhwp')

        # check nod tolerance
        nod_tol = self.getarg('nod_tol')
        nodpatt = self.datain.getheadval('nodpatt')
        nnod = len(set(nodpatt))
        if nodpatt == 'ABBA':
            for hwp in range(nhwp):
                left1 = len(np.where((hwpidx == hwp) & (nodidx == 0))[0])
                right = len(np.where((hwpidx == hwp) & (nodidx == 1))[0])
                left2 = len(np.where((hwpidx == hwp) & (nodidx == 2))[0])
                if left1 == 0:
                    msg = 'HWP %d: Zero chops in 1st left!' % hwp
                    log.error(msg)
                    raise ValueError(msg)
                elif left2 == 0:
                    msg = 'HWP %d: Zero chops in 2nd left!' % hwp
                    log.error(msg)
                    raise ValueError(msg)
                elif right == 0:
                    msg = 'HWP %d: Zero chops in right!' % hwp
                    log.error(msg)
                    raise ValueError(msg)

                diff1 = 100 * 2 * (abs(left1 - left2)
                                   / float(left1 + left2))
                diff2 = 100 * 2 * (abs(right - left1 - left2)
                                   / float(right + left1 + left2))
                if diff1 > nod_tol:
                    log.warning('HWP %d: Number of chops between 1st and '
                                '2nd left differ by %f%% (max %f%%)' %
                                (hwp, diff1, nod_tol))
                if diff2 > nod_tol:
                    log.warning('HWP %d: Number of chops between '
                                'left and right differ by '
                                '%f%% (max %f%%)' %
                                (hwp, diff2, nod_tol))

            # Note: we assume that we are doing a LRRL
            # nod pattern, and thus the second left,
            # with nodidx = 2, should be set to 0, so we can
            # pick it up below and only compute a Left and a Right Nod.
            mask = np.where(nodidx == 2)
            nodidx[mask] = 0

        elif nnod == 1:
            pass
        else:
            # unknown nod pattern
            msg = 'Can only process data with ABBA nod pattern'
            log.error(msg)
            raise ValueError(msg)

        # Make sure all pixels in the missing sub-array (T1) are assigned
        # as bad pixels. This will allow the pixels in R1 to be considered
        # as widow pixels (and the flux will be multiplied by 2,
        # see steps below)
        tbadmask[:, 32:] = 2

        # Flag to use only R or only T array (by setting
        # all the pixels in the other array as bad pixels)
        rtarrays = self.getarg('rtarrays')
        if rtarrays == 'R':
            tbadmask[:, :] = 2
        elif rtarrays == 'T':
            rbadmask[:, :] = 1

        # If any bad pixel in the T bad pixel
        # mask is assigned with any number >= 1,
        # then change it to 2. In the same way,
        # if any bad pixel in the R bad pixel
        # mask is assigned with any number >= 1,
        # then change it to 1.
        tbadmask[np.where(tbadmask >= 1)] = 2
        rbadmask[np.where(rbadmask >= 1)] = 1

        # identify bad pixels
        # R pixels bad in bit 1 (bitwise AND) (0 or 1)
        tmp_r = rbadmask & 1
        # T pixels bad in bit 2 (bitwise AND) (0 or 2)
        tmp_t = tbadmask & 2

        # set fluxes of tmp_r and tmp_t pixels to zero.
        # This is needed if we later fill in the fluxes.
        # first broadcast to 3-D so we select the right pixels in r and t
        tmp_r = np.ones((rdata.shape[0], 1, 1), dtype=np.int32) * tmp_r
        tmp_t = np.ones((tdata.shape[0], 1, 1), dtype=np.int32) * tmp_t
        rdata[tmp_r == 1] = 0
        tdata[tmp_t == 2] = 0
        rvar[tmp_r == 1] = 0
        tvar[tmp_t == 2] = 0

        # If using only R or only T, set data in bad pixels to NaN
        if rtarrays == 'R':
            rdata[tmp_r == 1] = np.nan
            rvar[tmp_r == 1] = np.nan
        elif rtarrays == 'T':
            tdata[tmp_t == 2] = np.nan
            tvar[tmp_t == 2] = np.nan

        # Note: it seems silly to compute R-T if nhwp = 1, but we use the R-T
        # data for chauvenet's criterion in stepcombine. By using R-T, we
        # mostly remove sky noise and thus get a more reliable outlier
        # rejection.
        # R - T
        rmt = rdata - tdata
        # R + T
        rpt = rdata + tdata
        rptvar = rvar + tvar

        # multiply flux of widow pixels by two. This way, later equations in
        # stepstokes can compute correct stokes parameters for these widow
        # pixels.
        # this should make a boolean array of the widow pixels
        widow = ((tmp_r == 1) & (tmp_t == 0)) + ((tmp_r == 0) & (tmp_t == 2))
        rmt[widow] = 2 * rmt[widow]
        rpt[widow] = 2 * rpt[widow]
        rptvar[widow] = 4 * rptvar[widow]

        # build output pipedata object by splitting on HWP angle and Nod
        self.dataout = DataFits(config=self.datain.config)
        self.dataout.filename = self.datain.filename
        self.dataout.setheader(self.datain.header)
        masks = []
        for hwp in range(nhwp):
            for nod in range(nnod):
                mask = np.where((hwpidx == hwp) & (nodidx == nod))
                masks.append(mask)
                self.dataout.imageset(
                    rmt[mask], 'DATA R-T HWP%d NOD%d' % (hwp, nod))
                self.dataout.imageset(
                    rpt[mask], 'DATA R+T HWP%d NOD%d' % (hwp, nod))
                self.dataout.imageset(
                    rptvar[mask], 'VAR R-T HWP%d NOD%d' % (hwp, nod))
                self.dataout.imageset(
                    rptvar[mask], 'VAR R+T HWP%d NOD%d' % (hwp, nod))
                self.dataout.imageset(
                    rvar[mask], 'VAR R HWP%d NOD%d' % (hwp, nod))
                self.dataout.imageset(
                    tvar[mask], 'VAR T HWP%d NOD%d' % (hwp, nod))
                tbhdu = self._subtable(table, mask)
                self.dataout.tableset(
                    tbhdu.data, 'TABLE HWP%d NOD%d' % (hwp, nod),
                    tbhdu.header)

        # Merge R and T masks
        bad = rbadmask | tbadmask

        # write out combined bad pixel mask
        self.dataout.imageset(bad, 'Bad Pixel Mask')
