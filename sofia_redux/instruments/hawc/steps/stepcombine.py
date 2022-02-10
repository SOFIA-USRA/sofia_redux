# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Time series combination pipeline step."""

import numpy as np

from astropy import log
from astropy.io import fits

from sofia_redux.instruments.hawc.steps.basehawc import clipped_mean
from sofia_redux.instruments.hawc.stepparent import StepParent

__all__ = ['StepCombine']


class StepCombine(StepParent):
    """
    Combine time series data for R+T and R-T flux samples.

    This step averages all chop-subtracted samples for each nod and
    HWP setting, for the R+T and R-T images.  Outliers are identified
    via iterative sigma-clipping (Chauvenet's criterion).  Errors
    are propagated from the input variance images or, optionally,
    reported as the standard deviation across the time samples.

    After this step, R-T images are propagated for polarimetry data
    only (NHWP > 1).

    This step should be run after the
    `sofia_redux.instruments.hawc.steps.StepSplit` pipeline step.
    It requires the following extensions: for each
    HWP angle *M* and Nod *N* there should be six images: DATA R-T
    HWP M NOD N, DATA R+T HWP M NOD N, VAR R-T HWP M,
    NOD N, VAR R+T HWP M NOD N, VAR R HWP M NOD N, VAR T
    HWP M NOD N. In addition, there must be a table containing the
    rows corresponding to a given HWP and Nod, named TABLE HWP M
    NOD N. Finally, there should be a single bad pixel mask image.

    The output extensions are the same as the input extensions,
    except that VAR R+T and VAR R-T are replaced with ERROR R+T
    and ERROR R-T extensions.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'combine', and are named with
        the step abbreviation 'CMB'.

        Parameters defined for this step are:

        sigma : float
            Reject outliers more than this many sigma from the mean.
        sum_sigma : float
            Reject additional R+T outliers more than this many sigma
            from the mean.
        use_error : bool
            Set to True to use the standard deviation across the time
            samples as the output error, rather than propagating input
            variances.
        """
        # Name of the pipeline reduction step
        self.name = 'combine'
        self.description = 'Combine Time Series'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'cmb'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['sigma', 3.0,
                               "Reject outliers more than this many "
                               "sigma from the mean"])
        self.paramlist.append(['sum_sigma', 4.0,
                               "Reject additional R+T outliers more "
                               "than sum_sigma from the mean"])
        self.paramlist.append(['use_error', False,
                               "Set to True to use Chauvenet output "
                               "errors rather than propagating input "
                               "variances"])

    def comb_table(self, table, newmask):
        """
        Average all rows for a table.

        Parameters
        ----------
        table : fits.FITS_rec
            The table to average.
        newmask : array-like of bool
            Table rows to combine.

        Returns
        -------
        BinTableHDU
            The averaged table.
        """
        names = table.names
        formats = table.columns.formats
        dims = table.columns.dims
        units = table.columns.units
        cols = []

        outrow = self.dataout.tablemergerows(table[newmask])
        for n, f, d, u in zip(names, formats, dims, units):
            cols.append(fits.Column(name=n, format=f, dim=d, unit=u,
                                    array=[outrow[n]]))
        tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
        return tbhdu

    def run(self):
        r"""
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object.  The output
        is also a DataFits object, stored in self.dataout.

        This step combines the chop cycles for each HWP angle
        and Nod separately (and each pixel as well). This happens first for
        the R-T data, as follows:

        1. For each pixel in R-T, compute the mean value and standard
           deviation.
        2. Reject any chop cycles more than *sigma* away from the mean.
        3. Repeat 1-2 until no more chop cycles are removed.

        Any masked pixels from R-T deglitching are also masked in R+T.
        Additional deglitching in R+T follows the same outlier rejection as
        for R-T, with the sigma cutoff specified by the *sum\_sigma*
        parameter.
        """
        self.dataout = self.datain.copy()
        nhwp = self.datain.getheadval('nhwp')
        nodpatt = self.datain.getheadval("nodpatt")
        nnod = len(set(nodpatt))

        chauvenet = self.getarg('sigma')
        sum_sigma = self.getarg('sum_sigma')
        use_error = self.getarg('use_error')

        for hwp in range(nhwp):
            for nod in range(nnod):
                log.debug('starting hwp %d nod %d' % (hwp, nod))
                rmt_data = 'DATA R-T HWP%d NOD%d' % (hwp, nod)
                rpt_data = 'DATA R+T HWP%d NOD%d' % (hwp, nod)
                rmt_var = 'VAR R-T HWP%d NOD%d' % (hwp, nod)
                rpt_var = 'VAR R+T HWP%d NOD%d' % (hwp, nod)
                r_var = 'VAR R HWP%d NOD%d' % (hwp, nod)
                t_var = 'VAR T HWP%d NOD%d' % (hwp, nod)
                rmt_sigma = 'ERROR R-T HWP%d NOD%d' % (hwp, nod)
                rpt_sigma = 'ERROR R+T HWP%d NOD%d' % (hwp, nod)

                # make sure data is float64
                rmt = self.datain.imageget(rmt_data).astype(np.float64)
                rpt = self.datain.imageget(rpt_data).astype(np.float64)
                rptv = self.datain.imageget(rpt_var).astype(np.float64)
                rv = self.datain.imageget(r_var).astype(np.float64)
                tv = self.datain.imageget(t_var).astype(np.float64)
                table = self.datain.tableget('TABLE HWP%d NOD%d' % (hwp, nod))

                nplane, nrow, ncol = rmt.shape
                mask = np.zeros_like(rmt)

                # run Chauvenet's criterion to reject outliers
                mean, sigma = clipped_mean(rmt, mask, sigma=chauvenet)

                maskvar = rptv.copy()
                maskvar[mask == 1] = np.nan
                count = np.sum(1 - mask, axis=0)
                rmtv = np.nansum(maskvar, axis=0) / count ** 2

                num = int(np.sum(mask))
                log.info('R-T deglitching: masked %d of %d '
                         'samples in hwp %d, nod %d' %
                         (num, nplane * nrow * ncol, hwp, nod))

                # Note: if nhwp = 1, then it doesn't really make sense to have
                # R-T data.  We'll keep it for this step for consistency,
                # but it will be discarded after this step.
                self.dataout.imageset(mean, rmt_data)
                if use_error:
                    self.dataout.imageset(sigma, rmt_sigma)
                else:
                    self.dataout.imageset(np.sqrt(rmtv), rmt_sigma)
                self.dataout.imagedel(rmt_var)

                # run Chauvenet's criterion to find additional outliers
                mean, sigma = clipped_mean(rpt, mask, sigma=sum_sigma)

                rptv[mask == 1] = np.nan
                count = np.sum(1 - mask, axis=0)
                rptv = np.nansum(rptv, axis=0) / count ** 2

                # propagate R and T variance as well --
                # needed for covariance calculations later
                rv[mask == 1] = np.nan
                tv[mask == 1] = np.nan
                rv = np.nansum(rv, axis=0) / count ** 2
                tv = np.nansum(tv, axis=0) / count ** 2

                log.info('R+T deglitching: masked additional '
                         '%d of %d samples in hwp %d, nod %d' %
                         (int(np.sum(mask)) - num,
                          nplane * nrow * ncol, hwp, nod))
                self.dataout.imageset(mean, rpt_data)

                if use_error:
                    # keep the output errors from the Chauvenet algorithm
                    self.dataout.imageset(sigma, rpt_sigma)

                    # also set the R and V variances to zero; no covariances
                    # available for this error propagation method
                    log.warning('Covariances between initial '
                                'Stokes parameters are '
                                'not propagated with Chauvenet errors')
                    rv *= 0.0
                    tv *= 0.0

                else:
                    # otherwise, keep the propagated variances
                    self.dataout.imageset(np.sqrt(rptv), rpt_sigma)

                # delete the old variances, add the new ones onto the end
                self.dataout.imagedel(rpt_var)
                self.dataout.imagedel(r_var)
                self.dataout.imagedel(t_var)
                self.dataout.imageset(rv, r_var)
                self.dataout.imageset(tv, t_var)

                tmpmask = np.ones(len(table), dtype=np.bool)
                tbhdu = self.comb_table(table, tmpmask)
                self.dataout.tableset(tbhdu.data,
                                      'TABLE HWP%d NOD%d' % (hwp, nod),
                                      tbhdu.header)
