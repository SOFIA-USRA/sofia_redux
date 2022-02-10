# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Chop/nod mode Stokes parameters pipeline step."""

from astropy import log
import numpy as np

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.stepparent import StepParent

__all__ = ['StepStokes']


class StepStokes(StepParent):
    r"""
    Compute Stokes parameters for chop/nod polarimetry data.

    This step derives Stokes I, Q, and U images with associated
    uncertainties and covariances from R and T array images.
    If the data only has one HWP angle, then only Stokes I is
    computed.

    Input for this step is a DataFits object, as produced by the
    `sofia_redux.instruments.hawc.steps.StepNodPolSub` pipeline step:
    R-T and R+T images for each HWP angle, with associated errors, and
    variance images for R and T arrays.

    Output for this step contains the following image extensions:
    STOKES I, ERROR I, STOKES Q, ERROR Q, STOKES U, ERROR U,
    COVAR Q I, COVAR U I, COVAR Q U. Also, a table named TABLE DATA is
    created, and BAD PIXEL MASK is copied from the input file.
    For Nod-Pol data, the value of the initial HWP angle is saved as a
    header keyword (HWPINIT), to be read by
    `sofia_redux.instruments.hawc.steps.StepRotate`.

    Notes
    -----
    Stokes I is computed by averaging the R+T signal over
    all HWP angles (where N is the number of HWP angles):

       .. math:: I = \frac{1}{N} \sum_{\phi=1}^N (R+T)_{\phi}

       .. math:: \sigma_I = \frac{1}{N}
                            \sqrt{\sum_{\phi=1}^N \sigma_{R+T,\phi}^2}.

    The associated uncertainty in I is generally propagated from the
    previously calculated errors for R+T as above, but may be inflated by
    the median of the standard deviation of the R+T values across the HWP
    angles if necessary. In the most common case of four HWP angles at 0,
    45, 22.5, and 67.5 degrees, Stokes Q and U are computed as:

      .. math:: Q = \frac{1}{2} [(R-T)_{0} - (R-T)_{45}]

      .. math:: U = \frac{1}{2} [(R-T)_{22.5} - (R-T)_{67.5}]

    where :math:`(R-T)_{\phi}` is the differential R-T flux at the HWP
    angle :math:`\phi`. Uncertainties in Q and U are propagated from the
    input error values on R-T:

       .. math:: \sigma_Q = \frac{1}{2} \sqrt{\sigma_{R-T,0}^2
                                              + \sigma_{R-T,45}^2}

       .. math:: \sigma_U = \frac{1}{2} \sqrt{\sigma_{R-T,22.5}^2
                                              + \sigma_{R-T,67.5}^2}.

    Covariances between the Stokes parameters are derived from the
    variances in R and T as follows:

       .. math:: \sigma_{QI} = \frac{1}{8} [\sigma_{R,0}^2
                                            - \sigma_{R,45}^2
                                            - \sigma_{T,0}^2
                                            + \sigma_{T,45}^2]

       .. math:: \sigma_{UI} = \frac{1}{8} [\sigma_{R,22.5}^2
                                            - \sigma_{R,67.5}^2
                                            - \sigma_{T,22.5}^2
                                            + \sigma_{T,67.5}^2]

    The covariance between Q and U (:math:`\sigma_{QU}`) is zero at this
    stage, since they are derived from data for different HWP angles.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'scanstokes', and are named with
        the step abbreviation 'STK'.

        Parameters defined for this step are:

        hwp_tol : float
            Tolerance for difference from expected values for HWP
            angles. HWP angles for Stokes parameters must differ
            by no more than 45 +/- hwp_tol degrees.
        erri : str
            Inflation method for Stokes I errors.  May be
            'median', 'mean', or 'none'.
        erripolmethod : str
            Method for calculating Stokes I error.  Options are
            'hwpstddev', to compute them as a standard deviation
            across the HWP angles, or 'meansigma', to propagate
            them from the input errors (recommended).
        removeR1stokesi : bool
            If set, the R1 subarray for Stokes I is removed from
            the output.
        override_hwp_order : bool
            If set, then the first two HWP angles will be used
            for Q, last two for U, regardless of value.  This is
            necessary in the case where the HWP value is incorrectly
            recorded, but the HWP position as observed was correct.
        """
        # Name of the pipeline reduction step
        self.name = 'stokes'
        self.description = 'Compute Stokes'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'stk'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['hwp_tol', 5.0,
                               'HWP angles for Stokes parameters must '
                               'differ by no more than 45+-hwp_tol '
                               'degrees'])
        self.paramlist.append(['erri', 'median',
                               'How to inflate errors in I.  Can be '
                               'median, mean, or none'])
        self.paramlist.append(['erripolmethod', 'meansigma',
                               'Options are "hwpstddev" or "meansigma"'])
        self.paramlist.append(['removeR1stokesi', True,
                               'Remove R1 subarray for Stokes I'])
        self.paramlist.append(['override_hwp_order', False,
                               'If True, the first two HWP angles '
                               'will be used for Q, last two for U, '
                               'regardless of value'])

    def stokes(self, idx1, idx2, rmt_data, rmt_sigma, r_var, t_var):
        """
        Compute stokes Q and U.

        The index parameters control which Stokes parameter image
        is computed

        Parameters
        ----------
        idx1 : `list` of int
            Index for angle 1.
        idx2 : `list` of int
            Index for angle 2.
        rmt_data : array-like
            R - T flux data array. Should have three dimensions,
            where the first dimension indexes the HWP angle.
        rmt_sigma : array-like.
            R - T error data array. Dimensions should match rmt_data.
        r_var : array-like
            Variance for the R array. Dimensions should match rmt_data.
        t_var : array-like
            Variance for the T array. Dimensions should match rmt_data.

        Returns
        -------
        stokes : array-like
            The Stokes Q or U flux image.
        dstokes : array-like
            The error on the Stokes Q or U flux.
        stokes_icov : array-like
            The covariance on the Stokes Q or U image, with respect
            to the Stokes I image.
        """

        # propagation equations:
        # (for the most common 4 HWP case)
        # Q = (1/2) (R1 - R3 - T1 + T3)
        # U = (1/2) (R2 - R4 - T2 + T4)
        # VQ = (1/4) (VR1 + VR3 + VT1 + VT3)
        # VU = (1/4) (VR2 + VR4 + VT2 + VT4)
        # cov(Q, I) = (1/8) (VR1 - VR3 - VT1 + VT3)
        # cov(U, I) = (1/8) (VR2 - VR4 - VT2 + VT4)
        # cov(Q, U) = 0

        count = float(2 * len(idx1))
        stokes = (rmt_data[idx1].sum(axis=0)
                  - rmt_data[idx2].sum(axis=0)) / count
        dstokes = np.sqrt(np.sum(rmt_sigma[idx1 + idx2] ** 2, axis=0)) / count
        stokes_icov = np.sum(r_var[idx1] - r_var[idx2]
                             - t_var[idx1] + t_var[idx2],
                             axis=0) / (2 * count**2)
        return stokes, dstokes, stokes_icov

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object. The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Compute Stokes I from R+T at all angles.
           Propagate or recalculate errors on Stokes I.
        2. Compute Stokes Q and U from R-T at angles separated
           by 45 degrees.  Propagate associated errors
           and covariances.
        """
        self.dataout = DataFits(config=self.datain.config)
        self.dataout.filename = self.datain.filename
        self.dataout.setheader(self.datain.header)

        nhwp = self.datain.getheadval('nhwp')
        hwp_tol = abs(self.getarg('hwp_tol'))
        erri_inflate = self.getarg('erri').lower()
        erripolmethod = self.getarg('erripolmethod').lower()
        remove_r1stokesi = self.getarg('removeR1stokesi')
        override_hwp = self.getarg('override_hwp_order')

        if erri_inflate not in ['median', 'mean', 'none']:
            msg = 'errI parameter value must be MEDIAN, MEAN, or NONE'
            log.error(msg)
            raise ValueError(msg)
        if erripolmethod not in ['hwpstddev', 'meansigma']:
            msg = 'Method to calculate Error I for polarization ' \
                  'data must be HWPSTDDEV or MEANSIGMA'
            log.error(msg)
            raise ValueError(msg)

        # Number of HWP angles is 1, or a multiple of 4 and
        # less than 16.  This assumes
        # that HWP angles will always be separated by 22.5 deg
        # if they are taken in multiples of 4.
        # Initial angle must be 0.
        if nhwp != 1 and nhwp % 4 != 0:
            msg = 'Number of HWP angles must be multiple ' \
                  'of 4.'
            log.error(msg)
            raise ValueError(msg)
        if nhwp > 16:
            msg = 'Maximum number of HWP angles is 16'
            log.error(msg)
            raise ValueError(msg)

        # arrays to contain the data and errors for each HWP angle
        rpt_data = np.array([self.datain.imageget('Data R+T HWP%d' % hwp)
                             for hwp in range(nhwp)])
        rpt_sigma = np.array([self.datain.imageget('ERROR R+T HWP%d' % hwp)
                              for hwp in range(nhwp)])

        if nhwp > 1:
            rmt_data = np.array([self.datain.imageget('Data R-T HWP%d' % hwp)
                                 for hwp in range(nhwp)])
            rmt_sigma = np.array([self.datain.imageget('ERROR R-T HWP%d' % hwp)
                                  for hwp in range(nhwp)])
            r_var = np.array([self.datain.imageget('VAR R HWP%d' % hwp)
                              for hwp in range(nhwp)])
            t_var = np.array([self.datain.imageget('VAR T HWP%d' % hwp)
                              for hwp in range(nhwp)])
        else:
            rmt_data = None
            rmt_sigma = None
            r_var = None
            t_var = None

        table = [self.datain.tableget('TABLE HWP%d' % hwp)
                 for hwp in range(nhwp)]
        hwplist = [(table[hwp].field('HWP Angle')[0], hwp)
                   for hwp in range(nhwp)]

        # Always Compute Stokes I
        stokes_i = rpt_data.sum(axis=0) / float(nhwp)
        if nhwp == 1:
            err_i = rpt_sigma[0]
        else:
            # if more than 1 HWP, use std. dev. of fluxes (hwpsttedv)
            # or mean of sigmas (meansigma)
            if erripolmethod == 'hwpstddev':
                err_i = rpt_data.std(ddof=1, axis=0)
            else:
                err_i = np.sqrt((rpt_sigma ** 2).sum(axis=0)
                                / float(nhwp ** 2))

        # inflate errors in I
        if erri_inflate == 'median':
            med = np.nanmedian(err_i.flatten())
            mask = np.where(err_i < med)
            err_i[mask] = med
        elif erri_inflate == 'mean':
            mean = np.nanmean(err_i.flatten())
            mask = np.where(err_i < mean)
            err_i[mask] = mean

        # Check for angle pairs
        if nhwp != 1:
            # method using pairs 1/3 and 2/4 of HWP angles

            hwpinit = hwplist[0][0]

            # Write header keyword for the 'actual'
            # value of the initial HWP angle
            self.dataout.setheadval("HWPINIT", hwpinit,
                                    'Actual value of the initial HWP angle')

            # check HWP angles
            if override_hwp:
                # assume the currently most common observing method:
                # Stokes Q are the first two angles,
                # Stokes U are the next two
                qidx1 = [hwplist[i][1] for i in range(0, nhwp, 4)]
                qidx2 = [hwplist[i][1] for i in range(1, nhwp, 4)]
                uidx1 = [hwplist[i][1] for i in range(2, nhwp, 4)]
                uidx2 = [hwplist[i][1] for i in range(3, nhwp, 4)]
            else:
                # otherwise, sort by value
                sort_hwp = sorted(hwplist)
                log.debug('Sorted HWP list: {}'.format(sort_hwp))
                qidx1 = [sort_hwp[i][1] for i in range(0, nhwp, 4)]
                qidx2 = [sort_hwp[i][1] for i in range(2, nhwp, 4)]
                uidx1 = [sort_hwp[i][1] for i in range(1, nhwp, 4)]
                uidx2 = [sort_hwp[i][1] for i in range(3, nhwp, 4)]
            for i in range(nhwp // 2 - 1):
                log.info('')
                log.info('Stokes Q:')
                log.info('HWP indices are: %d, %d' %
                         (qidx1[i], qidx2[i]))
                log.info('Values are: %.1f, %.1f' %
                         (hwplist[qidx1[i]][0], hwplist[qidx2[i]][0]))

                diff = abs(hwplist[qidx1[i]][0] - hwplist[qidx2[i]][0])
                if abs(diff - 45) > hwp_tol:
                    log.warning('Stokes Q: HWP angles differ '
                                'by %.1f degrees (should be 45)' % diff)
                if qidx2[i] - qidx1[i] != 1:
                    # warn for data sets not taken next to each other
                    log.warning('Unexpected indices for Stokes Q '
                                'angles. Check HWP angle timestream.')

                log.info('')
                log.info('Stokes U:')
                log.info('HWP indices are: %d, %d' %
                         (uidx1[i], uidx2[i]))
                log.info('Values are: %.1f, %.1f' %
                         (hwplist[uidx1[i]][0], hwplist[uidx2[i]][0]))

                diff = abs(hwplist[uidx1[i]][0] - hwplist[uidx2[i]][0])
                if abs(diff - 45) > hwp_tol:
                    log.warning('Stokes U: HWP angles differ '
                                'by %.1f degrees (should be 45)' % diff)
                if uidx2[i] - uidx1[i] != 1:
                    # warn for data sets not taken next to each other
                    log.warning('Unexpected indices for Stokes U '
                                'angles. Check HWP angle timestream.')

            # Compute Stokes Parameters
            stokes_q, err_q, cov_qi = self.stokes(qidx1, qidx2,
                                                  rmt_data, rmt_sigma,
                                                  r_var, t_var)

            stokes_u, err_u, cov_ui = self.stokes(uidx1, uidx2,
                                                  rmt_data, rmt_sigma,
                                                  r_var, t_var)
        else:
            stokes_q, err_q, cov_qi = None, None, None
            stokes_u, err_u, cov_ui = None, None, None

        # Assigning pixels to NaNs
        badmask = self.datain.imageget('BAD PIXEL MASK')

        # Option to reject R1 subarray (assign to strictly
        # bad pixels) or not
        if remove_r1stokesi:
            badmask[:, 32:] = 3

        # Stokes I pixels with strictly bad pixels
        # (mask = 3) are assigned to NaNs
        badpix_i = np.where(badmask > 2)
        stokes_i[badpix_i] = np.nan
        err_i[badpix_i] = np.nan

        if nhwp != 1:
            # Stokes Q/U pixels with bad and widow
            # pixels (mask != 0) are assigned to NaNs
            badpix_qu = np.where(badmask != 0)
            stokes_q[badpix_qu] = np.nan
            err_q[badpix_qu] = np.nan
            cov_qi[badpix_qu] = np.nan

            stokes_u[badpix_qu] = np.nan
            err_u[badpix_qu] = np.nan
            cov_ui[badpix_qu] = np.nan

            # Since T1 is not present, for pol-nod the second half
            # of the array is ALWAYS assigned to NaNs
            stokes_q[:, 32:] = np.nan
            err_q[:, 32:] = np.nan

            stokes_u[:, 32:] = np.nan
            err_u[:, 32:] = np.nan

        # Write out images
        self.dataout.imageset(stokes_i, "STOKES I")
        self.dataout.imageset(err_i, "ERROR I")

        tbhdu = self.dataout.tablemergetables(table)
        if nhwp != 1:
            self.dataout.imageset(stokes_q, "STOKES Q")
            self.dataout.imageset(err_q, "ERROR Q")

            self.dataout.imageset(stokes_u, "STOKES U")
            self.dataout.imageset(err_u, "ERROR U")

            self.dataout.imageset(cov_qi, "COVAR Q I")
            self.dataout.imageset(cov_ui, "COVAR U I")
            self.dataout.imageset(np.zeros_like(cov_qi), "COVAR Q U")

        self.dataout.tableset(tbhdu.data, 'Table Data', tbhdu.header)

        self.dataout.copydata(self.datain, 'Bad Pixel Mask')
