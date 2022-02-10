# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Scanning mode Stokes parameters pipeline step."""

import re

from astropy import log
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
import numpy as np
from scipy.ndimage import median_filter
from scipy.ndimage.filters import generic_filter

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.stepparent import StepParent
from sofia_redux.toolkit.image import adjust

__all__ = ['StepScanStokes']


class StepScanStokes(StepParent):
    """
    Compute Stokes parameters for scanning polarimetry data.

    This step derives Stokes I, Q, and U images with associated
    uncertainties and covariances from R and T array images.

    Since the input data was produced by the scan map algorithm, it has
    already been resampled into sky coordinates. The R and T images for each
    HWP angle must be shifted into a common reference frame before
    addition and subtraction. Shift values are determined from the
    WCS values recorded in the extension headers, as output by
    `sofia_redux.instruments.hawc.steps.StepScanMapPol`. Shift interpolations
    are performed via `sofia_redux.toolkit.image.adjust.shift`.

    Optionally, a zero-level correction may be applied to the R and
    T images, using a mean- or median-filter to identify the lowest
    negative region in the image.

    Thereafter, R and T arrays are directly added and subtracted
    to produce Stokes parameter fluxes, as in the standard Stokes
    algorithm for chop-nod polarimetry data (see
    `sofia_redux.instruments.hawc.steps.StepStokes`).

    Input for this step must be a single DataFits that contains
    3 image planes (HDUs) for each subarray (R0 and T0), at each of
    4 HWP angles. The three images are: DATA, ERROR, and EXPOSURE.
    Output from this step is a DataFits with the following image
    extensions: STOKES I, ERROR I, STOKES Q, ERROR Q, STOKES U,
    ERROR U, COVAR Q I, COVAR U I, COVAR Q U.
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
        zero_level_method : {'mean', 'median', 'none'}
            Statistic for zero-level calculation. If 'none', the
            zero-level will not be corrected.  For the other
            options, either a mean or median statistic will be used to
            determine the zero-level value from the region set by the
            region parameters.
        zero_level_region : str
            If set to 'header', the zero-level region will be determined
            from the ZERO_RA, ZERO_DEC, ZERO_RAD keywords
            (for RA center, Dec center, and radius, respectively).
            If set to 'auto', a mean- or median-filter will be
            applied to the R and T images, with the radius specified by the
            zero_level_radius parameter.  The lowest negative local
            average that is negative in both R and T for all HWP angles
            is assumed to be the zero level.  R and T values are applied
            separately, from the value of the average at the same pixel.
            Otherwise, a region may be directly provided as a list of
            [RA center, Dec center, radius], in degrees.
        zero_level_radius : list of float
            Filter radius for zero-level calculation, in arcseconds
            (per band).  Used only for zero_level_region = 'auto'.
        zero_level_sigma : float
            Sigma value for statistics clipping.  Ignored for
            zero_level_region = 'auto'.
        """
        # Name of the pipeline reduction step
        self.name = 'scanstokes'
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
        self.paramlist.append(['zero_level_method', 'none',
                               'Statistic for zero-level calculation '
                               '(mean, median, none)'])
        self.paramlist.append(['zero_level_region', 'header',
                               'Zero level region method (header, auto, '
                               'or [RA, Dec, radius] in '
                               'degrees).'])
        self.paramlist.append(['zero_level_radius',
                               [4.84, 7.80, 7.80, 13.6, 18.2],
                               'Filter radius for zero-level calculation '
                               'in auto mode.'])
        self.paramlist.append(['zero_level_sigma', 5.0,
                               'Sigma value for statistics clipping in '
                               'non-auto mode.'])

    def read_radius(self):
        """
        Read a radius value from the parameters.

        The parameters are expected to be defined as a list, with
        one entry for each HAWC band.  The correct value for the
        input data is selected from the list.

        Returns
        -------
        radius : float
            Radius value for the input data.
        """
        radius = self.getarg('zero_level_radius')
        waveband = self.datain.getheadval('spectel1')
        bands = ['A', 'B', 'C', 'D', 'E']
        try:
            idx = bands.index(waveband[-1])
        except (ValueError, IndexError):
            # waveband not in list
            msg = 'Cannot parse waveband: %s' % waveband
            log.error(msg)
            raise ValueError(msg)
        try:
            radius = radius[idx]
        except IndexError:
            msg = 'Missing radius values for all wavebands'
            log.error(msg)
            raise IndexError(msg)

        return radius

    def stokes(self, idx1, idx2, rmt_data, rmt_sigma,
               r_var, t_var):
        """
        Compute stokes Q and U.

        The index parameters control which Stokes parameter image
        is computed.

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

    def wcs_shift(self, header, refheader):
        """
        Calculate the WCS shift between two headers.

        Parameters
        ----------
        header : fits.Header
            The header for the data to shift.
        refheader : fits.Header
            The reference header to shift to.

        Returns
        -------
        shift : array-like
            The (x, y) shift to apply to the data associated with
            header (dx = shift[0], dy = shift[1]).
        """
        wcs_in = WCS(header)
        wcs_out = WCS(refheader)
        xy = [refheader['CRPIX1'], refheader['CRPIX2']]
        xyin = np.array([xy])
        wxy = wcs_in.wcs_pix2world(xyin, 1)
        xyin_on_ref = wcs_out.wcs_world2pix(wxy, 1)
        shift = (xyin_on_ref - xyin)[0]
        return shift

    def _expand_array(self, data, shape):
        """
        Expand an image array to a new shape.

        Data dimensions must be smaller than the new shape.
        Only two dimensions are expected.

        data : array-like
            The data to expand.
        shape : tuple of int
            The shape to expand to.
        """
        result = np.full(shape, np.nan)
        s0 = data.shape
        result[0: s0[0], 0: s0[1]] = data.copy()
        return result

    def correct_zero_level_auto(self, r_data, t_data, method, radius):
        """
        Correct image zero level from automatically determined regions.

        Data arrays are updated in place.

        Parameters
        ----------
        r_data : array-like
            R data to correct.
        t_data : array_like
            T data corresponding to r_data.
        method : {'mean', 'median'}
            Filter method.
        radius : int
            Radius in pixels for filter kernel.
        """
        # circular aperture kernel
        kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        kernel[x ** 2 + y ** 2 <= radius ** 2] = 1

        r_filters = []
        t_filters = []
        for r, t in zip(r_data, t_data):
            # mean value and error within aperture at each point,
            # ignoring any regions containing NaNs
            if 'mean' in method:
                r_filter = generic_filter(
                    r, np.mean, footprint=kernel,
                    mode='constant', cval=np.nan)
                t_filter = generic_filter(
                    t, np.mean, footprint=kernel,
                    mode='constant', cval=np.nan)
            else:
                # median filter is much faster than generic filter
                # with median function
                r_filter = median_filter(
                    r, footprint=kernel,
                    mode='constant', cval=np.nan)
                t_filter = median_filter(
                    t, footprint=kernel,
                    mode='constant', cval=np.nan)

                # account for NaNs that the median filter ignores
                def func(data):
                    return np.any(np.isnan(data))

                r_nan = generic_filter(
                    r, func, footprint=kernel,
                    mode='constant', cval=1)
                t_nan = generic_filter(
                    t, func, footprint=kernel,
                    mode='constant', cval=1)
                r_filter[r_nan > 0] = np.nan
                t_filter[t_nan > 0] = np.nan

            r_filters.append(r_filter)
            t_filters.append(t_filter)
        r_filters = np.array(r_filters)
        t_filters = np.array(t_filters)

        # check for negative regions in either R or T at all
        # HWP positions
        neg = np.all(r_filters < 0, axis=0) | np.all(t_filters < 0, axis=0)
        if np.any(neg):
            # block regions that aren't generally non-negative
            block = np.tile(~neg, (r_filters.shape[0], 1, 1))
            r_filters[block] = np.nan
            t_filters[block] = np.nan

            # find collectively lowest region in R and T across all HWP
            zero_pix = np.nanargmin(np.sum(r_filters, axis=0)
                                    + np.sum(t_filters, axis=0))
            zero_pix = np.unravel_index(zero_pix, r_data[0].shape)
            log.info(f'Correcting zero level '
                     f'at pix x,y: {zero_pix[1]},{zero_pix[0]}')

            # subtract zero level from individual filter images
            # at the determined pixel
            for r, t, rf, tf in zip(r_data, t_data, r_filters, t_filters):
                rzero = rf[zero_pix]
                log.info(f'  R level: {rzero}')
                tzero = tf[zero_pix]
                log.info(f'  T level: {tzero}')
                r -= rzero
                t -= tzero
        else:
            log.info('No negative zero level found; not '
                     'subtracting background.')

    def correct_zero_level_region(self, r_data, t_data, method,
                                  reglist, refheader, robust=5.0):
        """
        Correct image zero level from specified circular regions.

        Data arrays are updated in place.

        Parameters
        ----------
        r_data : array-like
            R data to correct.
        t_data : array_like
            T data corresponding to r_data.
        method : {'mean', 'median'}
            Statistics method.
        reglist : list of list of float
            List of regions as [RA, Dec, radius] in degrees,
            matching length of r_data and t_data lists.
        refheader : astropy.Header
            Reference header to use for WCS.
        robust : float
            Sigma value to use for clipping statistics.  Set to 0
            to turn off clipping.

        Raises
        ------
        ValueError
            If any specified region is not on the array.
        """
        # reference WCS for identifying background pixels
        # This should be run after shifting images, so that
        # all are in the same reference frame.
        ref_wcs = WCS(refheader)

        # coordinates for region check
        ny, nx = r_data[0].shape
        y, x = np.ogrid[0:ny, 0:nx]

        # iterate first to collect levels; correct later only
        # if all regions are appropriate
        r_levels = []
        t_levels = []
        for r, t, reg in zip(r_data, t_data, reglist):
            # region center and radius
            cx, cy = ref_wcs.wcs_world2pix(reg[0], reg[1], 0)
            cr = reg[2] / np.abs(ref_wcs.wcs.cdelt[0])

            # check that region center is in the array
            if (not np.all(np.isfinite([cx, cy]))
                    or (cx < 0 or cx >= nx or cy < 0 or cy >= ny)):
                msg = f"Region {reg}, center {cx},{cy} is not " \
                      f"on array with size x,y={nx},{ny}"
                log.error(msg)
                raise ValueError(msg)

            test = ((x - cx) ** 2 + (y - cy) ** 2 <= cr ** 2)
            if robust > 0:
                # sigma clip data before taking stats
                rmed, rmean, rsig = sigma_clipped_stats(r[test], sigma=robust)
                tmed, tmean, tsig = sigma_clipped_stats(t[test], sigma=robust)
                if 'mean' in method:
                    rzero = rmean
                    tzero = tmean
                else:
                    rzero = rmed
                    tzero = tmed
            else:
                # just take stats
                if 'mean' in method:
                    rzero = np.nanmean(r[test])
                    tzero = np.nanmean(t[test])
                else:
                    rzero = np.nanmedian(r[test])
                    tzero = np.nanmedian(t[test])

            log.info(f'Correcting zero level at pix x,y: {cx:.2f},{cy:.2f}')
            log.info(f'  R level: {rzero}')
            log.info(f'  T level: {tzero}')
            r_levels.append(rzero)
            t_levels.append(tzero)

        # now subtract levels from images
        for r, t, rl, tl in zip(r_data, t_data, r_levels, t_levels):
            r -= rl
            t -= tl

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object. The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Check and gather all input data.
        2. Shift all data to a common reference frame.
        3. Compute Stokes I from R+T at all angles.
        4. Compute Stokes Q and U from R-T at angles separated
           by 45 degrees.
        5. Propagate errors and covariances.
        """

        self.dataout = DataFits(config=self.datain.config)
        self.dataout.filename = self.datain.filename
        self.dataout.setheader(self.datain.header)

        # input: R and T data at each HWP are rotated to north up, and
        #   sampled onto the same scale, but are shifted relative to
        #   each other, as recorded in the WCS in each extension header.
        #   HWP angle value and subarray are recorded in extension headers.

        # Number of HWP angles must be 4.
        nhwp = self.datain.getheadval('nhwp')
        if nhwp != 4:
            msg = 'Unexpected number of HWP angles. Must be NWHP=4.'
            log.error(msg)
            raise ValueError(msg)

        # get expected tolerance and difference
        hwp_tol = abs(self.getarg('hwp_tol'))
        hwp_diff = 45.0

        # get zero-level parameters
        zl_method = str(self.getarg('zero_level_method')).lower()
        zl_sigma = self.getarg('zero_level_sigma')
        zl_radius = self.read_radius()
        zl_region = self.getarg('zero_level_region')
        if str(zl_region) not in ['header', 'auto']:
            try:
                if type(zl_region) is str:
                    region = [re.sub(r'[\'"\[\]]', '', v).strip()
                              for v in str(zl_region).split(',')]
                else:
                    region = zl_region
                assert len(region) == 3
                region = [float(r) for r in region]
            except (TypeError, ValueError, AssertionError):
                msg = f'Badly formatted zero_level_region: {zl_region}. ' \
                      f'Should be [RA, Dec, radius] in degrees.'
                log.error(msg)
                raise ValueError(msg) from None
            zl_region = region

        hwplist = []
        data_shape = [0, 0]
        r_data = []
        t_data = []
        r_sigma = []
        t_sigma = []
        r_exp = []
        t_exp = []
        r_head = []
        t_head = []
        df = self.datain
        for i in range(nhwp):
            rd_ext = 'DATA R HWP%d' % i
            td_ext = 'DATA T HWP%d' % i
            re_ext = 'ERROR R HWP%d' % i
            te_ext = 'ERROR T HWP%d' % i
            rx_ext = 'EXPOSURE R HWP%d' % i
            tx_ext = 'EXPOSURE T HWP%d' % i

            hwplist.append(df.getheadval('HWPINIT', dataname=rd_ext))

            rimg = df.imageget(rd_ext)
            data_shape[0] = max(data_shape[0], rimg.shape[0])
            data_shape[1] = max(data_shape[1], rimg.shape[1])

            timg = df.imageget(td_ext)
            data_shape[0] = max(data_shape[0], timg.shape[0])
            data_shape[1] = max(data_shape[1], timg.shape[1])

            r_data.append(rimg)
            t_data.append(timg)

            r_sigma.append(df.imageget(re_ext))
            t_sigma.append(df.imageget(te_ext))

            r_exp.append(df.imageget(rx_ext))
            t_exp.append(df.imageget(tx_ext))

            r_head.append(df.getheader(rd_ext))
            t_head.append(df.getheader(td_ext))

        # set HWPINIT for the file
        self.dataout.setheadval('HWPINIT', hwplist[0],
                                'Actual value of the initial HWP angle')

        # fix missing or bad HWPSTART (old scanpol files)
        try:
            hwpstart = self.dataout.getheadval('HWPSTART', errmsg=False)
        except KeyError:
            hwpstart = -9999
        if hwpstart == -9999:
            self.dataout.setheadval('HWPSTART', hwplist[0],
                                    'HWP initial angle [deg]')

        # check HWP angles
        sort_idx = np.argsort(hwplist)
        hwplist = np.array(hwplist)[sort_idx]
        diff1 = abs(hwplist[0] - hwplist[2])
        diff2 = abs(hwplist[1] - hwplist[3])
        if abs(diff1 - hwp_diff) > hwp_tol:
            log.warning('Stokes Q: HWP angles differ '
                        'by %.1f degrees '
                        '(should be %.1f)' % (diff1, hwp_diff))
        if abs(diff2 - hwp_diff) > hwp_tol:
            log.warning('Stokes U: HWP angles differ '
                        'by %.1f degrees '
                        '(should be %.1f)' % (diff2, hwp_diff))

        # shift data to reference
        refheader = df.header
        zl_reg_list = []
        for i, header in enumerate(r_head):
            # first expand all images to max size
            d = self._expand_array(r_data[i], data_shape)
            s = self._expand_array(r_sigma[i], data_shape)
            e = self._expand_array(r_exp[i], data_shape)

            shift_val = self.wcs_shift(header, refheader)
            if np.allclose(shift_val, 0.0, atol=0.01):
                r_data[i] = d
                r_sigma[i] = s
                r_exp[i] = e
            else:
                log.info('Shifting R image {} by x,y='
                         '{:.1f},{:.1f}'.format(i + 1,
                                                shift_val[0],
                                                shift_val[1]))
                # reverse x and y for shift function
                shift_val = [shift_val[1], shift_val[0]]
                r_data[i] = adjust.shift(d, shift_val)
                r_sigma[i] = adjust.shift(s, shift_val)
                r_exp[i] = adjust.shift(e, shift_val)

            # check for zero level keywords in the header:
            # will be used for both R and T
            if zl_method in ['mean', 'median']:
                if str(zl_region) == 'header':
                    try:
                        zra = header['ZERO_RA']
                        zdec = header['ZERO_DEC']
                        zrad = header['ZERO_RAD']
                        zl_reg_list.append([zra, zdec, zrad])
                    except KeyError:
                        # use the first specified region if possible
                        if len(zl_reg_list) >= 1:
                            log.debug('Using primary header region '
                                      'for zero level.')
                            zl_reg_list.append(zl_reg_list[0])
                        else:
                            log.warning('Missing zero-level region keys '
                                        '(ZERO_RA, ZERO_DEC, ZERO_RAD).')
                            log.warning('Falling back to auto region method.')
                            zl_region = 'auto'
                            zl_reg_list = []
                    else:
                        # catch missing values
                        if np.any(np.isclose(zl_reg_list, -9999)):
                            log.warning('Missing zero-level region keys '
                                        '(ZERO_RA, ZERO_DEC, ZERO_RAD).')
                            log.warning('Falling back to auto region method.')
                            zl_region = 'auto'
                            zl_reg_list = []

                elif type(zl_region) is list:
                    zl_reg_list.append(zl_region)

        for i, header in enumerate(t_head):
            # first expand all images to max size
            d = self._expand_array(t_data[i], data_shape)
            s = self._expand_array(t_sigma[i], data_shape)
            e = self._expand_array(t_exp[i], data_shape)

            shift_val = self.wcs_shift(header, refheader)
            if np.allclose(shift_val, 0.0, atol=0.01):
                t_data[i] = d
                t_sigma[i] = s
                t_exp[i] = e
            else:
                log.info('Shifting T image {} by x,y='
                         '{:.1f},{:.1f}'.format(i + 1, shift_val[0],
                                                shift_val[1]))
                shift_val = [shift_val[1], shift_val[0]]
                t_data[i] = adjust.shift(d, shift_val)
                t_sigma[i] = adjust.shift(s, shift_val)
                t_exp[i] = adjust.shift(e, shift_val)

        # Correct for zero level from specified region or
        # lowest average value within a specified window
        if zl_method in ['mean', 'median']:
            do_auto = True
            if len(zl_reg_list) == len(r_data):
                try:
                    self.correct_zero_level_region(r_data, t_data, zl_method,
                                                   zl_reg_list, refheader,
                                                   robust=zl_sigma)
                except ValueError:
                    log.warning('Not applying zero level correction '
                                'from specified regions; falling back '
                                'to auto method.')
                else:
                    do_auto = False
            if do_auto:
                cdelt = np.abs(3600. * df.getheadval('CDELT1'))
                radius = int(np.round(zl_radius / cdelt))
                self.correct_zero_level_auto(r_data, t_data, zl_method, radius)
        else:
            log.debug('No zero level correction attempted.')

        # Compute Stokes I
        # Don't use nansum here, since data must overlap to be useful
        all_data = np.array(r_data + t_data)
        all_sigma = np.array(r_sigma + t_sigma)
        stokes_i = np.sum(all_data, axis=0) / float(nhwp)
        err_i = np.sqrt(np.sum(all_sigma ** 2, axis=0) / float(nhwp ** 2))

        # Compute Stokes Q and U
        # method using R-T pairs 1/3 and 2/4 of HWP angles
        rmt_data = np.array(r_data) - np.array(t_data)

        r_var = np.array(r_sigma) ** 2
        t_var = np.array(t_sigma) ** 2
        rmt_sigma = np.sqrt(r_var + t_var)

        idx1 = [sort_idx[0]]
        idx2 = [sort_idx[2]]
        stokes_q, err_q, cov_qi = self.stokes(
            idx1, idx2, rmt_data, rmt_sigma,
            r_var, t_var)

        idx1 = [sort_idx[1]]
        idx2 = [sort_idx[3]]
        stokes_u, err_u, cov_ui = self.stokes(
            idx1, idx2, rmt_data, rmt_sigma,
            r_var, t_var)

        # bad pixel mask: set nans to bad val
        bpm = np.zeros_like(stokes_i).astype(int)
        bpm[np.isnan(stokes_i) | np.isnan(stokes_q) | np.isnan(stokes_u)] = 3

        # Write out images
        self.dataout.imageset(stokes_i, "STOKES I")
        self.dataout.imageset(err_i, "ERROR I")
        self.dataout.imageset(stokes_q, "STOKES Q")
        self.dataout.imageset(err_q, "ERROR Q")
        self.dataout.imageset(stokes_u, "STOKES U")
        self.dataout.imageset(err_u, "ERROR U")
        self.dataout.imageset(cov_qi, "COVAR Q I")
        self.dataout.imageset(cov_ui, "COVAR U I")
        self.dataout.imageset(np.zeros_like(cov_qi), "COVAR Q U")
        self.dataout.imageset(bpm, "BAD PIXEL MASK")
