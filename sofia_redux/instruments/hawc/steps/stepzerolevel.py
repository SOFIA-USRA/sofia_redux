# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Zero level correction pipeline step."""

import re

from astropy import log
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
import numpy as np
from scipy.ndimage import median_filter
from scipy.ndimage.filters import generic_filter

from sofia_redux.instruments.hawc.stepparent import StepParent

__all__ = ['StepZeroLevel']


class StepZeroLevel(StepParent):
    """
    Correct zero level for scanning data.

    This step applies an optional zero-level correction to the
    Stokes I image for scan mode imaging data, using user input or a
    mean- or median-filter to identify the background region in the
    image.

    Input for this step must be a single DataFits that contains
    3 image planes (HDUs) for the total Stokes I flux.  The three images are:
    DATA, ERROR, and EXPOSURE.

    Output from this step has the same format as the input.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'zerolevel', and are named with
        the step abbreviation 'ZLC'.

        Parameters defined for this step are:

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
            applied to image, with the radius specified by the
            zero_level_radius parameter.  The lowest negative local
            average is assumed to be the zero level.
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
        self.name = 'zerolevel'
        self.description = 'Correct Zero Level'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'zlc'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['zero_level_method', 'none',
                               'Statistic for zero-level calculation '
                               '(mean, median, none)'])
        self.paramlist.append(['zero_level_region', 'header',
                               'Zero level region method (header, auto, '
                               'or [RA, Dec, radius] in '
                               'degrees).'])
        self.paramlist.append(['zero_level_radius',
                               [9.68, 15.6, 15.6, 27.2, 36.4],
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

    def correct_zero_level_auto(self, data, method, radius):
        """
        Correct image zero level from automatically determined regions.

        Data array is updated in place.

        Parameters
        ----------
        data : array-like
            Data to correct.
        method : {'mean', 'median'}
            Filter method.
        radius : int
            Radius in pixels for filter kernel.
        """
        # circular aperture kernel
        kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        kernel[x ** 2 + y ** 2 <= radius ** 2] = 1

        # mean value and error within aperture at each point,
        # ignoring any regions containing NaNs
        if 'mean' in method:
            d_filter = generic_filter(
                data, np.mean, footprint=kernel,
                mode='constant', cval=np.nan)
        else:
            # median filter is much faster than generic filter
            # with median function
            d_filter = median_filter(
                data, footprint=kernel,
                mode='constant', cval=np.nan)

            # account for NaNs that the median filter ignores
            def func(data):
                return np.any(np.isnan(data))

            d_nan = generic_filter(
                data, func, footprint=kernel,
                mode='constant', cval=1)
            d_filter[d_nan > 0] = np.nan

        # check for negative regions
        neg = d_filter < 0
        if np.any(neg):
            # find lowest region
            zero_pix = np.nanargmin(d_filter)
            zero_pix = np.unravel_index(zero_pix, data.shape)
            log.info(f'Correcting zero level '
                     f'at pix x,y: {zero_pix[1]},{zero_pix[0]}')

            # subtract zero level from filter image at the determined pixel
            zero = d_filter[zero_pix]
            log.info(f'  Zero level: {zero}')
            data -= zero
        else:
            log.info('No negative zero level found; not '
                     'subtracting background.')

    def correct_zero_level_region(self, data, method, region, refheader,
                                  robust=5.0):
        """
        Correct image zero level from specified circular regions.

        Data array is updated in place.

        Parameters
        ----------
        data : array-like
            Data to correct.
        method : {'mean', 'median'}
            Statistics method.
        region : list of float
            Region specified as [RA, Dec, radius] in degrees.
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
        ref_wcs = WCS(refheader)

        # coordinates for region check
        ny, nx = data.shape
        y, x = np.ogrid[0:ny, 0:nx]

        # region center and radius
        cx, cy = ref_wcs.wcs_world2pix(region[0], region[1], 0)
        cr = region[2] / np.abs(ref_wcs.wcs.cdelt[0])

        # check that region center is in the array
        if (not np.all(np.isfinite([cx, cy]))
                or (cx < 0 or cx >= nx or cy < 0 or cy >= ny)):
            msg = f"Region {region}, center {cx},{cy} is not " \
                  f"on array with size x,y={nx},{ny}"
            log.error(msg)
            raise ValueError(msg)

        test = ((x - cx) ** 2 + (y - cy) ** 2 <= cr ** 2)
        if robust > 0:
            # sigma clip data before taking stats
            dmed, dmean, dsig = sigma_clipped_stats(data[test], sigma=robust)
            if 'mean' in method:
                zero = dmean
            else:
                zero = dmed
        else:
            # just take stats
            if 'mean' in method:
                zero = np.nanmean(data[test])
            else:
                zero = np.nanmedian(data[test])

        # now subtract level from image
        log.info(f'Correcting zero level at pix x,y: {cx:.2f},{cy:.2f}')
        log.info(f'  Zero level: {zero}')
        data -= zero

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object. The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Check and gather all input data.
        2. Correct zero level if desired.
        """
        # copy input to output
        self.dataout = self.datain.copy()
        header = self.dataout.header
        data = self.dataout.image

        # get zero-level parameters
        zl_method = str(self.getarg('zero_level_method')).lower()
        if zl_method not in ['mean', 'median']:
            log.info("Method is not 'mean' or 'median'; "
                     "no zero level correction attempted.")

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

        # check for zero level keywords in the header
        zl_reg_list = None
        if zl_method in ['mean', 'median']:
            if str(zl_region) == 'header':
                try:
                    zra = header['ZERO_RA']
                    zdec = header['ZERO_DEC']
                    zrad = header['ZERO_RAD']
                    zl_reg_list = [zra, zdec, zrad]
                except KeyError:
                    log.warning('Missing zero-level region keys '
                                '(ZERO_RA, ZERO_DEC, ZERO_RAD).')
                    log.warning('Falling back to auto region method.')
                    zl_reg_list = None
                else:
                    # catch missing values
                    if np.any(np.isclose(zl_reg_list, -9999)):
                        log.warning('Missing zero-level region keys '
                                    '(ZERO_RA, ZERO_DEC, ZERO_RAD).')
                        log.warning('Falling back to auto region method.')
                        zl_reg_list = None
            elif isinstance(zl_region, list):
                zl_reg_list = zl_region

        # Correct for zero level from specified region or
        # lowest average value within a specified window
        if zl_method in ['mean', 'median']:
            do_auto = True
            if zl_reg_list is not None:
                try:
                    self.correct_zero_level_region(data, zl_method,
                                                   zl_reg_list, header,
                                                   robust=zl_sigma)
                except ValueError:
                    log.warning('Not applying zero level correction '
                                'from specified regions; falling back '
                                'to auto method.')
                else:
                    do_auto = False
            if do_auto:
                cdelt = np.abs(3600. * header.get('CDELT1', 1.0))
                radius = int(np.round(zl_radius / cdelt))
                self.correct_zero_level_auto(data, zl_method, radius)
        else:
            log.debug('No zero level correction attempted.')

        # Write out image
        self.dataout.imageset(data)
