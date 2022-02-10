# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Mix-in class for mapping utilities."""

from astropy import wcs as astwcs
from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.toolkit.utilities.func import stack

__all__ = ['BaseMap']


class BaseMap(object):
    """
    Mapping utilities for pipeline steps.

    This class is designed to be used as mix-in for any pipeline step
    class that requires mapping functionality.  It is assumed that the
    pipeline step will handle data input and output, as well as appropriate
    parameters.
    """
    def __init__(self):
        # placeholders for necessary attributes, to be
        # defined in pipeline steps or assigned by local methods
        self.datain = None
        self.dataout = None
        self.nhwp = 0
        self.pdata = []
        self.allhead = None
        self.pmap = {}

    def getarg(self, parname):
        """
        Get input parameter.

        This should be implemented by the pipeline step.
        """
        raise NotImplementedError

    def addgap(self, data, x, y):
        """
        Adjust x and y pixel coordinates for the gap between subarrays.

        Header keywords ALNGAPX, ALNGAPY, and ALNROTA are used to
        determine the gap parameters.  The x and y arrays are updated
        in place.

        Parameters
        ----------
        data : DataFits
            Input data to adjust
        x : array-like
            X coordinates for the input image.  Must be
            2-dimensional (nrow x ncol).
        y : array-like
            Y coordinates for the input image.  Must be
            2-dimensional (nrow x ncol).
        """
        try:
            gapx = data.getheadval("alngapx", errmsg=False)
        except KeyError:
            msg = 'Missing ALNGAPX, assuming equal to 0.0'
            log.debug(msg)
            gapx = 0.0

        try:
            gapy = data.getheadval("alngapy", errmsg=False)
        except KeyError:
            msg = 'Missing ALNGAPY, assuming equal to 0.0'
            log.debug(msg)
            gapy = 0.0

        try:
            gapangle = np.radians(data.getheadval("alnrota", errmsg=False))
        except KeyError:
            msg = 'Missing ALNROTA, assuming equal to 0.0'
            log.debug(msg)
            gapangle = 0.0

        if np.allclose([gapx, gapy, gapangle], 0):
            return

        # we assume that the sub-arrays are split across columns in the middle.
        nrow, ncol = x.shape
        ncol = ncol // 2

        # pixel coordinates of sub-array, relative to center
        suby, subx = np.mgrid[0:nrow, 0:ncol]

        # center of data is (0,0)
        subx = subx - ncol / 2. + 0.5
        suby = suby - nrow / 2. + 0.5

        # rotate sub-array by gapangle counterclockwise
        tmp = np.cos(gapangle) * subx - np.sin(gapangle) * suby
        suby = np.sin(gapangle) * subx + np.cos(gapangle) * suby
        subx = tmp

        # shift by gapx and gapy
        suby += gapy
        subx += gapx

        # revert to 0,0 being bottom left
        subx = subx - 0.5 + ncol / 2.
        suby = suby - 0.5 + nrow / 2.

        # update coordinates of right half of x,y
        x[:, ncol:] = subx + ncol
        y[:, ncol:] = suby

    def checkvalid(self):
        """
        Check data consistency.

        All input data are checked to make sure that SPECTEL1
        and the number of HWP angles are the same across all input
        files.  The number of HWP angles found is set as self.nhwp.
        """

        log.debug('Started checking for valid input files')

        nwave = len(set([d.getheadval('SPECTEL1') for d in self.datain]))
        if nwave != 1:
            msg = "SPECTEL1 not the same among all input files!"
            log.error(msg)
            raise ValueError(msg)

        nhwp = set([d.getheadval('nhwp') for d in self.datain])
        if len(nhwp) > 1:
            msg = "Number of HWP angles not the same among all input files!"
            log.error(msg)
            raise ValueError(msg)
        self.nhwp = list(nhwp)[0]
        log.debug('Finished checking valid input files')

    def sumexptime(self):
        """
        Sum exposure time over input data.

        The EXPTIME keyword in self.dataout is set to the summed value
        as the total exposure time on-source.
        """
        log.debug('Started summing exposure times for input files')
        totexptime = sum([d.getheadval('EXPTIME') for d in self.datain])
        self.dataout.setheadval('EXPTIME', totexptime,
                                'Total on-source exposure time [s]')
        log.debug('Finished summing exposure times for input files')

    def read_resample_data(self, maxgoodpix):
        """
        Read data from input images.

        This function ingests the data from self.datain, storing in
        the self.pdata list of dictionaries.  Keys for each data
        dictionary are as follows; values are flattened arrays
        corresponding to good pixels only.

            - mask: bad pixel mask values
            - ra: RA coordinates
            - dec: Dec coordinates
            - I: Stokes I flux values
            - dI: Stokes I error values

        The following additional keys are present only if self.nhwp > 1:
            - Q: Stokes Q flux values
            - dQ: Stokes Q error values
            - U: Stokes U flux values
            - dU: Stokes U error values
            - QIcov: Q-I covariance values
            - UIcov: U-I covariance values
            - QUcov: Q-U covariance values

        Parameters
        ----------
        maxgoodpix : int
            The maximum mask value to be considered good. If maxgoodpix=0,
            only good pixels present in both R and T arrays are used for
            stokes I. If maxgoodpix=2, widow pixels are also used for
            stokes I.
        """

        log.debug("starting read_resample_data()")

        # We create self.pdata.  pdata will be a list of python
        # dictionaries to contain all the data.
        self.pdata = []

        for i, f in enumerate(self.datain):
            temp = {}
            naxis1, naxis2 = f.getheadval('NAXIS1'), f.getheadval('NAXIS2')
            pix2, pix1 = np.mgrid[0:naxis2, 0:naxis1]
            self.addgap(f, pix1, pix2)
            tempwcs = astwcs.WCS(f.header)
            ra, dec = tempwcs.wcs_pix2world(pix1, pix2, 0)

            # convert the mask to a float so we can keep track of fractional
            # overlap based on gaussian smoothing
            if 'BAD PIXEL MASK' in f.imgnames:
                temp['mask'] = f.imageget('BAD PIXEL MASK').astype(np.float64)
            else:
                temp['mask'] = np.zeros((naxis2, naxis1), dtype=np.float64)

            ngood = np.where(temp['mask'] <= maxgoodpix)
            temp['mask'] = temp['mask'][ngood]

            temp['ra'] = ra[ngood]
            temp['dec'] = dec[ngood]

            if 'STOKES I' in f.imgnames:
                temp['I'] = f.imageget('STOKES I').astype(np.float64)[ngood]
                temp['dI'] = f.imageget('ERROR I').astype(np.float64)[ngood]
            else:
                temp['I'] = f.imageget('PRIMARY IMAGE').astype(
                    np.float64)[ngood]
                temp['dI'] = f.imageget('NOISE').astype(np.float64)[ngood]

            if self.nhwp > 1:
                # polarization
                for var in ('Q', 'U'):
                    temp[var] = f.imageget(
                        'STOKES %s' % var).astype(np.float64)[ngood]
                    temp['d' + var] = f.imageget(
                        'ERROR %s' % var).astype(np.float64)[ngood]

                temp['QIcov'] = f.imageget(
                    'COVAR Q I').astype(np.float64)[ngood]
                temp['UIcov'] = f.imageget(
                    'COVAR U I').astype(np.float64)[ngood]
                temp['QUcov'] = f.imageget(
                    'COVAR Q U').astype(np.float64)[ngood]

            log.debug('storing {} good pixels '
                      '(out of {})'.format(ngood[0].size,
                                           naxis1 * naxis2))

            self.pdata.append(temp)

        log.debug("finished read_resample_data()")

    def make_resample_map(self, cdelt):
        """
        Construct map parameters from input data.

        The coordinate grid for the map is determined from the
        range of the RA and Dec coordinates in the input data.
        The grid and associated reference values are stored in
        self.pmap, for use by the resampling algorithm.  A draft
        header containing appropriate WCS information for the output
        map is stored in self.allhead.

        The 'proj' parameter is used to set the projection type
        in the output header.  The 'sizelimit' argument is tested
        against, to determine if the output map is too large.  This
        error case is most likely due to inappropriately grouped
        input files.  These parameters must be defined by the pipeline
        step inheriting from this class.

        Parameters
        ----------
        cdelt : float
            The pixel scale for the output map, in arcsec.
        """

        log.debug("starting make_resample_map()")
        proj = self.getarg('proj')

        ra = np.hstack([d['ra'] for d in self.pdata])
        dec = np.hstack([d['dec'] for d in self.pdata])
        base_ra = np.mean(ra)
        base_dec = np.mean(dec)

        # offsets in arcsec, with RA reversed
        xs = -1 * (ra - base_ra) * np.cos(np.radians(base_dec)) * 3600.
        ys = (dec - base_dec) * 3600.
        coordinates = stack(xs, ys)

        xmin = np.nanmin(xs)
        xmax = np.nanmax(xs)
        ymin = np.nanmin(ys)
        ymax = np.nanmax(ys)
        xrange = xmax - xmin
        yrange = ymax - ymin

        # size of x and y axes in pixels
        naxis1 = int(np.ceil(xrange / cdelt))
        naxis2 = int(np.ceil(yrange / cdelt))

        # check for too large map
        limit = self.getarg('sizelimit')
        if naxis1 > limit or naxis2 > limit:
            msg = "Output map is too large (> %.0f pixels)" % limit
            log.error(msg)
            raise ValueError(msg)

        # output grid in arcsec offsets
        xout = np.arange(naxis1, dtype=float) * cdelt + xmin
        yout = np.arange(naxis2, dtype=float) * cdelt + ymin
        grid = xout, yout

        # update crval to match an actual output pixel
        crval2 = base_dec + (yout[naxis2 // 2] / 3600.)
        crval1 = base_ra - (xout[naxis1 // 2]
                            / (3600. * np.cos(np.radians(crval2))))

        # draft an output header
        allhead = fits.Header()
        allhead['SIMPLE'] = 'T'
        allhead['BITPIX'] = -64

        allhead['NAXIS'] = 2
        allhead['NAXIS1'] = naxis1
        allhead['NAXIS2'] = naxis2
        allhead['CDELT1'] = -1 * cdelt / 3600.
        allhead['CDELT2'] = cdelt / 3600.
        allhead['CRVAL1'] = crval1
        allhead['CRVAL2'] = crval2
        allhead['CRPIX1'] = naxis1 // 2 + 1
        allhead['CRPIX2'] = naxis2 // 2 + 1
        allhead['CTYPE1'] = 'RA---%s' % proj
        allhead['CTYPE2'] = 'DEC--%s' % proj
        allhead['EQUINOX'] = self.datain[0].getheadval('EQUINOX')
        allhead['NHWP'] = self.nhwp

        log.debug("make_resample_map: "
                  "naxis1,naxis2 = %d,%d" % (naxis1, naxis2))
        self.allhead = allhead

        # set the grid information in self.pmap
        self.pmap = {
            'base_ra': base_ra, 'base_dec': base_dec,
            'xmin': xmin, 'xmax': xmax,
            'ymin': ymin, 'ymax': ymax,
            'shape': (ra.size, naxis2, naxis1),
            'xout': xout, 'yout': yout,
            'xrange': xrange, 'yrange': yrange,
            'delta': (cdelt, cdelt),
            'coordinates': coordinates,
            'grid': grid}

        log.debug("finished make_resample_map()")
