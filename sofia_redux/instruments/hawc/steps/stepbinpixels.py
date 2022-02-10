# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Optional pixel binning pipeline step."""

from astropy import log
from astropy.wcs import WCS
import numpy as np

from sofia_redux.instruments.hawc.stepparent import StepParent

__all__ = ['StepBinPixels']


class StepBinPixels(StepParent):
    """
    Bin pixels to increase signal-to-noise.

    Input for this image must contain STOKES, ERROR, and COVAR images,
    as produced by the `sofia_redux.instruments.hawc.steps.StepStokes`
    pipeline step. The output DataFits has the same extensions as the
    input.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'binpixels', and are named with
        the step abbreviation 'BIN'.


        Parameters defined for this step are:

        block_size : {1, 2, 4, 8}
            Bin size, in pixels.  Must divide the 40x64 array evenly
            into square blocks.  If set to 1, no binning will be
            performed.

        """
        # Name of the pipeline reduction step
        self.name = 'binpixels'
        self.description = 'Bin Pixels'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'bin'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['block_size', 1,
                               'Bin size, in pixels. '
                               'Must divide 40x64 array evenly.'])

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object. The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Block sum detector pixels with specified binning.
        2. Propagate errors, covariances, and WCS keys.
        """
        # copy datain to dataout
        self.dataout = self.datain.copy()

        # get data shape from first image
        ny, nx = self.dataout.image.shape

        # block size parameter
        nw = self.getarg('block_size')
        if nw <= 1:
            log.info('No binning performed.')
            return
        elif (nx % nw != 0) or (ny % nw != 0):
            msg = f'Block size {nw} does not divide data shape ' \
                  f'{ny},{nx} evenly.'
            log.error(msg)
            raise ValueError(msg)
        else:
            log.info(f'Binning in {nw}x{nw} blocks.')

        # copy datain to dataout
        self.dataout = self.datain.copy()

        # new array shape, to use for block summing
        new_shape = (ny // nw, nw, nx // nw, nw)

        # slice the header WCS to get new CDELT and CRPIX
        header = self.dataout.header
        hwcs = WCS(header)
        bin_wcs = hwcs[::nw, ::nw]
        updates = {'CRPIX1': bin_wcs.wcs.crpix[0],
                   'CRPIX2': bin_wcs.wcs.crpix[1],
                   'CDELT1': bin_wcs.wcs.cdelt[0],
                   'CDELT2': bin_wcs.wcs.cdelt[1],
                   'PIXSCAL': header['PIXSCAL'] * nw}
        self.dataout.header.update(updates)

        # also add the pixel binning to the header
        self.dataout.setheadval('PIXELBIN', nw, 'Pixel binning block size')

        # correct each flux extension
        for stokes in ['I', 'Q', 'U']:
            fname = f'STOKES {stokes}'
            ename = f'ERROR {stokes}'

            # handle non-polarimetry case
            if fname not in self.dataout.imgnames:
                continue

            flux = self.dataout.imageget(fname)
            var = self.dataout.imageget(ename) ** 2

            # shape flux into blocks and count valid pixels in each
            block_flux = flux.reshape(new_shape)
            pix_count = np.sum(~np.isnan(block_flux), axis=(1, 3))

            # sum over blocks
            bin_flux = np.nansum(block_flux, axis=(1, 3))

            # all-NaN blocks should be NaN
            bin_flux[pix_count == 0] = np.nan

            # scale the rest for missing pixels
            bin_flux[pix_count != 0] *= nw**2 / pix_count[pix_count != 0]

            # assume var NaNs match flux NaNs
            bin_var = np.nansum(var.reshape(new_shape), axis=(1, 3))
            bin_var[pix_count == 0] = np.nan
            bin_var[pix_count != 0] *= (nw**2 / pix_count[pix_count != 0])**2

            self.dataout.imageset(bin_flux, fname)
            self.dataout.imageset(np.sqrt(bin_var), ename)

        # do the same for the covariance extensions
        for covar in ['Q I', 'U I', 'Q U']:
            cname = f'COVAR {covar}'
            if cname not in self.dataout.imgnames:
                continue
            cvar = self.dataout.imageget(cname)

            block_cvar = cvar.reshape(new_shape)
            pix_count = np.sum(~np.isnan(block_cvar), axis=(1, 3))
            bin_cvar = np.nansum(block_cvar, axis=(1, 3))
            bin_cvar[pix_count == 0] = np.nan
            bin_cvar[pix_count != 0] *= (nw**2 / pix_count[pix_count != 0])**2
            self.dataout.imageset(bin_cvar, cname)

        # for the bad pixel mask, just average and floor the
        # input values
        bname = 'BAD PIXEL MASK'
        mask = self.dataout.imageget(bname)
        bin_mask = np.sum(mask.reshape(new_shape), axis=(1, 3))
        self.dataout.imageset(bin_mask // nw**2, bname)
