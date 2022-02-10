# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Polarization vector pipeline step."""

from astropy import log
from astropy import wcs as astwcs
from astropy.io import fits
import numpy as np

from sofia_redux.instruments.hawc.stepparent import StepParent

__all__ = ['StepPolVec']


class StepPolVec(StepParent):
    r"""
    Calculate polarization vectors.

    This step uses the Stokes parameter images and associated
    errors and covariances to compute the percent polarization
    and angle for each pixel. It also computes the debiased
    polarization and the rotated angle (for B-field directions).
    These values are all stored in a FITS table extension named
    'Pol Data'.

    The input for this step is a DataFits object with images:
    STOKES I, ERROR I, STOKES Q, STOKES U, COVAR Q I, COVAR U I,
    COVAR Q U. This step is typically run after
    `sofia_redux.instruments.hawc.steps.StepMerge`.

    Output for this step contains the same image frames as the input
    files, plus a table with the polarization vectors for each
    pixel (POL DATA). Images of these data are also added as separate
    extensions. The added extensions are named: PERCENT POL,
    DEBIASED PERCENT POL, ERROR PERCENT POL, POL ANGLE, ROTATED POL
    ANGLE, ERROR POL ANGLE, POL FLUX, ERROR POL FLUX, DEBIASED POL FLUX.
    Finally, the PROCSTAT header keyword will be updated to LEVEL\_4
    after this step.

    Notes
    -----
    The standard equations are used to convert from Stokes
    parameters to polarization percentage and angle:

       .. math:: \theta = \frac{90}{\pi} arctan\Big(\frac{U}{Q}\Big)

       .. math:: \sigma_\theta = \frac{90}{\pi (Q^2 + U^2)}
                                 \sqrt{(U\sigma_Q)^2 +
                                       (Q\sigma_U)^2 - 2 Q U \sigma_{QU}}.

    The percent polarization (:math:`p`) and its error are calculated as

       .. math:: p = 100 \sqrt{\Big(\frac{Q}{I}\Big)^2
                               + \Big(\frac{U}{I}\Big)^2}

       .. math:: \sigma_p = \frac{100}{I} \sqrt{\frac{1}{(Q^2 + U^2)}
                            \Big[(Q \sigma_Q)^2
                                 + (U \sigma_U)^2 + 2 Q U \sigma_{QU}\Big]
                            + \Big[\Big(\frac{Q}{I}\Big)^2
                                   + \Big(\frac{U}{I}\Big)^2\Big]
                            \sigma_I^2 - 2 \frac{Q}{I}\sigma_{QI}
                            - 2 \frac{U}{I} \sigma_{UI}}.

    The debiased polarization percentage (:math:`p'`)is also calculated, as:

       .. math:: p' = \sqrt{p^2 - \sigma_p^2}.

    The polarization efficiency provided in the 'eff' parameter is
    applied to the Q and U values (and their associated errors and
    covariances) after calculating :math:`\theta`, but before calculating
    percent polarization.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'polvec', and are named with
        the step abbreviation 'VEC'.

        Parameters defined for this step are:

        eff : list of float
            Telescope + instrument polarization efficiency. One
            value per HAWC waveband.
        """
        # Name of the pipeline reduction step
        self.name = 'polvec'
        self.description = 'Compute Vectors'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'vec'

        # Clear Parameter list
        self.paramlist = []
        # Append parameters
        self.paramlist.append(['eff', [1.0, 1.0, 1.0, 1.0, 1.0],
                               'telescope + instrument polarization '
                               'efficiency for each waveband'])

    def read_eff(self):
        """
        Read an efficiency value from the parameters.

        The parameters are expected to be defined as a list, with
        one entry for each HAWC band.  The correct value for the
        input data is selected from the list.

        Returns
        -------
        float
            The polarization efficiency.
        """
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
            eff = self.getarg('eff')[idx]
        except IndexError:
            msg = 'Need efficiency values for all wavebands'
            log.error(msg)
            raise IndexError(msg)
        return eff

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object.  The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Compute polarization vector magnitude and direction.
           Propagate errors accordingly.
        2. Store the data in tables and images.
        """
        self.dataout = self.datain.copy()

        nhwp = self.dataout.getheadval('nhwp')

        # suppress warnings when I do stuff with NaN's or divide by zero
        np.seterr(invalid='ignore', divide='ignore')

        if nhwp == 1:
            log.info('Only 1 HWP, so skipping step %s' % self.name)
        else:

            si = self.dataout.imageget("STOKES I")
            sq = self.dataout.imageget("STOKES Q")
            su = self.dataout.imageget("STOKES U")
            di = self.dataout.imageget("ERROR I")
            dq = self.dataout.imageget("ERROR Q")
            du = self.dataout.imageget("ERROR U")
            covqi = self.dataout.imageget("COVAR Q I")
            covui = self.dataout.imageget("COVAR U I")
            covqu = self.dataout.imageget("COVAR Q U")

            # Compute parameters

            # theta = 90/pi * atan(U/Q)
            # Vtheta = (90/pi)^2 *
            #          (Q^2 + U^2)^-2 (U^2 VQ + Q^2VU - 2 Q U cov(Q,U))
            # (arctan2 puts angle in right quadrant)
            theta = (90. / np.pi) * np.arctan2(su, sq)
            dtheta = (90. / (np.pi * (sq**2 + su**2))) * \
                np.sqrt((su * dq)**2 + (sq * du)**2 - 2 * sq * su * covqu)
            rot_theta = theta + 90
            rot_theta[rot_theta > 90] -= 180

            # apply efficiency correction to Q and U
            # (this gets propagated to p)
            eff = self.read_eff()
            sq /= eff
            su /= eff
            dq /= eff
            du /= eff
            covqi /= eff
            covui /= eff
            covqu /= eff**2

            # polarized flux
            # f = (Q^2 + U^2) ^ (1/2)
            # Vf = (1/f^2)(Q^2 VQ + U^2 Vu + 2 Q U cov(Q,U))
            pflux = np.sqrt(sq**2 + su**2)
            dpflux = np.sqrt((sq * dq)**2
                             + (su * du)**2
                             + 2 * sq * su * covqu) / pflux

            debias_pflux = pflux**2 - dpflux**2  # ricean debiasing
            mask = np.where(debias_pflux < 0)  # catch negative square roots
            debias_pflux[mask] = 0
            debias_pflux = np.sqrt(debias_pflux)

            # polarization fraction
            # p = (1/I) (Q^2 + U^2)^(1/2)
            # Vp = (1/(I^4 p^2)) (Q^2 VQ + U^2 VU + 2 Q U cov(Q,U))
            #      + 1/I^4 ((Q^2 + U^2) VI - 2 Q I cov(Q,I) - 2 U I cov(U,i))
            p = np.sqrt((1 / si**2) * (sq**2 + su**2))
            vp = (p**-2 * si**-4) * ((sq * dq)**2
                                     + (su * du)**2
                                     + 2 * sq * su * covqu) \
                + (si**-4) * ((sq**2 + su**2) * di**2
                              - 2 * sq * si * covqi
                              - 2 * su * si * covui)
            dp = np.sqrt(vp)

            # convert p to percentage
            p = 100 * p
            dp = 100 * dp

            # debias data
            debias_p = p**2 - dp**2  # ricean debiasing
            mask = np.where(debias_p < 0)  # catch negative square roots
            debias_p[mask] = 0
            debias_p = np.sqrt(debias_p)

            # Note that here we set x equal to columns and y equal to rows
            ny, nx = si.shape
            y, x = np.mgrid[0:ny, 0:nx] + 1
            x = x.flatten()
            y = y.flatten()

            # Convert pixels to wcs
            wcs = astwcs.WCS(self.datain.header)
            ra, dec = wcs.wcs_pix2world(x, y, 1)  # zero-based input pixels

            # create table columns
            cols = [fits.Column(name="Pixel X", format='J', array=x),
                    fits.Column(name="Pixel Y", format='J', array=y),
                    fits.Column(name="Right Ascension", format='D',
                                array=ra, unit='deg'),
                    fits.Column(name="Declination", format='D',
                                array=dec, unit='deg'),
                    fits.Column(name="Percent Pol", format='D',
                                array=p.flatten()),
                    fits.Column(name="Debiased Percent Pol", format='D',
                                array=debias_p.flatten()),
                    fits.Column(name="Err. Percent Pol", format='D',
                                array=dp.flatten()),
                    fits.Column(name="Theta", format='D', unit='deg',
                                array=theta.flatten()),
                    fits.Column(name="Rotated Theta", format='D',
                                unit='deg', array=rot_theta.flatten()),
                    fits.Column(name="Err. Theta", format='D', unit='deg',
                                array=dtheta.flatten())]
            c = fits.ColDefs(cols)
            tbhdu = fits.BinTableHDU.from_columns(c)
            self.dataout.tableset(tbhdu.data, "POL DATA",
                                  tbhdu.header)

            # creating header with wcs info to use in new HDU's
            wcshead = fits.Header()
            wcshead['CDELT1'] = self.datain.header['CDELT1']
            wcshead['CDELT2'] = self.datain.header['CDELT2']
            wcshead['CRVAL1'] = self.datain.header['CRVAL1']
            wcshead['CRVAL2'] = self.datain.header['CRVAL2']
            wcshead['CRPIX1'] = self.datain.header['CRPIX1']
            wcshead['CRPIX2'] = self.datain.header['CRPIX2']
            wcshead['CTYPE1'] = self.datain.header['CTYPE1']
            wcshead['CTYPE2'] = self.datain.header['CTYPE2']
            wcshead['EQUINOX'] = self.datain.header['EQUINOX']
            wcshead['NHWP'] = self.datain.header['NHWP']

            # now write out images of the relevant data so they can easily
            # be viewed in something like ds9
            # Q/U and its errors will overwrite the previous HDUs
            # They now have pol. efficiency factor applied
            # Also replace dI to make sure WCS is set
            self.dataout.imageset(di, "Error I", wcshead)
            self.dataout.imageset(sq, "Stokes Q", wcshead)
            self.dataout.imageset(dq, "Error Q", wcshead)
            self.dataout.imageset(su, "Stokes U", wcshead)
            self.dataout.imageset(du, "Error U", wcshead)
            self.dataout.imageset(p, "Percent Pol", wcshead)
            self.dataout.imageset(debias_p, "Debiased Percent Pol", wcshead)
            self.dataout.imageset(dp, "Error Percent Pol", wcshead)
            self.dataout.imageset(theta, "Pol Angle", wcshead)
            self.dataout.imageset(rot_theta, "Rotated Pol Angle", wcshead)
            self.dataout.imageset(dtheta, "Error Pol Angle", wcshead)
            self.dataout.imageset(pflux, "Pol Flux", wcshead)
            self.dataout.imageset(dpflux, "Error Pol Flux", wcshead)
            self.dataout.imageset(debias_pflux, "Debiased Pol Flux", wcshead)

            # delete the covariance images -- they're no longer needed
            self.dataout.imagedel("COVAR Q I")
            self.dataout.imagedel("COVAR U I")
            self.dataout.imagedel("COVAR Q U")

            # set BUNIT appropriately for all extensions
            for name in ['Stokes I', 'Error I',
                         'Stokes Q', 'Error Q',
                         'Stokes U', 'Error U',
                         'Pol Flux', 'Error Pol Flux',
                         'Debiased Pol Flux']:
                self.dataout.setheadval('BUNIT', 'Jy/pixel',
                                        comment='Data units',
                                        dataname=name)
            for name in ["Percent Pol", "Debiased Percent Pol",
                         "Error Percent Pol"]:
                self.dataout.setheadval('BUNIT', 'percent',
                                        comment='Data units',
                                        dataname=name)
            for name in ["Pol Angle", "Rotated Pol Angle",
                         "Error Pol Angle"]:
                self.dataout.setheadval('BUNIT', 'deg',
                                        comment='Data units',
                                        dataname=name)

        self.dataout.setheadval('PROCSTAT', 'LEVEL_4')
