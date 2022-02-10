# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Instrumental polarization correction pipeline step."""

import os.path

from astropy import log
import numpy as np

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.stepparent import StepParent

__all__ = ['StepIP']


class StepIP(StepParent):
    r"""
    Remove instrumental polarization from Stokes images.

    This step subtracts instrumental polarization in the
    detector frame, either with a different value for each detector
    pixel, or with a uniform value assigned to all pixels.

    Input for this step is DataFits containing Stokes and Error
    images for each of I, Q, and U, as well as COVAR Q I and
    COVAR U I images, a bad pixel image, and a table of data.

    This step is typically run after
    `sofia_redux.instruments.hawc.steps.StepWcs` and before
    `sofia_redux.instruments.hawc.steps.StepRotate`.  Output from this
    step contains the same extensions as the input, with values modified
    for the Stokes Q and U images, along with their associated error
    and covariance images.

    Notes
    -----
    The correction is applied as

       .. math:: Q' = Q - q' I

       .. math:: U' = U - u' I

    and propagated to the associated error and covariance images as

       .. math:: \sigma_Q' = \sqrt{\sigma_Q^2
                 + (q' \sigma_I)^2 +  2q'\sigma_{QI}}

       .. math:: \sigma_U' = \sqrt{\sigma_U^2
                 + (u' \sigma_I)^2 +  2u'\sigma_{UI}}

       .. math:: \sigma_{Q'I} = \sigma_{QI} - q' \sigma_I^2

       .. math:: \sigma_{U'I} = \sigma_{UI} - u' \sigma_I^2

       .. math:: \sigma_{Q'U'} = -u' \sigma_{QI} - q' \sigma_{UI}
                 + qu\sigma_I^2.

    Note that the input :math:`\sigma_{QU}` is assumed to be identically
    zero, since this correction is applied before rotation of the Q and U
    parameters.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'ip', and are named with
        the step abbreviation 'IPS'.

        Parameters defined for this step are:

        qinst : list of float
            Fractional instrumental polarization in q (one value for
            each waveband).
        uinst : list of float
            Fractional instrumental polarization in u (one value for
            each waveband).
        fileip : str
            If set to 'uniform', then the qinst and uinst values will
            be applied uniformly to all pixels.  Otherwise, this
            parameter is expected to be a path to a FITS file
            containing IP values for each pixel, at each waveband.
        """
        # Name of the pipeline reduction step
        self.name = 'ip'
        self.description = 'Correct IP'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'ips'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['qinst', [0.0, 0.0, 0.0, 0.0, 0.0],
                               'Fractional instrumental polarization '
                               'in q (each waveband)'])
        self.paramlist.append(['uinst', [0.0, 0.0, 0.0, 0.0, 0.0],
                               'Fractional instrumental polarization '
                               'in u (each waveband)'])
        self.paramlist.append(['fileip', 'uniform',
                               'Fits file with ip array values. If set '
                               'to "uniform", will use qinst and uinst '
                               'to make a uniform correction for all pixels'])

    def read_ip(self):
        """
        Read IP q and u values from the parameters.

        The parameters are expected to be defined as a list, with
        one entry for each HAWC band.  The correct value for the
        input data is selected from the list.

        Returns
        -------
        qi : float
            IP q value for the input data.
        ui : float
            IP u value for the input data.
        """
        # NOTE: These values are reduced Stokes parameters
        qi = self.getarg('qinst')
        ui = self.getarg('uinst')
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
            qi = qi[idx]
            ui = ui[idx]
        except IndexError:
            msg = 'Missing IP values for all wavebands'
            log.error(msg)
            raise IndexError(msg)

        # since these are normalized stokes parameters, ensure none are > 1.
        for stokes in [qi, ui]:
            if abs(float(stokes)) > 1:
                msg = "Absolute value of IP parameters must be <= 1!"
                log.error(msg)
                raise ValueError(msg)

        return qi, ui

    def fill_nan_median(self, array):
        """
        Fill in NaNs in an array with its median value.

        Parameters
        ----------
        array : array-like
            The array to process.

        Returns
        -------
        array-like
            The array with NaN values replaced.
        """
        nans = np.where(array != array)
        median = np.nanmedian(array)
        array[nans] = median

        return array

    def read_file_ip(self, fileip):
        """
        Read IP values from a file.

        Extract correct arrays given the observing wavelength,
        and return them.  For waveband X, the file must contain
        extensions 'IP q band X', 'Error IP q band X',
        'IP u band X', and 'Error IP u band X'.

        Parameters
        ----------
        fileip : str
            Path to an IP FITS file.

        Returns
        -------
        qi : array-like
            IP q value matching the input data.
        ui : array-like
            IP u value matching the input data.
        eqi : array-like
            Error on the IP q value.
        eui : array-like
            Error on the IP u value.
        band : str
            Filter band name.
        """
        ipdata = DataFits(config=self.config)
        ipdata.load(fileip)

        waveband = self.datain.getheadval('spectel1')
        band = waveband[-1]

        try:
            qi = ipdata.imageget('IP q band %s' % band)
            eqi = ipdata.imageget('Error IP q band %s' % band)
            ui = ipdata.imageget('IP u band %s' % band)
            eui = ipdata.imageget('Error IP u band %s' % band)
        except ValueError:
            msg = 'Problem with band %s HDU in fileip. ' \
                  'Unable to perform IP correction' % band
            log.error(msg)
            raise ValueError(msg)

        # NaNs are replaced by the array's median value
        qi = self.fill_nan_median(qi)
        eqi = self.fill_nan_median(eqi)
        ui = self.fill_nan_median(ui)
        eui = self.fill_nan_median(eui)

        return qi, ui, eqi, eui, band

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object.  The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Determine the IP correction from the parameters.
        2. Apply it to Q and U images, and propagate errors
           and covariances.
        """

        self.dataout = self.datain.copy()
        nhwp = self.dataout.getheadval('nhwp')

        if nhwp == 1:
            log.info('Only 1 HWP, so skipping step %s' % self.name)
            return

        fileip = self.getarg('fileip')
        if fileip is not None:
            fileip = os.path.expandvars(fileip)
        if fileip == 'uniform':
            qi, ui = self.read_ip()
            log.debug("Using q, u correction: "
                      "{:.4f}, {:.4f}".format(qi, ui))
        else:
            fileexist = os.path.isfile(fileip)
            if fileexist:
                qi, ui, eqi, eui, band = self.read_file_ip(fileip)
                msg = "Using fileip (%s) for band %s IP correction." % \
                      (fileip, band)
                log.info(msg)
                log.debug("Median q, u correction: "
                          "{:.4f}, {:.4f}".format(np.nanmedian(qi),
                                                  np.nanmedian(ui)))
            else:
                msg = "Fileip (%s) was not found. Set fileip " \
                      "to 'uniform' to use qinst and uinst instead" % \
                      fileip
                log.error(msg)
                raise ValueError(msg)

        qraw = qi
        uraw = ui

        image_i = self.dataout.imageget('STOKES I')
        var_i = (self.dataout.imageget('ERROR I')) ** 2

        # propagation equations:
        # I' = I
        # Q' = Q - qI
        # U' = U - uI
        # VI' = VI
        # VQ' = VQ + q^2 VI - 2q cov(Q,I)
        # VU' = VU + u^2 VI - 2u cov(U,I)
        # cov(Q',I') = cov(Q,I) - q VI
        # cov(U',I') = cov(U,I) - u VI
        # cov(Q',U') = -u cov(Q,I) - q cov(U,I) + q u VI

        # get images
        q = self.dataout.imageget('STOKES Q')
        u = self.dataout.imageget('STOKES U')
        var_q = (self.dataout.imageget('ERROR Q')) ** 2
        var_u = (self.dataout.imageget('ERROR U')) ** 2
        cov_qi = self.dataout.imageget('COVAR Q I')
        cov_ui = self.dataout.imageget('COVAR U I')

        # Q and U
        q -= qraw * image_i
        var_q += qraw**2 * var_i - 2 * qraw * cov_qi
        u -= uraw * image_i
        var_u += uraw**2 * var_i - 2 * uraw * cov_ui

        # covariances
        cov_qu = -uraw * cov_qi - qraw * cov_ui + qraw * uraw * var_i
        cov_qi -= qraw * var_i
        cov_ui -= uraw * var_i

        # set images
        self.dataout.imageset(q, 'STOKES Q')
        self.dataout.imageset(u, 'STOKES U')
        self.dataout.imageset(np.sqrt(var_q), 'ERROR Q')
        self.dataout.imageset(np.sqrt(var_u), 'ERROR U')
        self.dataout.imageset(cov_qi, 'COVAR Q I')
        self.dataout.imageset(cov_ui, 'COVAR U I')
        self.dataout.imageset(cov_qu, 'COVAR Q U')
