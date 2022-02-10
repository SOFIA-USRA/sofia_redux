# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Stokes Q and U rotation pipeline step."""

from astropy import log
import numpy as np

from sofia_redux.instruments.hawc.stepparent import StepParent

__all__ = ['StepRotate']


class StepRotate(StepParent):
    r"""
    Rotate Stokes Q and U from detector reference frame to sky.

    The rotation angle is derived from telescope data and the
    half-wave plate (HWP) position for the observation.
    The telescope vertical position angle (VPA) is read from the FITS
    header keyword VPOS_ANG.  HWP zero angle and actual initial angle are
    read from the keywords HWPSTART and HWPINIT, respectively.

    Input for this step is a DataFits containing STOKES and ERROR frames
    for I, Q and U each, as well as COVAR Q I, COVAR U I, and COVAR Q U
    extensions. This step is typically run after the
    `sofia_redux.instruments.hawc.steps.StepIP` pipeline step.  The
    output image contains the same image frames as the input image.
    STOKES and ERROR frames for Q and U have been rotated;
    the covariance frames have been propagated.

    Notes
    -----
    The rotation angle is defined per the HAWC+ Geometric
    Reference (Kovács memo, Appendix H), and is applied to the Q and U
    images with a standard rotation matrix as follows:

       .. math:: Q' = cos(\alpha) Q + sin(\alpha) U

       .. math:: U' = sin(\alpha) Q - cos(\alpha) U.

       .. math:: \sigma_Q' = \sqrt{(cos(\alpha)\sigma_Q)^2
                             + (sin(\alpha) \sigma_U)^2
                             +  2 cos(\alpha) sin(\alpha) \sigma_{QU}}

       .. math:: \sigma_U' = \sqrt{(sin(\alpha)\sigma_Q)^2
                             + (cos(\alpha) \sigma_U)^2
                             -  2 cos(\alpha) sin(\alpha) \sigma_{QU}}

       .. math:: \sigma_{Q'I} = cos(\alpha) \sigma_{QI}
                                + sin(\alpha) \sigma_{UI}

       .. math:: \sigma_{U'I} = sin(\alpha) \sigma_{QI}
                                - cos(\alpha) \sigma_{UI}

       .. math:: \sigma_{Q'U'} = cos(\alpha)sin(\alpha)(\sigma_Q^2
                                                        - \sigma_U^2)
                                 + (sin^2(\alpha) - cos^2(\alpha)) \sigma_{QU}.

    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'rotate', and are named with
        the step abbreviation 'ROT'.

        Parameters defined for this step are:

        gridangle : list of float
            Angle of the grid in degrees (one value for
            each waveband).
        hwpzero_tol : float
            Tolerance for the difference between commanded and
            actual initial HWP angles.
        hwpzero_option : str
            If set to 'commanded', then the HWPSTART keyword will
            be used if the difference between the initial HWP angles
            is > hwpzero_tol.  If set to 'actual', the HWPINIT keyword
            will be used.  Otherwise, an error will be raised.
        """
        # Name of the pipeline reduction step
        self.name = 'rotate'
        self.description = 'Rotate Stokes'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'rot'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['gridangle', [0.0, 0.0, 0.0, 0.0, 0.0],
                               'Angle of the grid in degrees '
                               '(for each waveband)'])
        self.paramlist.append(['hwpzero_tol', 3.0,
                               'Tolerance in the difference between '
                               'commanded and actual initial HWP angles'])
        self.paramlist.append(['hwpzero_option', 'commanded',
                               'Option to use between "commanded" or '
                               '"actual" in case the difference between '
                               'the initial HWP angles is > hwpzero_tol'])

    def read_angle(self):
        """
        Read a grid angle value from the parameters.

        The parameters are expected to be defined as a list, with
        one entry for each HAWC band.  The correct value for the
        input data is selected from the list.

        Returns
        -------
        float
            The grid angle.
        """
        gridang = self.getarg('gridangle')

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
            gridang = gridang[idx]
        except IndexError:
            msg = 'Need grid angle values for all wavebands'
            log.error(msg)
            raise IndexError(msg)

        return gridang

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object.  The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Compute the rotation angle from VPA, HWP zero, and
           the grid angle.
        2. Rotate Stokes Q and U arrays.
        3. Propagate errors and covariances.
        """
        self.dataout = self.datain.copy()
        nhwp = self.dataout.getheadval('nhwp')

        if nhwp == 1:
            log.info('Only 1 HWP, so skipping step %s' % self.name)
        else:
            vpa = self.dataout.getheadval('VPOS_ANG')
            gridang = self.read_angle()
            hwpzero_tol = self.getarg('hwpzero_tol')
            hwpzero_option = self.getarg('hwpzero_option')

            # Test if commanded and actual initial
            # HWP angles are consistent (within tolerance)
            hwpzero = self.dataout.getheadval('hwpstart')
            hwpinit = self.dataout.getheadval('hwpinit')
            if hwpzero > 180.:
                hwpzero = hwpzero - 360.
            if hwpinit > 180.:
                hwpinit = hwpinit - 360.
            diffhwp = abs(hwpinit - hwpzero)
            if diffhwp > hwpzero_tol:
                if hwpzero_option == 'commanded':
                    log.error('Initial HWP angle difference is above '
                              'the tolerance. Will use the %s '
                              'value (%s)' % (hwpzero_option, hwpzero))
                elif hwpzero_option == 'actual':
                    log.error('Initial HWP angle difference is above '
                              'the tolerance. Will use the %s '
                              'value (%s)' % (hwpzero_option, hwpinit))
                    hwpzero = hwpinit
                else:
                    msg = 'Initial HWP angle difference is above the ' \
                          'tolerance. hwpzero_option parameter value ' \
                          'must be commanded or actual'
                    log.error(msg)
                    raise ValueError(msg)

            # The equations below are described at Appendix H of
            # Attila's memo on 'HAWC+ Geometric Reference'

            # propagation equations:
            # I' = I
            # Q' = c Q + s U
            # U' = s Q - c U
            # VI' = VI
            # VQ' = c^2 VQ + s^2 VU + 2 c s cov(Q,U)
            # VU' = s^2 VQ + c^2 VU - 2 c s cov(Q,U)
            # cov(Q',I') = c cov(Q,I) + s cov(U,I)
            # cov(U',I') = s cov(Q,I) - c cov(U,I)
            # cov(Q',U') = c s VQ - c s VU + (s^2 - c^2) cov(Q,U)

            theta = (vpa + 2. * (hwpzero - 5.) + gridang) * np.pi / 180.
            cos = np.cos(2. * theta)
            sin = np.sin(2. * theta)
            oldq = self.dataout.imageget('STOKES Q')
            oldu = self.dataout.imageget('STOKES U')
            q = cos * oldq + sin * oldu
            self.dataout.imageset(q, 'STOKES Q')
            u = sin * oldq - cos * oldu
            self.dataout.imageset(u, 'STOKES U')

            varq = (self.dataout.imageget('ERROR Q'))**2
            varu = (self.dataout.imageget('ERROR U'))**2
            covqi = self.dataout.imageget('COVAR Q I')
            covui = self.dataout.imageget('COVAR U I')
            covqu = self.dataout.imageget('COVAR Q U')

            varqprime = cos**2 * varq + \
                sin**2 * varu + \
                2 * cos * sin * covqu
            varuprime = sin**2 * varq + \
                cos**2 * varu - \
                2 * cos * sin * covqu

            covqiprime = cos * covqi + sin * covui
            covuiprime = sin * covqi - cos * covui
            covquprime = cos * sin * varq - \
                cos * sin * varu + \
                (sin**2 - cos**2) * covqu

            self.dataout.imageset(np.sqrt(varqprime), 'ERROR Q')
            self.dataout.imageset(np.sqrt(varuprime), 'ERROR U')
            self.dataout.imageset(covqiprime, 'COVAR Q I')
            self.dataout.imageset(covuiprime, 'COVAR U I')
            self.dataout.imageset(covquprime, 'COVAR Q U')
