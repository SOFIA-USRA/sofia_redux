# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Opacity correction pipeline step."""

from astropy import log

from sofia_redux.calibration.pipecal_util import \
    apply_tellcor, get_tellcor_factor
from sofia_redux.calibration.pipecal_config import pipecal_config
from sofia_redux.instruments.hawc.stepparent import StepParent

__all__ = ['StepOpacity']


class StepOpacity(StepParent):
    """
    Apply an atmospheric opacity correction.

    This step corrects for atmospheric transmission based on a
    standard model of the atmosphere, the altitude/ZA in the
    header, and the instrument response by band. The correction
    factor is multiplied into the flux images, and propagated
    to the variance and covariance images.

    Opacity correction factors are calculated and applied by
    algorithms in the `sofia_redux.calibration` package:

      - `sofia_redux.calibration.pipecal_config`: `pipecal_config`
      - `sofia_redux.calibration.pipecal_util`:
        `get_tellcor_factor`, `apply_tellcor`

    Input for this image must contain either STOKES and ERROR images,
    as produced by the `sofia_redux.instruments.hawc.steps.StepStokes`
    pipeline step, or else PRIMARY IMAGE and NOISE images, as produced by the
    `sofia_redux.instruments.hawc.steps.StepScanMap` pipeline step. For
    polarimetry data, covariance images (COVAR Q I, COVAR U I, and COVAR Q U)
    are also expected. The output DataFits has the same extensions as the
    input.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'calibrate', and are named with
        the step abbreviation 'CAL'.

        No parameters are currently defined for this step.
        """
        # Name of the pipeline reduction step
        self.name = 'opacity'
        self.description = 'Correct Opacity'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'opc'

        # Clear Parameter list
        self.paramlist = []

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object. The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Retrieve an opacity correction factor from the
           `sofia_redux.calibration` package.
        2. Multiply flux and error images by the correction factor.
           Multiply covariance images by the correction factor squared.
        """

        # Copy datain to dataout
        self.dataout = self.datain.copy()

        # Get pipecal config
        cal_conf = pipecal_config(self.dataout.header)

        # Assemble extensions to correct
        imgnames = self.dataout.imgnames
        if 'PRIMARY IMAGE' in imgnames and 'NOISE' in imgnames:
            # for scan products
            extnames = ['PRIMARY IMAGE', 'NOISE']
        else:
            try:
                nhwp = self.dataout.getheadval('nhwp')
            except KeyError:
                nhwp = 1
            if nhwp == 1:
                stokes = ['I']
            else:
                stokes = ['I', 'Q', 'U']

            # flux, error
            extnames = []
            for var in stokes:
                extnames.append('STOKES %s' % var)
                extnames.append('ERROR %s' % var)

            # stokes covariances
            if nhwp > 1:
                stokes = ['Q I', 'U I', 'Q U']
                for var in stokes:
                    extnames.append('COVAR %s' % var)

        # Correct each extension
        header = self.dataout.header
        corrfac = 1.0
        for i, extname in enumerate(extnames):
            log.debug('Correcting extension: {}'.format(extname))
            data = self.dataout.imageget(extname)

            if i == 0:
                # correct data and write values to primary header
                corrdata = apply_tellcor(data, header, cal_conf)
                corrfac = get_tellcor_factor(header, cal_conf)
                log.info('Opacity correction factor: '
                         '{:.4f}'.format(corrfac))
            else:
                if 'VAR' in extname:
                    corrdata = corrfac ** 2 * data
                else:
                    corrdata = corrfac * data

            self.dataout.imageset(corrdata, extname)
