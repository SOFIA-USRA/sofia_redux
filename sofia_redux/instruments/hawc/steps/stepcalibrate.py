# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Flux calibration pipeline step."""

from astropy import log

from sofia_redux.calibration.pipecal_util \
    import apply_fluxcal, get_fluxcal_factor
from sofia_redux.calibration.pipecal_config import pipecal_config
from sofia_redux.instruments.hawc.stepparent import StepParent

__all__ = ['StepCalibrate']


class StepCalibrate(StepParent):
    """
    Flux calibrate Stokes images.

    This step multiplies the flux in each image by a calibration
    factor to convert to Jy/pixel units.

    It is assumed that the input data have been opacity-corrected
    to a reference altitude and zenith angle, and that the factors were
    derived from flux standards that were similarly corrected. This
    step should be run after the
    `sofia_redux.instruments.hawc.steps.StepOpacity` pipeline step.

    Calibration factors are tracked and applied by configuration files and
    algorithms in the `sofia_redux.calibration` package:

      - `sofia_redux.calibration.pipecal_config`: `pipecal_config`
      - `sofia_redux.calibration.pipecal_util`:
        `get_fluxcal_factor`, `apply_fluxcal`
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'calibrate', and are named with
        the step abbreviation 'CAL'.

        No parameters are currently defined for this step.
        """
        # Name of the pipeline reduction step
        self.name = 'calibrate'
        self.description = 'Calibrate Flux'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'cal'

        # Clear Parameter list
        self.paramlist = []

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object. The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Retrieve a calibration factor from configuration files
           stored in the `sofia_redux.calibration` package.
        2. Multiply flux and error images by the calibration factor.
           Multiply covariance images by the calibration factor squared.
        3. Set the BUNIT keyword in each extension accordingly (to
           'Jy/pixel' or 'Jy2/pixel2' for covariances).
        """

        # copy input to output
        self.dataout = self.datain.copy()
        header = self.dataout.header

        # check obstype -- if standard, just continue
        try:
            obstype = header['OBSTYPE']
        except KeyError:
            obstype = 'UNKNOWN'
        if str(obstype).strip().upper() == 'STANDARD_FLUX':
            log.info('Flux standard; not applying calibration.')
            return

        # Get pipecal config
        cal_conf = pipecal_config(self.dataout.header)

        # Assemble extensions to correct
        imgnames = self.dataout.imgnames
        if 'PRIMARY IMAGE' in imgnames and 'NOISE' in imgnames:
            # for scan products
            extnames = ['PRIMARY IMAGE', 'NOISE']
        else:
            nhwp = self.dataout.getheadval('nhwp')
            if nhwp == 1:
                stokes = ['I']
            else:
                stokes = ['I', 'Q', 'U']

            # flux, error, pixel covariances
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
        corrfac = None
        for i, extname in enumerate(extnames):
            data = self.dataout.imageget(extname)
            hdr = self.dataout.getheader(extname)

            if i == 0:
                # correct data and write values to primary header
                corrdata = apply_fluxcal(data, header, cal_conf)
                corrfac, _ = get_fluxcal_factor(header, cal_conf)
                if corrfac is None:
                    log.warning('No calibration factor found; '
                                'not calibrating data.')
                else:
                    log.info('Flux calibration factor: '
                             '{:.4f}'.format(corrfac))
            else:
                if corrfac is None:
                    break
                if 'VAR' in extname:
                    corrdata = data / corrfac ** 2
                    hdr['BUNIT'] = ('Jy2/pixel2', 'Data units')
                else:
                    corrdata = data / corrfac
                    hdr['BUNIT'] = ('Jy/pixel', 'Data units')

            self.dataout.imageset(corrdata, extname)
