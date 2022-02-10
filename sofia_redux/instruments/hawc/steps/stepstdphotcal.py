# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Flux standard photometry pipeline step."""

from astropy import log

from sofia_redux.calibration.pipecal_util import run_photometry, \
    apply_fluxcal, get_fluxcal_factor
from sofia_redux.calibration.pipecal_config import pipecal_config

from sofia_redux.instruments.hawc.stepparent import StepParent

__all__ = ['StepStdPhotCal']


class StepStdPhotCal(StepParent):
    """
    Measure photometry and calibrate flux standard observations.

    This pipeline step runs aperture photometry on flux standards
    in raw units, then applies a standard flux calibration factor to
    calibrate the flux to physical units (Jy/pixel).

    It is assumed that the input data have been opacity-corrected
    to a reference altitude and zenith angle, and that the calibration
    factors were derived from flux standards that were similarly
    corrected. This step should be run after the
    `sofia_redux.instruments.hawc.steps.StepOpacity` pipeline step.

    Calibration factors are tracked and applied by configuration files and
    algorithms in the `sofia_redux.calibration` package:

      - `sofia_redux.calibration.pipecal_config`: `pipecal_config`
      - `sofia_redux.calibration.pipecal_util`:
        `get_fluxcal_factor`, `apply_fluxcal`

    Photometry routines are also provided by the
    `sofia_redux.calibration` package, via:

      - `sofia_redux.calibration.pipecal_util`: `run_photometry`

    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'stdphotcal', and are named with
        the step abbreviation 'STD'.

        Parameters defined for this step are:

        srcpos : str
            Initial guess position for photometry, given as "x,y".
            If a blank string is provided, the brightest peak in the
            image will be used as the source position.
        fitsize : int
            Sub-image size to use for profile fit, in pixels.
        fwhm : float
            Initial FWHM for profile fit, in pixels.
        profile : str
            Profile type for source fit (moffat, gaussian).
        aprad : float
            Aperture radius for photometry, in pixels.
        skyrad : list of float
            Background annulus radii, in pixels, given as [inner, outer].
        """
        # Name of the pipeline reduction step
        self.name = 'stdphotcal'
        self.description = 'Compute Photometry'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'std'

        # Clear Parameter list
        self.paramlist = []
        self.paramlist.append(['srcpos', '',
                               'Initial guess position for photometry'])
        self.paramlist.append(['fitsize', 100,
                               'Photometry fit size (pix)'])
        self.paramlist.append(['fwhm', 5.0,
                               'Initial FWHM for fits (pix)'])
        self.paramlist.append(['profile', 'moffat',
                               'Profile type for fits (moffat, gaussian)'])
        self.paramlist.append(['aprad', 20.0,
                               'Aperture radius (pixels)'])
        self.paramlist.append(['skyrad', [25.0, 35.0],
                               'Background annulus radii '
                               '(inner,outer in pixels)'])

    def run_phot(self, cal_conf, kwargs, write=False):
        """
        Run aperture photometry measurement.

        Data in self.dataout are used as input.  If a 'PRIMARY IMAGE'
        extension is present (as from ScanMap), it is used.  Otherwise,
        a 'STOKES I' image is used.  Associated error planes are also
        passed to the photometry algorithm.

        Parameters
        ----------
        cal_conf : dict
            Pipecal configuration information.
        kwargs : dict
            Arguments to pass to the run_photometry function.
        write : bool, optional
            If set, photometry keywords are written to the primary
            header for the self.dataout DataFits.

        Returns
        -------
        list of str
            Extension names used for photometry.
        """
        header = self.dataout.header.copy()
        imgnames = self.dataout.imgnames
        if 'PRIMARY IMAGE' in imgnames and 'NOISE' in imgnames:
            # for scan products
            flux = self.dataout.imageget('PRIMARY IMAGE')
            variance = self.dataout.imageget('NOISE')
            extnames = ['PRIMARY IMAGE', 'NOISE']
        else:
            flux = self.dataout.imageget('STOKES I')
            variance = self.dataout.imageget('ERROR I') ** 2
            extnames = None
        try:
            run_photometry(flux, header, variance, cal_conf, **kwargs)
            if write:
                self.dataout.header = header
        except ValueError:
            log.warning('Unable to run photometry on flux standard.')
        return extnames

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object. The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Retrieve calibration configuration from the
           `sofia_redux.calibration` package.
        2. Run photometry on the Stokes I image.
        3. Multiply flux and error images by the calibration factor.
           Multiply covariance images by the calibration factor squared.
        4. Set the BUNIT keyword in each extension accordingly (to
           'Jy/pixel' or 'Jy2/pixel2' for covariances).
        """

        # get parameters
        kwargs = {}
        srcpos = self.getarg('srcpos')
        if str(srcpos).strip().lower() not in ['none', '']:
            msg = 'Invalid source position.'
            try:
                srcpos = [float(x) for x in str(srcpos).split(',')]
            except ValueError:
                log.error(msg)
                raise ValueError(msg)
            if len(srcpos) != 2:
                log.error(msg)
                raise ValueError(msg)
            kwargs['srcpos'] = srcpos
        kwargs['fitsize'] = self.getarg('fitsize')
        kwargs['fwhm'] = self.getarg('fwhm')
        kwargs['profile'] = self.getarg('profile')
        kwargs['aprad'] = self.getarg('aprad')
        kwargs['skyrad'] = self.getarg('skyrad')

        # Copy datain to dataout
        self.dataout = self.datain.copy()
        header = self.dataout.header

        # Test for previous calibration
        try:
            bunit = header['BUNIT']
        except KeyError:
            bunit = 'UNKNOWN'
        if 'JY' not in str(bunit).strip().upper():
            calibrated = False
        else:
            calibrated = True

        # Get pipecal config
        cal_conf = pipecal_config(self.dataout.header)
        log.debug('Full calibration config:')
        for key, val in cal_conf.items():
            log.debug('  {}: {}'.format(key, val))

        # first run photometry on the Stokes I image
        log.info('')
        if not calibrated:
            log.info('Before calibration:')

        # run the photometry
        extnames = self.run_phot(cal_conf, kwargs, write=True)

        if calibrated:
            # skip calibration if already done
            log.info('')
            return

        # Then calibrate to Jy

        # Assemble extensions to correct, if not already retrieved
        if extnames is None:
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
                    log.info('')
                    log.info('Applying flux calibration factor: '
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

        # Print source flux after calibration
        try:
            flux = header['STAPFLX'] / corrfac
            flux_err = header['STAPFLXE'] / corrfac
            log.info('')
            log.info('After calibration:')
            log.info('Source Flux: '
                     '{:.2f} +/- {:.2f} Jy'.format(flux, flux_err))
        except (KeyError, ValueError, TypeError):
            pass
        else:
            try:
                modlflx = header['MODLFLX']
                modlflxe = header['MODLFLXE']
                log.info('Model Flux: '
                         '{:.3f} +/- {:.3f} Jy'.format(modlflx, modlflxe))
                log.info('Percent difference from model: '
                         '{:.1f}%'.format(100 * (flux - modlflx) / modlflx))
            except KeyError:
                pass
            log.info('')
