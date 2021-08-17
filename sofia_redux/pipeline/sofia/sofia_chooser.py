# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Choose SOFIA reduction objects based on input data."""

import os

from astropy import log
from astropy.io import fits
import configobj

from sofia_redux.pipeline.reduction import Reduction
from sofia_redux.pipeline.chooser import Chooser
from sofia_redux.pipeline.sofia.sofia_exception import SOFIAImportError

# attempt to import all available instruments
try:
    from sofia_redux.pipeline.sofia.forcast_imaging_reduction import \
        FORCASTImagingReduction
    from sofia_redux.pipeline.sofia.forcast_spectroscopy_reduction import \
        FORCASTSpectroscopyReduction
    from sofia_redux.pipeline.sofia.forcast_wavecal_reduction import \
        FORCASTWavecalReduction
    from sofia_redux.pipeline.sofia.forcast_spatcal_reduction import \
        FORCASTSpatcalReduction
    from sofia_redux.pipeline.sofia.forcast_slitcorr_reduction import \
        FORCASTSlitcorrReduction
    from sofia_redux.pipeline.sofia.exes_quicklook_reduction import \
        EXESQuicklookReduction
except SOFIAImportError:  # pragma: no cover
    FORCASTImagingReduction = Reduction
    FORCASTSpectroscopyReduction = Reduction
    FORCASTWavecalReduction = Reduction
    FORCASTSpatcalReduction = Reduction
    FORCASTSlitcorrReduction = Reduction
    EXESQuicklookReduction = Reduction
    FORCAST_ERROR = None
except ImportError as err:  # pragma: no cover
    FORCASTImagingReduction = None
    FORCASTSpectroscopyReduction = None
    FORCASTWavecalReduction = None
    FORCASTSpatcalReduction = None
    FORCASTSlitcorrReduction = None
    EXESQuicklookReduction = None
    FORCAST_ERROR = err
else:
    FORCAST_ERROR = None

try:
    from sofia_redux.pipeline.sofia.hawc_reduction import HAWCReduction
except SOFIAImportError:  # pragma: no cover
    HAWCReduction = Reduction
    HAWC_ERROR = None
except ImportError as err:  # pragma: no cover
    HAWCReduction = None
    HAWC_ERROR = err
else:
    HAWC_ERROR = None

try:
    from sofia_redux.pipeline.sofia.fifils_reduction import FIFILSReduction
except SOFIAImportError:  # pragma: no cover
    FIFILSReduction = Reduction
    FIFILS_ERROR = None
except ImportError as err:  # pragma: no cover
    FIFILSReduction = None
    FIFILS_ERROR = err
else:
    FIFILS_ERROR = None

try:
    from sofia_redux.pipeline.sofia.flitecam_imaging_reduction import \
        FLITECAMImagingReduction
    from sofia_redux.pipeline.sofia.flitecam_spectroscopy_reduction import \
        FLITECAMSpectroscopyReduction
    from sofia_redux.pipeline.sofia.flitecam_wavecal_reduction import \
        FLITECAMWavecalReduction
    from sofia_redux.pipeline.sofia.flitecam_spatcal_reduction import \
        FLITECAMSpatcalReduction
    from sofia_redux.pipeline.sofia.flitecam_slitcorr_reduction import \
        FLITECAMSlitcorrReduction
except SOFIAImportError:  # pragma: no cover
    FLITECAMImagingReduction = Reduction
    FLITECAMSpectroscopyReduction = Reduction
    FLITECAMWavecalReduction = Reduction
    FLITECAMSpatcalReduction = Reduction
    FLITECAMSlitcorrReduction = Reduction
    FLITECAM_ERROR = None
except ImportError as err:  # pragma: no cover
    FLITECAMImagingReduction = None
    FLITECAMSpectroscopyReduction = None
    FLITECAMWavecalReduction = None
    FLITECAMSpatcalReduction = None
    FLITECAMSlitcorrReduction = None
    FLITECAM_ERROR = err
else:
    FLITECAM_ERROR = None


class SOFIAChooser(Chooser):
    """
    Choose SOFIA Redux reduction objects.

    Currently, HAWC+, FORCAST, FIFI-LS, and FLITECAM instruments
    are fully supported.

    Input data that cannot be read as a FITS file by `astropy.io.fits`
    is ignored.  If there is no good data to reduce, a null value
    (no reduction object) is returned.

    HAWC data is determined by the value of the INSTRUME keyword.
    If it is set to 'HAWC_PLUS', a HAWCReduction object is
    instantiated and returned.  This reduction object handles all
    instrument modes for the HAWC DRP pipeline.

    FORCAST data has keyword INSTRUME = 'FORCAST'.  Imaging data is
    determined from the value of the SPECTEL1 and 2 keywords: if
    the primary filter is not a known grism, then the data is
    assumed to be imaging.  In this case, a FORCASTImagingReduction
    object is instantiated and returned.  Otherwise, a
    FORCASTSpectroscopyReduction object is returned.

    FIFI-LS data has keyword INSTRUME = 'FIFI-LS'.  A FIFILSReduction
    object is returned for all FIFI-LS data.

    FLITECAM data has keyword INSTRUME = 'FLITECAM'.  Imaging and
    spectroscopy types are distinguished via the INSTCFG keyword,
    and a FLITECAMImagingReduction or FLITECAMSpectroscopyReduction
    object is returned, as appropriate.

    Final EXES data products are supported for quicklook products only.
    These data have keyword INSTRUME = 'EXES' and should be either
    combined or merged spectral products (CMB, MRD).

    If input data types do not match, or if no more specific
    reduction object was found, a generic Reduction object is returned.
    """
    def __init__(self):
        """Initialize the chooser."""
        super().__init__()

        self.supported = {
            'FORCAST Imaging': FORCASTImagingReduction,
            'FORCAST Spectroscopy': FORCASTSpectroscopyReduction,
            'HAWC': HAWCReduction,
            'FIFI-LS': FIFILSReduction,
            'FLITECAM Imaging': FLITECAMImagingReduction,
            'FLITECAM Spectroscopy': FLITECAMSpectroscopyReduction,
        }

        # check for any failed imports
        for instrument in self.supported:
            if self.supported[instrument] == Reduction:  # pragma: no cover
                log.warning("{} modules not found.  "
                            "{} reductions will not "
                            "be available.".format(instrument, instrument))

    def get_key_value(self, header, key):
        """
        Get a key value from a header.

        Parameters
        ----------
        header : `astropy.io.fits.Header`
            FITS header.
        key : str
            Key to retrieve.

        Returns
        -------
        str
            String representation of the value; UNKNOWN if not found.
        """
        try:
            value = str(header[key]).strip().upper()
        except KeyError:
            value = 'UNKNOWN'
        return value

    def choose_reduction(self, data=None, config=None):
        """
        Choose a reduction object.

        Parameters
        ----------
        data : list of str, optional
            Input FITS file paths.  If not provided, None will
            be returned.
        config : str, dict, or ConfigObj, optional
            Configuration file or object.  May be any type
            accepted by the `configobj.ConfigObj` constructor.
            If present, may be used to choose specialized reductions
            for some instruments.

        Returns
        -------
        Reduction or None
            The reduction object appropriate to the input data.
        """
        reduction = Reduction()

        # return generic reduction if no data provided
        if data is None:
            return reduction

        # check for config object
        if config is not None:
            config = configobj.ConfigObj(config)

        # loop over files, reading headers and collecting key values
        test_params = None
        if type(data) is not list:
            data = [data]
        for datafile in data:
            # skip any input that isn't a file
            # (eg. a number at the top of a manifest)
            if not os.path.isfile(datafile):
                continue

            # skip any input that doesn't end in .fits
            if not datafile.endswith('.fits'):
                continue

            # try to open anything else as a FITS file
            try:
                with fits.open(datafile, mode='readonly',
                               ignore_missing_end=True) as hdul:
                    hdul.verify('silentfix')
                    header = hdul[0].header
            except (OSError, ValueError, fits.verify.VerifyError):
                # silently continue -- this may be an auxiliary file
                # that the pipeline will know how to handle
                continue

            # instrument and product type are needed for all files
            instrume = self.get_key_value(header, 'INSTRUME')
            prodtype = self.get_key_value(header, 'PRODTYPE')

            if instrume == 'FORCAST':
                # instrument and sky modes
                detchan = self.get_key_value(header, 'DETCHAN')
                instmode = self.get_key_value(header, 'INSTMODE')

                # spectel1/2 depending on detector channel
                # (0 / 1 or SW / LW, depending on date)
                if detchan == '1' or detchan == 'LW':
                    spectel = self.get_key_value(header, 'SPECTEL2')
                else:
                    spectel = self.get_key_value(header, 'SPECTEL1')

                # grism options
                spec_opt = ['FOR_G063', 'FOR_G111',
                            'FOR_G227', 'FOR_G329']
                if spectel in spec_opt:
                    instmode = 'SPEC'

                # these keys have to match to return a consistent
                # reduction object
                param = [instrume, instmode, prodtype]
            elif instrume == 'HAWC' or \
                    instrume == 'HAWC+' or \
                    instrume == 'HAWC_PLUS':
                instrume = 'HAWC'
                param = [instrume, prodtype]
            elif instrume == 'FIFI-LS':
                obstype = self.get_key_value(header, 'OBSTYPE')
                detchan = self.get_key_value(header, 'DETCHAN')
                param = [instrume, prodtype, obstype, detchan]
            elif instrume == 'EXES':
                # ignore product type mismatch for quicklook
                param = [instrume]
            elif instrume == 'FLITECAM':
                instcfg = self.get_key_value(header, 'INSTCFG')
                param = [instrume, prodtype, instcfg]
            else:
                param = [instrume, prodtype]

            if test_params is None:
                test_params = param
            else:
                if param != test_params:
                    log.warning('Files do not match; using '
                                'generic reduction.')
                    log.info('  File: {}'.format(datafile))
                    log.info('  Current parameters: {}'.format(param))
                    log.info('  Previous parameters: {}'.format(test_params))
                    return reduction

        # return generic if no good files found
        if test_params is None:
            log.warning("No good files found; no reduction to run.")
            return None

        # make appropriate object
        instrume = test_params[0]
        if instrume == 'FORCAST':
            if FORCAST_ERROR:  # pragma: no cover
                raise FORCAST_ERROR
            instmode, prodtype = test_params[1:]
            if instmode == 'SPEC':
                reduction = FORCASTSpectroscopyReduction()

                # check for specialized mode
                if config is not None:
                    if 'wavecal' in config and config['wavecal']:
                        reduction = FORCASTWavecalReduction()
                    elif 'spatcal' in config and config['spatcal']:
                        reduction = FORCASTSpatcalReduction()
                    elif 'slitcorr' in config and config['slitcorr']:
                        reduction = FORCASTSlitcorrReduction()
            else:
                reduction = FORCASTImagingReduction()
        elif instrume == 'HAWC':
            if HAWC_ERROR:  # pragma: no cover
                raise HAWC_ERROR
            reduction = HAWCReduction()

            # check for specialized mode
            if config is not None:
                if 'mode' in config and config['mode']:
                    reduction.override_mode = config['mode']

        elif instrume == 'FIFI-LS':
            if FIFILS_ERROR:  # pragma: no cover
                raise FIFILS_ERROR
            reduction = FIFILSReduction()

        elif instrume == 'EXES':
            # quicklook is borrowed from the forcast pipeline
            if FORCAST_ERROR:  # pragma: no cover
                raise FORCAST_ERROR
            reduction = EXESQuicklookReduction()

        elif instrume == 'FLITECAM':
            # some functionality is borrowed from the
            # forcast pipeline
            if FORCAST_ERROR:  # pragma: no cover
                raise FORCAST_ERROR
            if FLITECAM_ERROR:  # pragma: no cover
                raise FLITECAM_ERROR
            prodtype, instcfg = test_params[1:]
            if instcfg == 'IMAGING':
                reduction = FLITECAMImagingReduction()
            else:
                reduction = FLITECAMSpectroscopyReduction()

                # check for specialized mode
                if config is not None:
                    if 'wavecal' in config and config['wavecal']:
                        reduction = FLITECAMWavecalReduction()
                    elif 'spatcal' in config and config['spatcal']:
                        reduction = FLITECAMSpatcalReduction()
                    elif 'slitcorr' in config and config['slitcorr']:
                        reduction = FLITECAMSlitcorrReduction()

        return reduction
