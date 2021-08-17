# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility and convenience functions for common pipecal use cases."""

import warnings

from astropy import log
import numpy as np
import photutils

from sofia_redux.calibration.pipecal_calfac import pipecal_calfac
from sofia_redux.calibration.pipecal_error import PipeCalError
from sofia_redux.calibration.pipecal_photometry import pipecal_photometry
from sofia_redux.calibration.pipecal_rratio import pipecal_rratio
from sofia_redux.toolkit.utilities.fits import hdinsert


__all__ = ['average_za', 'average_alt', 'guess_source_position',
           'add_calfac_keys', 'add_phot_keys', 'get_fluxcal_factor',
           'apply_fluxcal', 'get_tellcor_factor', 'apply_tellcor',
           'run_photometry']


def average_za(header):
    """
    Robust average of zenith angle from FITS header.

    Keys used are ZA_START and ZA_END.  If both are good
    (>= 0), they will be averaged.  If only one is good,
    it will be returned.  If both are bad, a PipeCalError
    will be raised.

    Parameters
    ----------
    header : `astropy.io.fits.header.Header`
        FITS header containing ZA keys.

    Returns
    -------
    float
        The ZA value

    Raises
    ------
    PipeCalError
        If no valid ZA is found.
    """
    try:
        zasta = float(header['ZA_START'])
    except (ValueError, TypeError, KeyError):
        zasta = -9999
    try:
        zaend = float(header['ZA_END'])
    except (ValueError, TypeError, KeyError):
        zaend = -9999
    if zasta > 0 and zaend < 0:
        za = zasta
    elif zasta < 0 and zaend > 0:
        za = zaend
    elif zasta < 0 and zaend < 0:
        msg = 'Bad ZA value in header'
        log.error(msg)
        raise PipeCalError(msg)
    else:
        za = (zasta + zaend) / 2
    return za


def average_alt(header):
    """
    Robust average of altitude from FITS header.

    Keys used are ALTI_STA and ALTI_END.  If both are good
    (>= 0), they will be averaged.  If only one is good,
    it will be returned.  If both are bad, a PipeCalError
    will be raised.

    Input values are expected in feet; return value is in
    kilo-feet, i.e. input values are divided by 1000 before
    returning.

    Parameters
    ----------
    header : `astropy.io.fits.header.Header`
        FITS header containing altitude keys.

    Returns
    -------
    float
        The altitude value, in kilo-feet

    Raises
    ------
    PipeCalError
        If no valid altitude is found.
    """
    try:
        altsta = float(header['ALTI_STA'])
    except (ValueError, TypeError, KeyError):
        altsta = -9999
    try:
        altend = float(header['ALTI_END'])
    except (ValueError, TypeError, KeyError):
        altend = -9999
    if altsta > 0 and altend < 0:
        alt = altsta
    elif altsta < 0 and altend > 0:
        alt = altend
    elif altsta < 0 and altend < 0:
        msg = 'Bad altitude value in header'
        log.error(msg)
        raise PipeCalError(msg)
    else:
        alt = (altsta + altend) / 2

    alt /= 1000
    return alt


def average_pwv(header):
    """
    Robust average of precipitable water vapor from FITS header.

    Keys used are WVZ_STA and WVZ_END.  If both are good
    (>= 0), they will be averaged.  If only one is good,
    it will be returned.  If both are bad, a PipeCalError
    will be raised.

    Parameters
    ----------
    header : `astropy.io.fits.header.Header`
        FITS header containing PWV keys.

    Returns
    -------
    float
        The PWV value

    Raises
    ------
    PipeCalError
        If no valid PWV is found.
    """
    try:
        pwvsta = float(header['WVZ_STA'])
    except (ValueError, TypeError, KeyError):
        pwvsta = -9999
    try:
        pwvend = float(header['WVZ_END'])
    except (ValueError, TypeError, KeyError):
        pwvend = -9999
    if pwvsta > 0 and pwvend < 0:
        pwv = pwvsta
    elif pwvsta < 0 and pwvend > 0:
        pwv = pwvend
    elif pwvsta < 0 and pwvend < 0:
        msg = 'Bad PWV value in header'
        log.error(msg)
        raise PipeCalError(msg)
    else:
        pwv = (pwvsta + pwvend) / 2

    return pwv


def guess_source_position(header, image, srcpos=None):
    """
    Estimate the position of a standard source in the image.

    The following information sources are checked in order:

        1. The srcpos parameter
        2. SRCPOSX and SRCPOSY keywords in the header
        3. The brightest peak found by photutils.find_peaks.
        4. CRPIX1 and CRPIX2 keywords in the header.

    If a successful value is found at any stage, no further
    checks are done.

    Parameters
    ----------
    header : `astropy.io.fits.header.Header`
        FITS header corresponding to the image.
    image : array
        2D image data to check for peaks.
    srcpos : list or tuple, optional
        If provided, should be the desired source
        position, listed as (x,y), indexed from 0.

    Returns
    -------
    list, or None
        The estimated source position, as (x,y), zero-indexed.
        None is returned if no valid source positions could be found.
    """
    # Check the srcpos input. If it isn't provided, fill it
    # with info from the header
    if not srcpos:
        if 'SRCPOSX' in header and 'SRCPOSY' in header:
            if header['SRCPOSX'] == 0 and header['SRCPOSY'] == 0:
                srcpos = None
            else:
                srcpos = [header['SRCPOSX'], header['SRCPOSY']]
                log.debug('Found SRCPOS in header: {}'.format(srcpos))

    # if still no srcpos, try find_peaks
    if not srcpos:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            peak = photutils.find_peaks(image, np.nanmedian(image), npeaks=1)
        try:
            srcpos = [peak['x_peak'][0], peak['y_peak'][0]]
            log.debug('SRCPOS from find_peaks: {}'.format(srcpos))
        except (TypeError, IndexError):
            srcpos = None

    # if *still* no srcpos, try CRPIX
    if not srcpos:
        if 'CRPIX1' in header and 'CRPIX2' in header:
            srcpos = [header['CRPIX1'], header['CRPIX2']]
            log.debug('SRCPOS from CRPIX: {}'.format(srcpos))
        else:
            srcpos = None
            log.debug('SRCPOS not found')

    return srcpos


def add_calfac_keys(header, config):
    """
    Add calibration-related keywords to a header.

    The following keys are added or updated:
    PROCSTAT, BUNIT, CALFCTR, ERRCALF, LAMREF, LAMPIVOT,
    COLRCORR, REFCALZA, REFCALAW, and REFCALF3.

    Parameters
    ----------
    header : `astropy.io.fits.header.Header`
        The FITS header to update.
    config : dict-like
        Calibration configuration values, as produced by
        `pipecal.pipecal_config.pipecal_config`.
    """
    # assume all necessary values are present in config
    runits = config['runits']
    f3 = config['refcal_file'].partition(config['caldata'])[-1]
    try:
        pstat = str(header['PROCSTAT']).strip().upper()
    except KeyError:
        pstat = 'UNKNOWN'

    if pstat != 'LEVEL_4':
        hdinsert(header, 'PROCSTAT', 'LEVEL_3', 'Processing status')
    hdinsert(header, 'BUNIT', 'Jy/pixel', 'Data units')

    hdinsert(header, 'CALFCTR', config['calfac'],
             'Calibration factor ({}/Jy)'.format(runits))
    hdinsert(header, 'ERRCALF', config['ecalfac'],
             'Calibration factor uncertainty ({}/Jy)'.format(runits))
    hdinsert(header, 'LAMREF', config['wref'],
             'Reference wavelength (microns)')
    hdinsert(header, 'LAMPIVOT', config['lpivot'],
             'Pivot wavelength (microns)')
    hdinsert(header, 'COLRCORR', config['color_corr'],
             'Color correction factor')
    hdinsert(header, 'REFCALZA', config['rfit_am']['zaref'],
             'Reference calibration zenith angle')
    hdinsert(header, 'REFCALAW', config['rfit_alt']['altwvref'],
             'Reference calibration altitude')
    hdinsert(header, 'REFCALF3', f3,
             'Calibration reference file')


def add_phot_keys(header, phot, config=None, srcpos=None):
    """
    Add photometry-related keywords to a header.

    The primary input for this function (`phot`) should be the data
    structure returned by the pipecal_photometry function.
    All keys in this structure are added to the FITS header.

    In addition, the following keys may be added or updated,
    from the `config` or `srcpos` input:
    SRCPOSX, SRCPOSY, MODLFLX, MODLFLXE, REFSTD1, REFSTD2,
    REFSTD3, LAMREF, LAMPIVOT, COLRCORR, REFCALZA,
    REFCALAW, AVGCALFC, AVGCALER, and RUNITS.

    Parameters
    ----------
    header : `astropy.io.fits.header.Header`
        The FITS header to update.
    phot : list
        Photometry values, as produced by
        `pipecal.pipecal_photometry.pipecal_photometry`.
    config : dict-like, optional
        Calibration configuration values, as produced by
        `pipecal.pipecal_config.pipecal_config`.
    srcpos : list-like, optional
        The starting position estimate for photometry, as
        (x,y), zero-indexed.  If provided, will be set in the
        SRCPOSX and SRCPOSY keywords.
    """
    # add the starting guess position
    if srcpos is not None:
        hdinsert(header, 'SRCPOSX', srcpos[0],
                 'Initial source position for photometry (x)')
        hdinsert(header, 'SRCPOSY', srcpos[1],
                 'Initial source position for photometry (y)')

    # add photometry keys
    if phot is None:
        phot = []
    for i in range(len(phot)):
        entry = phot[i]
        key = entry['key']
        val = entry['value']
        com = entry['comment']
        if isinstance(val, list):
            hdinsert(header, key, val[0], com)
            hdinsert(header, key + 'E', val[1], 'Error in {}'.format(com))
        else:
            hdinsert(header, key, val, com)

    # add other standard values from config
    if config is not None:
        # try the model flux keys; pass if not present
        try:
            mflux = config['std_flux']
            mfluxe = mflux * config['std_eflux'] / 100
            hdinsert(header, 'MODLFLX', mflux, 'Model flux (Jy)')
            hdinsert(header, 'MODLFLXE', mfluxe, 'Model flux error (Jy)')
        except KeyError:
            pass

        # same for the standard reference files
        try:
            f1 = config['filterdef_file'].partition(config['caldata'])[-1]
            f2 = config['stdflux_file'].partition(config['caldata'])[-1]
            f3 = config['stdeflux_file'].partition(config['caldata'])[-1]
            hdinsert(header, 'REFSTD1', f1, 'Standard reference file')
            hdinsert(header, 'REFSTD2', f2, 'Standard reference file')
            hdinsert(header, 'REFSTD3', f3, 'Standard reference file')
        except KeyError:
            pass

        # assume basic reference values are present together
        if 'wref' in config:
            hdinsert(header, 'LAMREF', config['wref'],
                     'Reference wavelength (microns)')
            hdinsert(header, 'LAMPIVOT', config['lpivot'],
                     'Pivot wavelength (microns)')
            hdinsert(header, 'COLRCORR', config['color_corr'],
                     'Color correction factor')
            hdinsert(header, 'REFCALZA', config['rfit_am']['zaref'],
                     'Reference calibration zenith angle')
            hdinsert(header, 'REFCALAW', config['rfit_alt']['altwvref'],
                     'Reference calibration altitude')

        # check for an average cal factor
        if 'avgcalfc' in config:
            hdinsert(header, 'AVGCALFC', config['avgcalfc'],
                     'Average calibration factor')
            hdinsert(header, 'AVGCALER', config['avgcaler'],
                     'Average cal factor error')
            try:
                f4 = config['avgcal_file'].partition(config['caldata'])[-1]
                hdinsert(header, 'AVCLFILE', f4,
                         'Average calibration reference file')
            except KeyError:  # pragma: no cover
                pass

        # add the raw units for the instrument
        if 'runits' in config:
            hdinsert(header, 'RUNITS', config['runits'],
                     'Raw data units (before calibration)')


def get_fluxcal_factor(header, config,
                       update=False, write_history=False):
    """
    Retrieve a flux calibration factor from configuration.

    The returned factor is intended to divide raw image data
    in order to calibrate it to physical units (Jy/pixel).
    That is::

       calibrated = flux / cal_factor.

    The input header may be updated with FITS keywords,
    via `add_calfac_keys`, if desired.  A history message
    may also be added to the header, if desired.

    Parameters
    ----------
    header : `astropy.io.fits.header.Header`
        The FITS header to update.
    config : dict-like
        Calibration configuration values, as produced by
        `pipecal.pipecal_config.pipecal_config`.
    update : bool, optional
        If set, calibration keywords will be updated.
    write_history : bool, optional
        If set, and update is also True, a history message
        will be added to the header.

    Returns
    -------
    float, float
        The calibration factor and its associated error.

    """
    if not update:
        write_history = False

    if 'calfac' not in config:
        if write_history:
            msg1 = 'No reference flux calibration available'
            try:
                msg2 = 'for SPECTEL={}, ALTCFG1={}, ' \
                       'DATE={}. '.format(config['spectel'],
                                          config['altcfg1'],
                                          config['date'])
            except KeyError:
                msg2 = '. '
            msg3 = 'Data is not calibrated.'
            log.warning(msg1 + msg2 + msg3)

            header['HISTORY'] = '  '
            header['HISTORY'] = msg1
            if len(msg2) > 2:
                header['HISTORY'] = msg2
            header['HISTORY'] = msg3
            header['HISTORY'] = '  '
        calfac, ecalfac = None, None
    else:
        f1 = config['refcal_file'].partition(config['caldata'])[-1]
        calfac = config['calfac']
        ecalfac = config['ecalfac']

        # write relevant keys to header
        if update:
            add_calfac_keys(header, config)

        if write_history:
            header['HISTORY'] = '  '
            header['HISTORY'] = 'Flux calibration information:'
            header['HISTORY'] = '  Using reference file: {}'.format(f1)
            header['HISTORY'] = '  Average reference calibration factor:'
            header['HISTORY'] = '    {} +/- {}'.format(calfac, ecalfac)
            header['HISTORY'] = 'Data has been divided by cal ' \
                                'factor to convert'
            header['HISTORY'] = '  from {} to Jy'.format(config['runits'])
            header['HISTORY'] = '  '

    return calfac, ecalfac


def apply_fluxcal(data, header, config,
                  variance=None, covariance=None, write_history=True):
    """
    Apply a flux calibration factor to an image.

    The image is calibrated as::

        calibrated = flux / cal_factor

    If provided, the variance and/or covariance are propagated
    as well, as::

        calibrated_variance = variance / cal_factor^2
        calibrated_covariance = covariance / cal_factor^2

    The provided header is also updated with calibration keys and,
    optionally, a history message.

    Parameters
    ----------
    data : array
        Image to calibrate.
    header : `astropy.io.fits.header.Header`
        FITS header to update.
    config : dict-like
        Calibration configuration values, as produced by
        `pipecal.pipecal_config.pipecal_config`.
    variance : array, optional
        Variance image to propagate, if desired.
    covariance : array, optional
        Covariance image to propagate, if desired.
    write_history : bool, optional
        If set, a history message will be added to the header.

    Returns
    -------
    array, or 2- or 3-length tuple
        If only data is provided, it is returned as an array.
        If variance is provided, the return value is (data, variance).
        If covariance is provided, the return value is
        (data, variance, covariance); if the variance was not provided,
        it will be set to None.
    """

    calfac, _ = get_fluxcal_factor(header, config, update=True,
                                   write_history=write_history)

    if calfac is None:
        corrdata, corrvar, corrcovar = data, variance, covariance
    else:
        # apply factor to data
        # do not propagate error on calfactor to variance
        corrdata = data / calfac
        if variance is not None:
            corrvar = variance / calfac ** 2
        else:
            corrvar = None
        if covariance is not None:
            corrcovar = covariance / calfac ** 2
        else:
            corrcovar = None

    if covariance is not None:
        return corrdata, corrvar, corrcovar
    elif variance is not None:
        return corrdata, corrvar
    else:
        return corrdata


def get_tellcor_factor(header, config, update=False, use_wv=False):
    """
    Retrieve a telluric correction factor from configuration.

    The correction factor is calculated from the ratio of the R
    value at the reference ZA and altitude/PWV to the R value
    at the observed ZA and altitude/PWV.  R values are
    calculated by `pipecal.pipecal_rratio.pipecal_rratio`.
    The returned value is intended to be multiplied into the
    data, in order to correct it to the reference values.

    Observed values for ZA and altitude or PWV are read from the
    FITS header.

    Parameters
    ----------
    header : `astropy.io.fits.header.Header`
        The FITS header to update.
    config : dict-like
        Calibration configuration values, as produced by
        `pipecal.pipecal_config.pipecal_config`.
    update : bool, optional
        If set, calibration keywords will be updated.
    use_wv : bool, optional
        If set, precipitable water vapor will be used as the
        reference value, instead of altitude.

    Returns
    -------
    float
        The telluric correction factor.
    """
    # get ALT, ZA, PWV from header
    if use_wv:
        altwv = average_pwv(header)
    else:
        altwv = average_alt(header)
    za = average_za(header)

    # reference files
    f1 = config['rfitam_file'].partition(config['caldata'])[-1]

    zaref = config['rfit_am']['zaref']
    if use_wv:
        # R ratio for precipitable water vapor (PWV) reference
        altwv_str = 'PWV'
        f2 = config['rfitpwv_file'].partition(config['caldata'])[-1]
        altwvref = config['rfit_pwv']['altwvref']

        rratio = pipecal_rratio(za, altwv, zaref, altwvref,
                                config['rfit_am']['coeff'],
                                config['rfit_pwv']['coeff'],
                                pwv=True)
        ref_rratio = pipecal_rratio(zaref, altwvref, zaref, altwvref,
                                    config['rfit_am']['coeff'],
                                    config['rfit_pwv']['coeff'],
                                    pwv=True)
    else:
        # R ratio for altitude reference
        altwv_str = 'altitude'
        f2 = config['rfitalt_file'].partition(config['caldata'])[-1]
        altwvref = config['rfit_alt']['altwvref']

        rratio = pipecal_rratio(za, altwv, zaref, altwvref,
                                config['rfit_am']['coeff'],
                                config['rfit_alt']['coeff'],
                                pwv=False)
        ref_rratio = pipecal_rratio(zaref, altwvref, zaref, altwvref,
                                    config['rfit_am']['coeff'],
                                    config['rfit_alt']['coeff'],
                                    pwv=False)

    # correction factor to reference Alt/ZA
    corr_fac = ref_rratio / rratio

    # update header with keywords
    if update:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            hdinsert(header, 'TELCORR', corr_fac,
                     'Telluric correction factor to ZA/{}'.format(altwv_str))
            hdinsert(header, 'REFCALZA', zaref,
                     'Reference calibration zenith angle')
            hdinsert(header, 'REFCALAW', altwvref,
                     'Reference calibration {}'.format(altwv_str))
            hdinsert(header, 'REFCALF1', f1,
                     'Calibration reference file')
            hdinsert(header, 'REFCALF2', f2,
                     'Calibration reference file')

        # update header with history
        header['HISTORY'] = 'Using reference files:'
        header['HISTORY'] = '  {}'.format(f1)
        header['HISTORY'] = '  {}'.format(f2)
        header['HISTORY'] = 'Reference ZA={}, ' \
                            '{}={}'.format(zaref, altwv_str, altwvref)
        header['HISTORY'] = 'Telluric correction factor for ZA={:.2f}, ' \
                            '{}={}:'.format(za, altwv_str, altwv)
        header['HISTORY'] = '  {}'.format(corr_fac)

    return corr_fac


def apply_tellcor(data, header, config, variance=None, covariance=None):
    """
    Apply a telluric correction factor to an image.

    The image is corrected as::

        corrected = flux * tel_factor

    If provided, the variance and/or covariance are propagated
    as well, as::

        corrected_variance = variance * tel_factor^2
        corrected_covariance = covariance / tel_factor^2

    The provided header is also updated with relevant keywords and
    a history message.

    Parameters
    ----------
    data : array
        Image to correct.
    header : `astropy.io.fits.header.Header`
        FITS header to update.
    config : dict-like
        Calibration configuration values, as produced by
        `pipecal.pipecal_config.pipecal_config`.
    variance : array, optional
        Variance image to propagate, if desired.
    covariance : array, optional
        Covariance image to propagate, if desired.

    Returns
    -------
    array, or 2- or 3-length tuple
        If only data is provided, it is returned as an array.
        If variance is provided, the return value is (data, variance).
        If covariance is provided, the return value is
        (data, variance, covariance); if the variance was not provided,
        it will be set to None.

    Raises
    ------
    PipeCalError
        If no valid telluric correction factor is found.
    """

    # correction factor to reference Alt/ZA
    try:
        corr_fac = get_tellcor_factor(header, config, update=True)
    except (KeyError, AttributeError, ValueError, TypeError) as err:
        log.debug(str(err))
        raise PipeCalError('Response data not found; cannot apply '
                           'telluric correction') from None

    # correct data and variance
    corrdata = data * corr_fac
    if variance is not None:
        corrvar = variance * corr_fac**2
    else:
        corrvar = None
    if covariance is not None:
        corrcovar = covariance * corr_fac**2
    else:
        corrcovar = None

    if covariance is not None:
        return corrdata, corrvar, corrcovar
    elif variance is not None:
        return corrdata, corrvar
    else:
        return corrdata


def run_photometry(data, header, var, config, **kwargs):
    """
    Run photometry on an image of a standard source.

    Input images are assumed to be telluric-corrected
    (e.g. by the apply_tellcor function).

    If the image is uncalibrated, and a model flux is known for
    the source, values from aperture photometry
    will be used to attempt to calculate and record a reference
    calibration factor.  This value and its associated error
    will be recorded in REFCALFC and REFCALER, respectively.

    If the image is previously calibrated, and a model flux is
    known, the values will be compared, and a log message will
    display the percent difference from the model.

    If a model flux is unknown, the source flux will be reported
    and recorded in the FITS header, but no further calculations
    will be performed.

    Parameters
    ----------
    data : array
        Image to correct.
    header : `astropy.io.fits.header.Header`
        FITS header to update.
    var : array
        Variance image to propagate, if desired.
    config : dict-like
        Calibration configuration values, as produced by
        `pipecal.pipecal_config.pipecal_config`.
    kwargs : dict-like, optional
        Extra parameters to provide to the photometry function
        If not provided, but values are available in
        `config`, the configuration values will be used as
        the default.  If parameters are still not present,
        defaults defined by the pipecal_photometry function will
        be used.
    """
    # get provided parameters, or reasonable defaults
    if 'srcpos' not in kwargs:
        kwargs['srcpos'] = guess_source_position(header, data)
    if 'aprad' not in kwargs:
        if 'aprad' in config:
            kwargs['aprad'] = config['aprad']
    if 'skyrad' not in kwargs:
        try:
            kwargs['skyrad'] = [config['bgin'], config['bgout']]
        except KeyError:
            pass
    if 'fwhm' not in kwargs:
        if 'fwhm' in config:
            kwargs['fwhm'] = config['fwhm']
    if 'fitsize' not in kwargs:
        if 'fitsize' in config:
            kwargs['fitsize'] = config['fitsize']
    if 'runits' not in kwargs:
        if 'runits' in config:
            kwargs['runits'] = config['runits']

    phot = pipecal_photometry(data, var, **kwargs)

    # update header
    add_phot_keys(header, phot, config, kwargs['srcpos'])

    if ('BUNIT' in header and 'jy' in
            str(header['BUNIT']).lower().strip()):
        calib = True
    else:
        calib = False

    # log the position and flux
    try:
        log.info('Source: {}'.format(config['object']))
    except KeyError:
        pass
    try:
        log.info('Source Position (x,y): '
                 '{:.2f}, {:.2f}'.format(header['STCENTX'],
                                         header['STCENTY']))
        if calib:
            log.info('Source Flux: '
                     '{:.2f} +/- {:.2f} Jy'.format(header['STAPFLX'],
                                                   header['STAPFLXE']))
        else:
            try:
                runits = kwargs['runits']
            except KeyError:
                runits = 'counts'
            log.info('Source Flux: '
                     '{:.2f} +/- {:.2f} {}'.format(header['STAPFLX'],
                                                   header['STAPFLXE'],
                                                   runits))
    except KeyError:
        log.warning('Photometry failed.')
        return

    # if model flux is available, calculate ref cal factor
    if 'std_flux' not in config:
        log.warning('No model flux available; not calculating '
                    'reference cal factor.')
    else:
        flux = header['STAPFLX']
        flux_err = header['STAPFLXE']

        # log the model flux
        try:
            modlflx = header['MODLFLX']
            log.info('Model Flux: '
                     '{:.3f} +/- {:.3f} Jy'.format(modlflx,
                                                   header['MODLFLXE']))
            if calib:
                log.info('Percent difference: '
                         '{:.1f}%'.format(100 * (flux - modlflx) / modlflx))
        except KeyError:
            pass

        try:
            ref_calfac, ref_ecalfac = \
                pipecal_calfac(flux, flux_err, config)
            hdinsert(header, 'REFCALFC', ref_calfac,
                     'Reference calibration factor')
            hdinsert(header, 'REFCALER', ref_ecalfac,
                     'Reference calibration factor error')
            if not calib:
                log.info('Calculated calibration factor: '
                         '{:.4f}'.format(ref_calfac))
        except ValueError:
            log.warning('Bad flux; not adding REFCALFC.')
