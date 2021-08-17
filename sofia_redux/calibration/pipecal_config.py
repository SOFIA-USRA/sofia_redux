# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Calibration configuration."""

import re
import os

import numpy as np
from astropy import log

from sofia_redux.calibration.pipecal_error import PipeCalError

__all__ = ['pipecal_config', 'read_respfile']


class ParseConfigError(PipeCalError):
    """Standard exception for reading response files."""
    def __init__(self, message):
        message = f'ERROR: Badly formatted response file: {message}'
        super().__init__(message)


def read_respfile(fname, spectel):
    """
    Read response files.

    Response files contain the coefficients of polynomial fits to
    standard atmospheric models, folded through instrumental response models
    for a particular instrument and mode, at a range of wavelength
    band-passes (instrument filters).

    The format is assumed to be:

        - two lines beginning with '#' that contain reference
          values used for the response fit, e.g.::

              # ALTMIN=35.0 ALTMAX=45.0 ALTREF=41.0
              # ZAMIN=30.0 ZAMAX=70.0 ZAREF=45.0

        - one line for each filter containing:

            - the filter reference wavelength
            - the filter name
            - a reference response value
            - any number of polynomial coefficients, beginning
              with the constant term

    The filter name is matched to the provided `spectel`, and
    the corresponding fit coefficients are returned, along with the
    reference values.

    Parameters
    ----------
    fname : string
        Full path name of response file.
    spectel : string
        Name of filter used.

    Returns
    -------
    resp_config : dictionary
        A dictionary with the details of the filter's response.
        Keys are: respref, altwvref, altwvrange, zaref, zarange,
        coeff.

    Raises
    ------
    PipeCalError
        If errors are found while reading or parsing the file.
    """

    # Read file
    try:
        respfile = open(fname, 'r')
    except IOError:
        message = f'Cannot open {fname}'
        log.error(message)
        raise PipeCalError(message)

    # First two lines contain alt/za information:
    # ALTMIN=35.0 ALTMAX=45.0 ALTREF=41.0
    # ZAMIN=30.0 ZAMAX=70.0 ZAREF=45.0
    line = respfile.readline().strip()
    try:
        values = [float(i.split('=')[-1])
                  for i in line.split() if i != '#']
        if len(values) == 3:
            altrange = [values[0], values[1]]
            altref = values[2]
        else:
            raise ValueError('missing values')

        line = respfile.readline().strip()
        values = [float(i.split('=')[-1]) for i in line.split()
                  if i != '#']

        if len(values) == 3:
            zarange = [values[0], values[1]]
            zaref = values[2]
        else:
            raise ValueError('missing values')

    except ValueError:
        message = 'Could not read reference values from response file header'
        log.error(message)
        raise PipeCalError(message)

    # Read the rest of the file, stopping after finding matching spectel
    coeff = []
    respref = None
    for line in respfile:

        # If line is empty or starts with #, skip it
        if line.strip() != '' and not line.startswith('#'):
            # Split the line by whitespace.
            # If there are less than 4 fields, raise an error
            splitline = line.split()
            if len(splitline) < 4:
                raise ParseConfigError(fname)

            # Pull out the second field, this is the filter name (string)
            # Check if it matches spectel
            lam = splitline[1].upper()
            if lam == spectel:
                # If a match, pull out the third field, this is the
                # response reference (float). It if isn't a float,
                # raise an error
                try:
                    respref = float(splitline[2])
                except ValueError:
                    raise ParseConfigError(fname)

                # The rest of the line are a list of coefficients
                try:
                    coeff = [float(i) for i in splitline[3:]]
                except ValueError:
                    raise ParseConfigError(fname)

                # Found the first matching spectel, so break
                break

    respfile.close()

    if coeff:
        resp_config = {'respref': respref,
                       'altwvref': altref,
                       'altwvrange': altrange,
                       'zaref': zaref,
                       'zarange': zarange,
                       'coeff': coeff}
    else:
        resp_config = {}

    return resp_config


def _get_cal_path(instrument):
    """Helper function to retrieve the caldata path."""
    # Directory 2 steps up from where code is located
    pkgpath = (os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
               + os.path.sep)
    # Join in the instrument name
    caldata = os.path.join(pkgpath, 'calibration',
                           'data', instrument.lower(), '')
    return caldata


def pipecal_config(header):
    """
    Parse all reference files and return appropriate configuration values.

    Parameters
    ----------
    header : `astropy.io.fits.header.Header` or dict-like
        Header of FITS file.

    Returns
    -------
    config : dictionary
        Key values pulled from the header and calibration files needed
        for photometry. Possible keys are:

            - *caldata* : Path to calibration package data directory
            - *date* : Observation date
            - *instrument* : Instrument used to take observation
            - *runits* : Raw (uncalibrated) units for the instrument
            - *spectel* : Filter name for observation
            - *altcfg1* : Alternate configuration (0, 1, 2, or 3). For FORCAST,
              altcfg1=1 means dual mode with Barr2 dichroic, 2 means
              dual with Barr3, and 0 means single mode. For FLITECAM,
              altcfg1=1 means FLIPO configuration, 0 means FLITECAM only.
              For HAWC+, altcfg1=0 is chop/nod with HWP, altcfg1=1 is
              chop/nod without HWP, altcfg1=2 is scan with HWP, altcfg1=3
              is scan without HWP.
            - *object* : Object name
            - *filterdef_file* : File path to filter definition file
            - *wref* : Reference wavelength, as defined by filterdef_file
            - *lpivot* : Pivot wavelength, as defined by filterdef_file
            - *color_corr* : Color correction, as defined by filterdef_file
            - *aprad* : Aperture radius, as defined by filterdef_file
            - *bgin* : Background annulus inner radius, as defined
              by filterdef_file
            - *bgout* : Background annulus outer radius, as defined
              by filterdef_file
            - *fwhm* : FWHM, as defined by filterdef_file
            - *fitsize* : Subimage size for fits, as defined by filterdef_file
            - *refcal_file* : File path to reference cal factor file
            - *calfac* : Series average reference cal factor, as defined
              in refcal_file
            - *ecalfac* : Error on series average reference cal factor, as
              defined in refcal_file
            - *avgcal_file* : File path to average reference cal factor file
            - *avgcalfc* : Average reference cal factor, as defined
              in refcal_file
            - *avgcaler* : Error on average reference cal factor, as
              defined in refcal_file
            - *rfitam_file* : File path to response fit for airmass file
            - *rfit_am* : A dictionary containing airmass response fit
              coefficients and metadata, as read from rfitam_file
              by read_respfile
            - *rfitalt_file* : File path to response fit for altitude file
            - *rfit_alt* : A dictionary containing altitude response
              fit coefficients and metadata, as read from rfitalt_file
              by read_respfile
            - *rfitpwv_file* : File path to response fit for water vapor file
            - *rfit_pwv* : A dictionary containing water vapor response
              fit coefficients and metadata, as read from rfitam_file
              by read_respfile
            - *stdeflux_file* : File path to model error file
            - *std_eflux* : Percent error on the model of the standard
              observed, as read from stdeflux_file
            - *std_scale* : Scale factor to apply to the model of the
              standard observed, as read from stdeflux_file
            - *stdflux_file* : File path to standard model file
            - *std_flux* : Model flux of standard at observed wavelength

        Keywords are populated only if matching configuration data
        is found. In particular, standard model data is returned only
        if the observation object name matches a known standard, as
        defined in the stddefault.txt file in the data directory.
    """

    # Config is the data structure that'll be filled and returned
    config = dict()

    # Read configuration from header
    try:
        instrument = header['INSTRUME'].strip().upper()
        spectel1 = header['SPECTEL1'].strip().upper()
        spectel2 = header['SPECTEL2'].strip().upper()
        instcfg = header['INSTCFG'].strip().upper()
        obj = header['OBJECT'].strip().upper().replace(' ', '')
    except KeyError:
        log.error('Missing required keywords for pipecal configuration.')
        return config

    # Set the caldata path by instrument
    caldata = _get_cal_path(instrument)

    # Store one directory up (the /data path)
    config['caldata'] = os.path.join(
        os.path.dirname(os.path.dirname(caldata)), '')

    # Read the observation date from the file
    date_str = header['DATE-OBS'].strip()
    date_arr = re.split(r'[-T]', date_str)
    dateobs = 99999999
    if len(date_arr) > 2:
        date_join = ''.join(date_arr[:3])
        try:
            dateobs = int(date_join)
        except ValueError:
            pass
    config['date'] = dateobs

    # Read the instrument configuration
    if instrument == 'FORCAST':
        config['runits'] = 'Me/s'
        detchan = str(header['DETCHAN']).strip().upper()
        dichroic = header['DICHROIC'].strip().upper()
        if detchan == '1' or detchan == 'LW':
            spectel = spectel2
        else:
            spectel = spectel1
        # Set ALTCFG1, based on INSTRCFG and DICRHOIC
        # 0 = single, 1 = Barr2, 2 = Barr3
        if instcfg == 'IMAGING_DUAL':
            if re.match(r'.*BARR.*3.*', dichroic):
                altcfg1 = 2
            else:
                altcfg1 = 1
        else:
            altcfg1 = 0
    elif instrument == 'HAWC_PLUS':
        config['runits'] = 'counts'
        spectel = spectel1
        hwp = spectel2
        instmode = str(header['INSTMODE']).strip().upper()
        # Set ALTCFG1, based on INSTMODE
        # 0 = chop/nod, 1 = scan
        if instmode == 'OTFMAP':
            if 'OPEN' in hwp or hwp == 'UNKNOWN':
                altcfg1 = 3
            else:
                altcfg1 = 2
        else:
            if 'OPEN' in hwp or hwp == 'UNKNOWN':
                altcfg1 = 1
            else:
                altcfg1 = 0
    elif instrument == 'FLITECAM':
        config['runits'] = 'ct/s'
        spectel = spectel1

        # Read instrument config from MISSN-ID
        # altcfg1: 1 => FLIPO, 0 => just FLITECAM
        missnid = str(header['MISSN-ID']).strip().upper()
        if 'FP' in missnid:
            altcfg1 = 1
        else:
            altcfg1 = 0
    else:
        log.error('Unsupported instrument.')
        return config

    config['instrument'] = instrument
    config['spectel'] = spectel
    config['altcfg1'] = altcfg1

    # Remove _ and - from object name
    obj = ''.join(re.split(r'[_-]', obj)).upper()
    config['object'] = obj

    # Read the calibration default table
    default_file = os.path.join(caldata, 'caldefault.txt')
    try:
        def_cal = np.genfromtxt(default_file, comments='#',
                                dtype=np.unicode_, unpack=True)
        date, ac1, fdef, std_eflux, ref_calf, avg_calf, \
            rfit_am, rfit_alt, rfit_pwv = def_cal
    except IOError:
        log.error('Calibration default file {} does '
                  'not exist'.format(default_file))
        return config
    except ValueError:
        log.error('Calibration default file {} is poorly '
                  'formatted. Verify structure.'.format(default_file))
        return config

    # set all wildcard values to current config
    date = date.astype(int)
    ac1[ac1 == '.'] = altcfg1
    ac1 = ac1.astype(int)

    # Find the appropriate line in table
    idx = np.where((date >= dateobs) & (ac1 == altcfg1))[0]
    count = len(idx)
    if count == 0:
        log.error('No pipecal data found for date {}'.format(dateobs))
        return config

    # Take the first applicable date
    idx = idx[0]

    # Read the filter definition file
    # Columns: spectel, altcfg1, lambda_mean, lamda_pivot,
    #     color_correction, aperture radius, background
    #     annulus inner radius, background annulus outer
    #     radius
    # The color_correction, if not 1.0, is to correct for color leaks
    # in the filter
    fname = os.path.join(caldata, fdef[idx])
    if fdef[idx] != '.' and os.path.isfile(fname):
        fname = os.path.abspath(fname)
        config['filterdef_file'] = fname
        with open(fname, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                lval = line.split()
                spec = lval[0]
                try:
                    ac1 = int(lval[1])
                except ValueError:
                    ac1 = str(lval[1]).strip()
                if spec == spectel and (ac1 == '.' or ac1 == altcfg1):
                    config['wref'] = float(lval[2])
                    config['lpivot'] = float(lval[3])
                    config['color_corr'] = float(lval[4])
                    config['aprad'] = float(lval[5])
                    config['bgin'] = float(lval[6])
                    config['bgout'] = float(lval[7])
                    config['fwhm'] = float(lval[8])
                    config['fitsize'] = int(lval[9])
                    break

    # Read reference cal factors
    fname = os.path.join(caldata, ref_calf[idx])
    if ref_calf[idx] != '.' and os.path.isfile(fname):
        fname = os.path.abspath(fname)
        config['refcal_file'] = fname
        # Read the ref calfactor file
        # Columns: spectel, altcfg1, ref_calfac, ref_calfac_err
        try:
            spec, ac1, calf, ecalf = np.genfromtxt(fname, dtype=str,
                                                   unpack=True)
        except (ValueError, TypeError):
            log.error('Reference calibration factor file {} is poorly '
                      'formatted. Verify structure.'.format(fname))
            return config
        spec = np.array([i.strip().upper() for i in spec])
        ac1 = np.array([int(i) for i in ac1])
        cidx = np.where((spec == spectel) & (ac1 == altcfg1))[0]
        if len(cidx) > 0:
            fields = 'calfac ecalfac'.split()
            cols = [calf, ecalf]
            for field, col in zip(fields, cols):
                try:
                    config[field] = float(col[cidx[0]])
                except ValueError:
                    pass

    # Read average cal factors
    fname = os.path.join(caldata, avg_calf[idx])
    if avg_calf[idx] != '.' and os.path.isfile(fname):
        fname = os.path.abspath(fname)
        config['avgcal_file'] = fname
        # Read the avg calfactor file
        # Columns: spectel, altcfg1, ref_calfac, ref_calfac_err
        try:
            spec, ac1, calf, ecalf = np.genfromtxt(fname, dtype=str,
                                                   unpack=True)
        except (ValueError, TypeError):
            log.error('Reference calibration factor file {} is poorly '
                      'formatted. Verify structure.'.format(fname))
            return config
        spec = np.array([i.strip().upper() for i in spec])
        ac1 = np.array([int(i) for i in ac1])
        cidx = np.where((spec == spectel) & (ac1 == altcfg1))[0]
        if len(cidx) > 0:
            fields = ['avgcalfc', 'avgcaler']
            cols = [calf, ecalf]
            for field, col in zip(fields, cols):
                try:
                    config[field] = float(col[cidx[0]])
                except ValueError:
                    pass

    # Read response fit for airmass
    fname = os.path.join(caldata, rfit_am[idx])
    if rfit_am[idx] != '.' and os.path.isfile(fname):
        fname = os.path.abspath(fname)
        config['rfitam_file'] = fname
        rfit_am = read_respfile(fname, spectel)
        if len(rfit_am) > 0:
            config['rfit_am'] = rfit_am

    # Read response fit for altitude
    fname = os.path.join(caldata, rfit_alt[idx])
    if rfit_alt[idx] != '.' and os.path.isfile(fname):
        fname = os.path.abspath(fname)
        config['rfitalt_file'] = fname
        rfit_alt = read_respfile(fname, spectel)
        if len(rfit_alt) > 0:
            config['rfit_alt'] = rfit_alt

    # Read response fit for water vapor monitor
    fname = os.path.join(caldata, rfit_pwv[idx])
    if rfit_pwv[idx] != '.' and os.path.isfile(fname):
        fname = os.path.abspath(fname)
        config['rfitpwv_file'] = fname
        rfit_pwv = read_respfile(fname, spectel)
        if len(rfit_pwv) > 0:
            config['rfit_pwv'] = rfit_pwv

    # Read standard flux error
    fname = os.path.join(caldata, std_eflux[idx])
    if std_eflux[idx] != '.' and os.path.isfile(fname):
        fname = os.path.abspath(fname)
        config['stdeflux_file'] = fname

        # Read the model_error file
        # Columns: object_name, %model_error
        try:
            ob, merr, mscale = np.genfromtxt(fname, dtype=np.unicode_,
                                             unpack=True)
        except (ValueError, TypeError):
            log.error('Standard error file {} is poorly '
                      'formatted. Verify structure.'.format(fname))
            return config
        test = np.array([o.strip().upper() in obj for o in ob])
        cidx = np.where(test)[0]
        if len(cidx) > 0:
            fields = 'std_eflux std_scale'.split()
            cols = [merr, mscale]
            for field, col in zip(fields, cols):
                try:
                    config[field] = float(col[cidx[0]])
                except ValueError:
                    pass

    # Read the standard flux defaul table
    # Columns: date, altcfg1, object, flux_file
    default_file = os.path.join(caldata, 'stddefault.txt')
    try:
        date, ac1, ob, std_flux = np.genfromtxt(
            default_file, dtype=np.unicode_,
            unpack=True)
    except IOError:
        log.error('Standards default file {} does not '
                  'exist'.format(default_file))
        return config
    except ValueError:
        log.error('Standards default file {} is incorrectly '
                  'formatted.'.format(default_file))
        return config
    else:
        ac1[ac1 == '.'] = altcfg1
        date = np.array([int(i) for i in date])
        ac1 = np.array([int(i) for i in ac1])

    # Find appropriate line in table
    test = np.array([o.strip().upper() in obj for o in ob])
    idx = np.where((date >= dateobs) & (ac1 == altcfg1) & test)[0]

    if len(idx) == 0:
        return config

    # Take the first applicable date
    idx = idx[0]

    # Read the standard flux file
    fname = os.path.join(caldata, std_flux[idx])
    if std_flux[idx] != '.' and os.path.isfile(fname) and \
            'wref' in config.keys():
        fname = os.path.abspath(fname)
        config['stdflux_file'] = fname

        # Read the standard flux file
        # Columns: lambda_ref, lambda_mean, lambda_l, lambda_pivot,
        #          lambda_eff, lambda_eff_ph, lambda_io, width, Response,
        #          F_mean, Fnu_mean, Fnu_lammean, ColorTerm, ColorTerm,
        #          Total Count Rate, Source Size, Source FWHM, Bkgd,
        #          S/N, Fnu_limit, Filter
        # Only use mean wavelength and Fnu_mean
        # There are 6 rows of headers to skip
        try:
            lmean, fmean = np.genfromtxt(fname, usecols=(1, 10),
                                         unpack=True, skip_header=6)
        except ValueError:
            log.error('Standard flux file {} is poorly '
                      'formatted. Verify structure.'.format(fname))
            return config
        cidx = np.where(np.abs(lmean - config['wref']) < 1e-2)[0]

        if len(cidx) > 0:
            cidx = cidx[0]
            std_flux = float(fmean[cidx])
            if np.isnan(std_flux):
                log.error('Bad standard flux value')
            else:
                if 'std_scale' in config.keys():
                    std_flux *= config['std_scale']
                config['std_flux'] = std_flux
        else:
            log.warning('Standard flux not found for '
                        'wavelength {}'.format(config['wref']))

    return config
