# Licensed under a 3-clause BSD style license - see LICENSE.rst

from collections import OrderedDict
import os

from astropy import log
import pandas

import sofia_redux.instruments.forcast as drip
from sofia_redux.instruments.forcast.getpar import getpar
from sofia_redux.instruments.forcast.readmode import readmode

__all__ = ['getcalpath']


def getcalpath(header, pathcal=None):
    """
    Return the path of the ancillary files used for the pipeline.

    Looks up default calibration files and other reference data in
    the calibration data path, based on the characteristics of the
    observation, as recorded in the input header.

    Parameters
    ----------
    header : astropy.io.fits.header.Header
    pathcal : str, optional
        File path to calibration data directory that contains the default
        calibration files.  If not provided, the default data location
        will be used (sofia_redux.instruments.forcast/data).

    Returns
    -------
    collections.OrderedDict
        name : str
            Name of the config mode.
        gmode : int
            The grism mode to be used during the reduction.
        cnmode : int
            The chop/nod mode to be used during the reduction.
        obstype : str
            Observation type.
        srctype : str
            Source type.
        spectel : str
            Name of the spectral element used for the observation.
        slit : str
            Name of the slit used for the observation.
        pathcal : str
            Base file path for calibration data.
        conffile : str
            File path to the configuration file.
        kwfile : str
            File path to the keyword definition file.
        badfile : str
            File path to the bad pixel mask.
        pinfile : str
            File path to the pinhole mask locations.
        pixshiftx : float
            Shift to add to CRPIX1 to account for filter shifts.
        pixshifty : float
            Shift to add to CRPIX2 to account for filter shifts.
        maskfile : str
            File path to the grism order mask file.
        wavefile : str
            File path to the grism wavelength calibration file.
        resolution : float
            Spectral resolution for the grism mode
        respfile : str
            File path to the grism instrument response file.
        slitfile : float
            File path to the grism slit function file.
        linefile : str
            File path to a line list for wavelength calibration.
        wmin : int
            Approximate minimum wavelength for the instrument. Used
            to identify matching ATRAN files.
        wmax : int
            Approximate maximum wavelength for the instrument. Used
            to identify matching ATRAN files.
        error : bool
            True if there was an error retrieving the files.
    """
    if isinstance(pathcal, str):
        path = pathcal[:]
    else:
        path = os.path.join(os.path.dirname(drip.__file__), 'data')

    result = OrderedDict(
        (('name', ''), ('gmode', -1), ('cnmode', ''), ('dateobs', 99999999),
         ('obstype', ''), ('srctype', ''), ('spectel', ''), ('slit', ''),
         ('boresight', ''), ('pathcal', ''), ('conffile', ''),
         ('kwfile', ''), ('badfile', ''), ('pinfile', ''),
         ('pixshiftx', 0.), ('pixshifty', 0.),
         ('maskfile', ''), ('wavefile', ''), ('resolution', 0.),
         ('respfile', ''), ('slitfile', ''), ('linefile', ''),
         ('wmin', 4), ('wmax', 50),
         ('error', False)))

    detchan = getpar(header, 'DETCHAN', update_header=False, dripconf=False)
    camera = 'LWC' if detchan == 'LW' else 'SWC'

    spectel = [None, None]
    for idx, key in enumerate(['SPECTEL1', 'SPECTEL2']):
        value = header.get(key)
        if value is None or str(value).strip().upper() == '0':
            msg = ('getcalpath - Problem reading SPECTEL%i' % (idx + 1))
            msg += '\nRoutine failed. Returning default dict'
            log.warning(msg)
            result['error'] = True
            return result
        spectel[idx] = value

    result['spectel'] = spectel[0] if camera == 'SWC' else spectel[1]
    result['slit'] = header.get('SLIT', '').upper().strip()
    result['boresight'] = header.get('BORESITE', '').upper().strip()

    date = header.get('DATE-OBS', '').replace('T', ' ').replace('-', ' ')
    date = date.split()
    dateobs = 99999999
    if len(date) >= 3:
        try:
            dateobs = int(''.join(date[0:3]))
        except ValueError:
            pass
    result['dateobs'] = dateobs

    gmode_name = {'FOR_G063': (2, 'G063'),
                  'FOR_G111': (3, 'G111'),
                  'FOR_G227': (4, 'G227'),
                  'FOR_G329': (5, 'G329')}
    result['gmode'], result['name'] = gmode_name.get(
        result['spectel'], (-1, 'IMG_%s' % camera))

    # also read and store the sky mode, obstype, and srctype
    # from the header
    result['cnmode'] = readmode(header)
    obstype = getpar(header, 'OBSTYPE', default='OBJECT',
                     update_header=False, dripconf=False)
    result['obstype'] = obstype.strip().upper()
    srctype = getpar(header, 'SRCTYPE', default='UNKNOWN',
                     update_header=False, dripconf=False)
    result['srctype'] = srctype.strip().upper()

    result['pathcal'] = os.path.join(path, '')
    calfile_default = os.path.join(path, 'caldefault.txt')
    if not os.path.isfile(calfile_default):
        msg = "getcalpath - Problem reading default file %s" % calfile_default
        msg += "\nRoutine failed. Returning default dict"
        log.warning(msg)
        result['error'] = True
        return result

    calcols = ['name', 'conffile', 'kwfile', 'badfile', 'pinfile']
    df = pandas.read_csv(calfile_default, delim_whitespace=True,
                         comment='#', names=calcols, index_col=0)
    table = df[(df.index >= dateobs) & (df['name'] == result['name'])]
    if len(table) == 0:
        msg = 'getcalpath - Problem reading defaults for %s ' % result['name']
        msg += 'on date %s from %s' % (dateobs, calfile_default)
        log.warning(msg + '\nRoutine failed. Returning default dict')
        result['error'] = True
        return result
    # take the first applicable date
    row = table[table.index == table.index.min()].sort_index().iloc[0]
    for f in calcols:
        if f.endswith('file') and f in row and row[f] != '.':
            result[f] = os.path.join(path, row[f])

    # Read the filter shift_image table
    # This file lists the shift_image values that should be applied to CRPIX1
    # and CRPIX2 to compensate for shifts in the pixel position of a
    # source relative to the 11.1um filter, cause by optical effects
    # that vary for different filters.
    dual = header.get('INSTCFG', '').upper().strip() == 'IMAGING_DUAL'
    filter_file_default = os.path.join(path, 'filtershift.txt')
    filtcols = ['spectel', 'dichroic', 'shiftx', 'shifty']
    df = pandas.read_csv(filter_file_default, delim_whitespace=True,
                         comment='#', names=filtcols, index_col=0)
    table = df[(df.index > dateobs)
               & (df['spectel'] == result['spectel'])
               & (df['dichroic'] == dual)].sort_index()
    if len(table) != 0:
        row = table[
            table.index == table.index.min()].sort_index().iloc[0].fillna(0.)
        result['pixshiftx'] = row['shiftx']
        result['pixshifty'] = row['shifty']

    # Read additional grism defaults into result
    if result['gmode'] > 0:
        _get_grism_cal(path, result)

    return result


def _get_grism_cal(pathcal, result):
    """
    Return the path of the ancillary files used for the grism pipeline.

    Looks up additional default calibration files and other reference data
    needed for grism observations.  The input parameters are assumed to be
    passed from the getcalpath function: they should already be formatted
    as necessary.

    Parameters
    ----------
    header : astropy.io.fits.header.Header
    pathcal : str
        Absolute file path to calibration data directory that contains
        the default calibration files.
    result : collections.OrderedDict
        The configuration structure to update with grism values.
    """
    spectel = result['spectel']
    slit = result['slit']
    dateobs = result['dateobs']

    pathcal = os.path.join(pathcal, 'grism')
    calfile_default = os.path.join(pathcal, 'caldefault.txt')
    if not os.path.isfile(calfile_default):
        msg = "getcalpath - Problem reading default file %s" % calfile_default
        msg += "\nRoutine failed. Returning default dict."
        log.warning(msg)
        result['error'] = True
        return

    calcols = ['spectel', 'slit', 'maskfile', 'wavefile',
               'resolution', 'respfile', 'slitfile', 'linefile']
    df = pandas.read_csv(calfile_default, delim_whitespace=True,
                         comment='#', names=calcols, index_col=0)
    table = df[(df.index >= dateobs)
               & (df['spectel'] == spectel)
               & (df['slit'] == slit)]
    if len(table) == 0:
        msg = 'getcalpath - Problem reading defaults ' \
              'for %s, %s ' % (spectel, slit)
        msg += 'on date %s from %s' % (dateobs, calfile_default)
        log.warning(msg + '\nRoutine failed. Returning default dict')
        result['error'] = True
        return

    # take the first applicable date
    row = table[table.index == table.index.min()].sort_index().iloc[0]
    for f in calcols:
        if f.endswith('file') and f in row and row[f] != '.':
            result[f] = os.path.join(pathcal, row[f])
        elif f == 'resolution' and f in row and row[f] != '.':
            try:
                result[f] = float(row[f])
            except ValueError:
                msg = 'getcalpath - Problem reading resolution ' \
                      'for %s, %s ' % (spectel, slit)
                msg += 'on date %s from %s' % (dateobs, calfile_default)
                log.warning(msg)
                result['error'] = True
