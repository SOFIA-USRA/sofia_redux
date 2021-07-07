# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np
from sofia_redux.toolkit.utilities.fits import gethdul
from sofia_redux.toolkit.utilities.func import setnumber, valid_num

__all__ = ['readspec']


def readspec(filename):
    """
    Reads a SpeX spectral FITS image

    Parameters
    ----------
    filename : str
        File path to FITS file

    Returns
    -------
    dict
    """
    hdul = gethdul(filename, verbose=True)
    if hdul is None:
        return
    data = hdul[0].data
    header = hdul[0].header

    if data is None:
        log.error('Unexpected data format: first HDU has no data.')
        return

    result = {'orders': np.array(str(header.get('ORDERS')).split(','))}
    if all([x.isdigit() for x in result['orders']]):
        result['orders'] = result['orders'].astype(int)
    result['norders'] = setnumber(header.get('NORDERS'), default=1, minval=1)
    result['naps'] = setnumber(header.get('NAPS'), default=1, minval=1)
    result['instr'] = header.get('INSTR')
    result['obsmode'] = ''.join(str(header.get('MODENAME')).split())
    result['rp'] = setnumber(header.get('RP'), default=2000, dtype=float)
    result['start'] = setnumber(header.get('START'), default=0)
    result['stop'] = setnumber(header.get('STOP'), default=0)
    result['slith_pix'] = header.get('SLTH_PIX')
    result['slith_arc'] = header.get('SLTH_ARC')
    result['slitw_pix'] = header.get('SLTW_PIX')
    result['slitw_arc'] = header.get('SLTW_ARC')
    result['airmass'] = header.get('AIRMASS')
    result['xunits'] = str(header.get('XUNITS')).strip()
    result['yunits'] = str(header.get('YUNITS')).strip()
    result['runits'] = str(header.get('RAWUNITS')).strip()
    result['xtitle'] = str(header.get('XTITLE', '!7k!5 (%s)' %
                                      result['xunits'])).strip()
    result['ytitle'] = str(header.get('YTITLE', '!5f (%s)' %
                                      result['yunits'])).strip()

    bgr = str(header.get('BGR')).split(';')
    obg = []
    for bg in bgr:
        ob = []
        for b in bg.split(','):
            ob.append([float(x) if valid_num(x) else np.nan
                       for x in b.split('-')])
        obg.append(ob)
    result['bgr'] = np.array(obg)

    appos = np.full((result['norders'], result['naps']), np.nan)
    aprad = np.full((result['norders'], result['naps']), np.nan)
    psfrad = np.full((result['norders'], result['naps']), np.nan)

    shape = list(data.shape)
    if len(shape) == 3:
        shape = shape[1:]
    spectra = np.full(
        (result['norders'], result['naps'], *shape), np.nan)
    for j in range(result['norders']):
        onum = str(result['orders'][j]).zfill(2)
        p = ''.join(str(header.get('APPOSO' + onum)).split()).split(',')
        r = ''.join(str(header.get('APRADO' + onum)).split()).split(',')
        f = ''.join(str(header.get('PSFRAD' + onum)).split()).split(',')

        msg = "Mismatch: data size, number of apertures, " \
              "and number of orders"
        try:
            appos[j] = np.array([float(x) if valid_num(x)
                                 else np.nan for x in p])
            aprad[j] = np.array([float(x) if valid_num(x)
                                 else np.nan for x in r])
            psfrad[j] = np.array([float(x) if valid_num(x)
                                  else np.nan for x in f])
        except (KeyError, ValueError):
            log.error(msg)
            raise ValueError(msg) from None
        for k in range(result['naps']):
            ns = (j * result['naps']) + k
            if data.ndim == 3 and ns < data.shape[0]:
                spectra[j, k] = data[ns]
            elif j != 0 or k != 0:
                log.error(msg)
                raise ValueError(msg)
            elif data.ndim == 2:
                spectra[j, k] = data
            else:
                msg = "Invalid data shape."
                log.error(msg)
                raise ValueError(msg)

    result['appos'] = appos
    result['aprad'] = aprad
    result['psfrad'] = psfrad
    result['spectra'] = spectra
    result['header'] = header
    return result
