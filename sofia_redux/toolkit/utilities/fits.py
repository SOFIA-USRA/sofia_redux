# Licensed under a 3-clause BSD style license - see LICENSE.rst

from contextlib import contextmanager
import os

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.toolkit.utilities.func import goodfile

__all__ = ['hdinsert', 'add_history', 'add_history_wrap', 'robust_read',
           'getheader', 'getdata', 'header_to_chararray',
           'chararray_to_header', 'gethdul', 'write_hdul',
           'get_key_value', 'set_log_level']

kref = 'AAAAAAAA'  # Keywords reference key marker (via redux)
href = 'BBBBBBBB'  # HISTORY reference key marker (via redux)


def hdinsert(header, key, value, comment=None, refkey='HISTORY', after=False):
    """
    Insert or replace a keyword and value in the header

    Note that the insert method of astropy.io.fits.header.Header appears
    to be broken in cases where a value is to be inserted after a
    certain keyword and before HISTORY.  In these cases header.insert()
    inserts the new keyword after the first history card... which is
    bad.  Therefore, we do the insertion using a more manual indexing.

    In cases where both `refkey` and `after` are supplied, the `after`
    parameter will be evaluated first.

    Parameters
    ----------
    header : astropy.io.fits.header.Header
    key : str
        keyword name to insert or replace
    value
        The value for the keyword
    comment : str, optional
        If provided a new comment will be used.  Otherwise, the previous
        comment will be used when a keyword value is being replaced.
    refkey : str or int or None or 2-tuple, optional
        The keyword before which the keyword should be inserted in cases
        where the new keyword does not yet exist in the header.  By default
        new keywords will be placed before the first HISTORY card.  If None,
        use the end of the header.
    after : bool
        If True, insert after `refkey` rather than before.

    Returns
    -------
    None
        The header will be modified in place
    """
    if not isinstance(header, fits.header.Header):
        log.error("Invalid header")
        return
    if key in header and comment is None:
        comment = header.comments[key]
    if refkey not in header or key in header:
        header[key] = value, comment
    else:
        header.insert(refkey, (key, value, comment), after=after)


def add_history(header, msg, prefix=None, refkey=href):
    """
    Add HISTORY message to a FITS header before the pipeline.

    Parameters
    ----------
    header : astropy.io.fits.header.Header
        FITS header used to write the HISTORY message
    msg : str
        the HISTORY message
    prefix : str, optional
        Prefix message with this string.
    refkey : str, optional
        A reference key to insert the message after.
    """
    if header is not None:
        s = '' if prefix is None else str(prefix) + ': '
        if refkey not in header:
            refkey = 'HISTORY'
        hdinsert(header, 'HISTORY', s + msg, refkey=refkey)


def add_history_wrap(prefix):
    """
    Make a function to add HISTORY messages to a header,
    prefixed with a string.

    Parameters
    ----------
    prefix : str
        The message to prefix.

    Returns
    -------
    function
    """
    def wrapper(header, message):
        add_history(header, message, prefix=prefix)
    return wrapper


def robust_read(filename, data_hdu=0, header_hdu=0, extension=None,
                verbose=True):
    """
    Retrieve the data and header from a FITS file

    Does as many checks as possible to fix a potentially broken
    FITS file.  Feel free to add other stuff.

    Parameters
    ----------
    filename : str
        path to a FITS file
    header_hdu : int, optional
        Header Data Unit to retrieve header. Default is 0 (Primary)
    data_hdu : int, optional
        Header Data Unit to retrieve data. Default is 0 (Primary)
    extension : int, optional
        If supplied, overrides both `header_hdu` and `data_hdu`
    verbose : bool, optional
        If True, output log messages on error

    Returns
    -------
    numpy.ndarray, astropy.io.fits.header.Header
       The data array and header of the FITS file as a 2-tuple
    """

    hdul, dataout, headout = None, None, None
    if not goodfile(filename, verbose=verbose):
        return dataout, headout

    if isinstance(extension, int):
        data_hdu = extension
        header_hdu = extension

    try:
        hdul = fits.open(
            filename, mode='readonly', ignore_missing_end=True)
        hdul.verify('silentfix')
        for hidx in [data_hdu, header_hdu]:
            if not isinstance(hidx, int) or (hidx < 0) or (hidx >= len(hdul)):
                log.warning("HDU %s does not exist (data)" % data_hdu)
                return dataout, headout
        dataout = hdul[data_hdu].data
        headout = hdul[header_hdu].header
    except ValueError as err:
        if 'ASCII' in str(err):
            header = hdul[header_hdu].header
            try:
                for key in header.keys():
                    try:
                        _ = header[key]
                    except fits.verify.VerifyError:
                        header.remove(key)
                        header = header.copy()
                        header[key] = 'UNKNOWN'
                headout = header
            except (Exception, ValueError):
                pass
    except(Exception, AttributeError):
        pass
    finally:
        try:
            hdul.close()
        except (AttributeError, Exception):
            pass
    return dataout, headout


def getheader(filename, hdu=0, verbose=True):
    """
    Returns the header of a FITS file

    Uses robust_read to extract the header.

    Parameters
    ----------
    filename : str
        path to a FITS file
    hdu : int, optional
        Header Data Unit.  Primary HDU is 0 (default)
    verbose : bool, optional
        If True, output log messages on error

    Returns
    -------
    astropy.io.fits.header.Header
    """
    if not goodfile(filename, verbose=verbose):
        return
    _, header = robust_read(filename, header_hdu=hdu)
    if not isinstance(header, fits.header.Header):
        log.error("Could not read FITS header: %s" % filename)
        return
    return header


def getdata(filename, hdu=0, verbose=True):
    """
    Returns the data from a FITS file

    Uses robust_read to extract the data.

    Parameters
    ----------
    filename : str
        path to a FITS file
    hdu : int, optional
        Header Data Unit.  Primary HDU is 0 (default)
    verbose : bool, optional
        If True, output log messages on error

    Returns
    -------
    numpy.ndarray
    """
    if not goodfile(filename, verbose=verbose):
        return
    data, _ = robust_read(filename, data_hdu=hdu)
    if not isinstance(data, (fits.fitsrec.FITS_rec, np.ndarray)):
        log.error("Could not read FITS data: %s" % filename)
        return
    return data


def header_to_chararray(header):
    """
    Convert a FITS header to an array of strings

    For the weirdness of FIFI-LS

    Parameters
    ----------
    header : astropy.io.fits.header.Header

    Returns
    -------
    np.ndarray
    """
    if not isinstance(header, fits.header.Header):
        log.error("Invalid header")
        return
    c = repr(header).split('\n')
    c = [x.ljust(80)[:80] for x in c]
    return np.char.array([c], itemsize=80, unicode=True)


def chararray_to_header(chararray):
    """
    Convert an array of strings to a FITS header

    For the weirdness of FIFI-LS

    Parameters
    ----------
    chararray : np.ndarray

    Returns
    -------
    astropy.io.fits.header.Header
    """
    if not isinstance(chararray, np.ndarray):
        log.error("Invalid chararray")
        return

    if chararray.ndim not in [1, 2]:
        log.error("Invalid chararray features")
        return

    try:
        c = chararray[0] if chararray.ndim == 2 else chararray
        c = ''.join([x.ljust(80)[:80] for x in c])
        h = fits.header.Header.fromstring(c)
        return h
    except (ValueError, TypeError, AttributeError) as err:
        log.error(str(err))
        return


def gethdul(filename, verbose=True):
    """
    Returns the HDUList from a FITS file

    Performs a few additional sanity checks

    Parameters
    ----------
    filename : str or HDUList or array of HDU
        Path to a FITS file
    verbose : bool, optional
        If True, output log messages on error.

    Returns
    -------
    astropy.io.fits.HDUList
    """
    if isinstance(filename, fits.HDUList):
        return filename
    if (isinstance(filename, list)
            and len(filename) > 0
            and isinstance(filename[0], fits.PrimaryHDU)):
        return fits.HDUList(filename)
    if not goodfile(filename, verbose=verbose):
        return
    try:
        hdul = fits.open(filename, ignore_missing_end=True)
        hdul.verify('silentfix')
    except (AttributeError, Exception, fits.VerifyError) as err:
        if verbose:
            log.error("Unable to read FITS file %s: %s" %
                      (str(err), filename))
        return
    return hdul


def write_hdul(hdul, outdir=None, overwrite=True):
    """
    Write a HDULists to disk.

    Output filename is extracted from the primary header (extension 0) of
    each HDUList from the FILENAME keyword.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
    outdir : str, optional
        Name of the output path.  Default is the current working directory.
    overwrite : bool, optional
        If False, will fail if the output filename already exists.

    Returns
    -------
    str
        Filename written to disk
    """
    outfile = hdul[0].header.get('FILENAME')
    if outfile is None:
        log.error("Missing FILENAME in HDUList primary header")
        return
    if outdir is not None:
        if isinstance(outdir, str):
            outfile = os.path.join(outdir, outfile)
        else:
            raise ValueError('Invalid output directory type.')

    if os.path.isfile(outfile):
        if os.path.exists(outfile):
            if not overwrite:
                log.error("%s already exists - will not overwrite" % outfile)
                return
            try:
                os.remove(outfile)
            except (IOError, OSError):  # pragma: no cover
                log.error("Unable to remove file %s" % outfile)
                return
    try:
        hdul.writeto(outfile, overwrite=True,
                     output_verify='silentfix')
    except (fits.VerifyError, OSError, Exception) as err:
        log.error("Unable to write to %s: %s" % (outfile, str(err)))
        return
    finally:
        hdul.close()
    return outfile


def get_key_value(header, key, default='UNKNOWN'):
    """
    Get a key value from a header.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        FITS header.
    key : str
        Key to retrieve.
    default :
        Value to return if not retrievable from the header.

    Returns
    -------
    str, int, float, or bool
        FITS keyword value; default if not found.
    """
    try:
        value = header[key]
        if isinstance(value, str):
            value = str(value).strip().upper()
    except (KeyError, TypeError):
        value = default
    return value


@contextmanager
def set_log_level(level):
    """
    Context manager to temporarily set the log level.

    Parameters
    ----------
    level : str or int
        Logging level as defined in the `logging` module.
    """
    orig_level = log.level
    log.setLevel(level)
    try:
        yield
    finally:
        log.setLevel(orig_level)
