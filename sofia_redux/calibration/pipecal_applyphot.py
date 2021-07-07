# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Calculate aperture photometry and update FITS header."""

import argparse
import warnings

from astropy import log
from astropy.io import fits

from sofia_redux.calibration.pipecal_config import pipecal_config
from sofia_redux.calibration.pipecal_error import PipeCalError
from sofia_redux.calibration.pipecal_photometry import pipecal_photometry
from sofia_redux.calibration.pipecal_calfac import pipecal_calfac
from sofia_redux.calibration.pipecal_util \
    import guess_source_position, add_phot_keys

__all__ = ['pipecal_applyphot']


def pipecal_applyphot(fitsfile, srcpos=None, fitsize=None, fwhm=None,
                      profile='moffat', aprad=None, skyrad=None,
                      runits=None, overwrite=True):

    """
    Calculate  photometry on a FITS image and store results to FITS header.

    FITS images are expected to have been produced by either the
    HAWC+, FORCAST, or FLITECAM pipelines.  For any other data format,
    call the `pipecal_util.run_photometry` function instead.

    The procedure followed here is:

        1. Read the header, data, and variance from the FITS file.
        2. Call pipecal_photometry on the data.
        3. Add photometry keywords to header.
        4. Write file back to disk.

    Defaults for all photometry parameters except `srcpos` are
    determined from the instrument configuration (`pipecal_config`)
    if possible.  If not, they are set by `pipecal_photometry`
    instead.

    Parameters
    ----------
    fitsfile : string
        Path to a FITS image file.
    srcpos : 2d-array, optional
        Initial guess at source position (x,y), zero-indexed.
        If not provided, will be read from FITS header keywords
        SRCPOSX, SRCPOSY if present.
    fitsize : float, optional
        Size of subimage to fit.
    fwhm : float, optional
        Initial guess at PSF fwhm.
    profile : string, optional
        Fit type (Moffat, Lorentzian, Gaussian).
    aprad : float, optional
        Aperture radius for aperture photometry.
    skyrad : array-like, optional
        Sky radii (inner, outer) for aperture photometry.
    runits : string, optional
        Raw data units, before flux calibration.
    overwrite : bool, optional
        If set, the input FITS file will be overwritten with the updated
        header.  If not, a FITS file of the same base name and a
        '_new.fits` suffix will be written instead, to the same location
        as the input file.
    """
    if overwrite:
        mode = 'update'
    else:
        mode = 'readonly'

    # Read the data
    try:
        hdul = fits.open(fitsfile, mode=mode)
        header = hdul[0].header
    except FileNotFoundError as err:
        log.error(f'Unable to open {fitsfile}.')
        raise PipeCalError(err)
    except (OSError, IndexError) as err:
        log.error(f'Bad FITS file: {fitsfile}.')
        raise PipeCalError(err)

    inst = header['INSTRUME'].strip().upper()

    # Special handling for each instrument
    if inst == 'HAWC_PLUS':
        # First extension is flux
        # Error is either in NOISE or ERROR I extension
        image = hdul[0].data
        try:
            variance = hdul['NOISE'].data ** 2
        except KeyError:
            variance = hdul['ERROR I'].data ** 2
    elif inst == 'FLITECAM':
        # new FLITECAM data has first extension flux,
        # second extension error
        try:
            image = hdul[0].data
            variance = hdul[1].data ** 2
        except IndexError:
            # old FLITECAM has a data cube;
            # first plane is flux, second is error
            data = hdul[0].data
            image = data[0]
            variance = data[1] ** 2
    elif inst == 'FORCAST':
        # new FORCAST data has first extension flux,
        # second extension error
        try:
            image = hdul[0].data
            variance = hdul[1].data ** 2
        except IndexError:
            # old FORCAST has a data cube;
            # first plane is flux, second is variance
            data = hdul[0].data
            image = data[0]
            variance = data[1]
    else:
        msg = 'Unsupported instrument: {}'.format(inst)
        log.error(msg)
        raise PipeCalError(msg)

    # Read in pipecal config from header
    config = pipecal_config(header)
    if config is None:
        log.warning('No config found.')
    else:
        log.debug('Full pipecal configuration:')
        for key, value in config.items():
            log.debug('  {}: {}'.format(key, value))

    # Check the srcpos input. If it isn't filled, fill it
    # with info from the header
    try:
        srcposx = header['SRCPOSX']
        srcposy = header['SRCPOSY']
    except KeyError:
        srcposx = None
        srcposy = None
    srcpos = guess_source_position(header, image, srcpos=srcpos)
    log.debug('Starting guess position: {}'.format(srcpos))

    # Set defaults from config file
    if config and not aprad and 'aprad' in config:
        aprad = config['aprad']
        log.info('Aperture radius: {}'.format(aprad))
    if config and not skyrad and 'bgin' in config and 'bgout' in config:
        skyrad = [config['bgin'], config['bgout']]
        log.info('Sky radii: {}'.format(skyrad))
    if config and not fwhm and 'fwhm' in config:
        fwhm = config['fwhm']
    if config and not fitsize and 'fitsize' in config:
        fitsize = config['fitsize']
    if config and not runits and 'runits' in config:
        runits = config['runits']

    # Perform photometry
    phot = pipecal_photometry(image, variance, srcpos=srcpos,
                              fitsize=fitsize, fwhm=fwhm,
                              profile=profile, aprad=aprad,
                              skyrad=skyrad, runits=runits)

    # Loop through phot and update the header
    add_phot_keys(header, phot, config=config)

    log.info('Source Position (x,y): '
             '{:.2f}, {:.2f}'.format(header['STCENTX'], header['STCENTY']))
    if srcposx is not None and srcposy is not None:
        log.info('Diff in position: '
                 '{:.2f}, {:.2f}'.format(srcposx - header['STCENTX'],
                                         srcposy - header['STCENTY']))
    log.info('Source Flux: '
             '{:.2f} +/- {:.2f}'.format(header['STAPFLX'],
                                        header['STAPFLXE']))

    # Calculate reference calibration factor from flux
    if not config or 'std_flux' not in config:
        log.warning('No model found. Not writing REFCALFC.')
        if overwrite:
            hdul.flush()
        else:
            newfile = fitsfile.replace('.fits', '_new.fits')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                hdul.writeto(newfile, overwrite=True)
        return

    log.info('Model Flux: '
             '{:.3f} +/- {:.3f}'.format(header['MODLFLX'],
                                        header['MODLFLXE']))

    # Calculate reference cal factor, assuming already
    # telluric-corrected data
    flux = header['STAPFLX']
    flux_err = header['STAPFLXE']
    try:
        ref_calfac, ref_ecalfac = pipecal_calfac(flux, flux_err, config)
    except ValueError:
        log.warning('Negative flux; not adding REFCALFC.')
        return

    # Add to header
    header['REFCALFC'] = (ref_calfac, 'Reference calibration factor')
    header['REFCALER'] = (ref_ecalfac, 'Reference calibration factor error')

    log.info('Reference Cal Factor: '
             '{:.3f} +/- {:.3f}'.format(header['REFCALFC'],
                                        header['REFCALER']))

    # Write the new fits header to file
    if overwrite:
        hdul.flush()
    else:
        newfile = fitsfile.replace('.fits', '_new.fits')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            hdul.writeto(newfile, overwrite=True)


def main():
    """Run photometry from the command line."""
    parser = argparse.ArgumentParser(
        description='Compute photometry.')
    parser.add_argument('filename', metavar='filename', nargs='+',
                        help='Path to one or more input files to '
                             'modify in place.')
    parser.add_argument('-n', '--new', dest='overwrite',
                        action='store_false', default=True,
                        help='Set to write to _new.fits file instead '
                             'of overwriting the input.')
    parser.add_argument('-l', '--loglevel', dest='loglevel', type=str,
                        action='store', default='INFO',
                        help='Log level.')
    parser.add_argument('-z', '--fitsize', dest='fitsize', type=int,
                        action='store', default=None,
                        help='Fit subimage size (pix).')
    parser.add_argument('-s', '--srcpos', dest='srcpos', type=str,
                        action='store', default=None,
                        help='Estimated source position (x,y).')
    parser.add_argument('-f', '--fwhm', dest='fwhm', type=float,
                        action='store', default=None,
                        help='Estimated FWHM (pix).')
    parser.add_argument('-p', '--profile', dest='profile', type=str,
                        action='store', default='moffat',
                        help='Profile function (moffat, gaussian, '
                             'or lorentzian).')
    parser.add_argument('-r', '--aprad', dest='aprad', type=float,
                        action='store', default=None,
                        help='Aperture radius (pix).')
    parser.add_argument('-b', '--skyrad', dest='skyrad', type=str,
                        action='store', default=None,
                        help='Sky radii in pix (inner,outer).')
    parser.add_argument('-u', '--raw_units', dest='runits', type=str,
                        action='store', default=None,
                        help='Raw data units before calibration, '
                             'to use in header comments.')
    args = parser.parse_args()

    if args.srcpos is not None:
        try:
            srcpos = [float(x) for x in args.srcpos.split(',')]
            if len(srcpos) != 2:
                raise ValueError
        except ValueError:
            srcpos = None
            parser.error("Invalid srcpos argument.")
    else:
        srcpos = None
    if args.skyrad is not None:
        try:
            skyrad = [float(x) for x in args.skyrad.split(',')]
            if len(skyrad) != 2:
                raise ValueError
        except ValueError:
            skyrad = None
            parser.error("Invalid skyrad argument.")
    else:
        skyrad = None

    log.setLevel(args.loglevel.upper())
    for fname in args.filename:
        log.info('Running: {}'.format(fname))
        pipecal_applyphot(fname, srcpos=srcpos,
                          fitsize=args.fitsize, fwhm=args.fwhm,
                          profile=args.profile, aprad=args.aprad,
                          skyrad=skyrad, runits=args.runits,
                          overwrite=args.overwrite)
        log.info('')
