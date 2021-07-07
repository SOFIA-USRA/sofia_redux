# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log
from astropy.io import fits
import numpy as np
import pandas

from sofia_redux.toolkit.utilities.fits import hdinsert, kref

import sofia_redux.instruments.forcast as drip
import sofia_redux.instruments.forcast.configuration as dripconfig
from sofia_redux.instruments.forcast.getpar import getpar

__all__ = ['pinhole_defaults', 'read_pinhole_file',
           'pinhole_model', 'view_model', 'distcorr_model']


def pinhole_defaults():
    # get pinhole filename with getpar
    # (will load config if not already done)
    filename = getpar(fits.header.Header(),
                      'fpinhole', default='undefined')

    # may have subdirectories specified with /
    # replace them OS independently.
    filename = os.path.join(*filename.split('/'))
    datadir = os.path.join(os.path.dirname(drip.__file__), 'data', 'pinhole')
    fpinhole = os.path.join(datadir, filename)

    # retrieve the rest of the values directly
    # from the config
    config = dripconfig.configuration
    order = int(config.get('order', 4))

    return {
        'fpinhole': fpinhole, 'order': order
    }


def read_pinhole_file(pinhole_file):
    """
    Read the pinhole file and return a dataframe

    Note that 1 is subtracted from the xpos and ypos pixel
    positions for consistency with Python/IDL indexing.

    Parameters
    ----------
    pinhole_file : str, optional
        Path to the pinhole file.  If None is supplied, will read
        the default pinhole file

    Returns
    -------
    pandas.DataFrame
        contents of the pinhole file as a dataframe
    """
    if not isinstance(pinhole_file, str) or not os.path.isfile(pinhole_file):
        log.error("File %s does not exist" % repr(pinhole_file))
        return
    log.info("Reading pinhole file %s" % pinhole_file)
    table = pandas.read_csv(pinhole_file, comment='#',
                            delim_whitespace=True)
    return table


def pinhole_model(xpos, ypos, xid, yid):
    """
    Generate the pinhole model from paramters.

    Parameters
    ----------
    xpos : array-like
    ypos : array-like
    xid : array-like
    yid : array-like

    Returns
    -------
    dict
        The model parameters. Keys are 'avgdx', 'avgdy',
        'angle', 'xmodel', 'ymodel'.
    """
    # the FORCAST array is 256 x 256
    nximg = 256
    nyimg = 256

    # number of pinhole points in table
    nxpts = np.max(xid) + 1
    nypts = np.max(yid) + 1

    # Check that positions make sense (greater than 0 and lower than
    # image size)
    if (xpos[~np.isnan(xpos)] < 0).any() or \
            (xpos[~np.isnan(xpos)] >= nximg).any():
        log.error("X positions are lower than 0 or greater than %s" % nximg)
        return
    elif (ypos[~np.isnan(ypos)] < 0).any() or \
            (ypos[~np.isnan(ypos)] >= nyimg).any():
        log.error("Y positions are lower than 0 or greater than %s" % nyimg)
        return

    results = {}

    # Rearrange pinholes into arrays
    xpos_arr = np.full((nypts, nxpts), np.nan)
    ypos_arr = np.full((nypts, nxpts), np.nan)
    xpos_arr[yid, xid] = xpos
    ypos_arr[yid, xid] = ypos

    # Diff adjacent pins to get dx and dy, for pin spacing
    # NaNs are propagated
    dxpos = xpos_arr[:, 1:] - xpos_arr[:, :-1]
    dypos = ypos_arr[1:, :] - ypos_arr[:-1, :]

    # Also get the change in x position up a column, for angle
    # estimation
    dxlpos = xpos_arr[1:, :] - xpos_arr[:-1, :]
    theta = np.arctan2(dxlpos[:, :-1], dxpos[:-1, :])

    results['angle'] = float(np.nanmean(theta))

    # If angle is small, it's better to just set it
    # to zero
    log.debug('Estimated angle: {}'.format(np.rad2deg(results['angle'])))
    if np.abs(np.rad2deg(results['angle'])) < 0.9:
        log.debug('Estimated angle is close to zero: '
                  'ignoring angle correction.')
        results['angle'] = 0.0

    # Average x and y separation of pin holes,
    # corrected by the angle
    results['avgdx'] = float(np.nanmean(dxpos)) / abs(np.cos(results['angle']))
    results['avgdy'] = float(np.nanmean(dypos)) / abs(np.cos(results['angle']))

    # Create model x,y coordinates
    mody, modx = np.mgrid[:nypts, :nxpts].astype('float')
    modx = modx * results['avgdx'] + xpos_arr[0, 0]
    mody = mody * results['avgdy'] + ypos_arr[0, 0]

    # Rotate model about the center of the array
    xcen = nximg / 2.0 + 0.5
    ycen = nyimg / 2.0 + 0.5
    ang = results['angle']
    xp = ((modx - xcen) * np.cos(ang)) + (mody - ycen) * np.sin(ang) + xcen
    yp = ((modx - xcen) * -np.sin(ang)) + (mody - ycen) * np.cos(ang) + ycen

    # Offset to match a central pinhole
    distsq = (xpos_arr - xcen)**2 + (ypos_arr - ycen)**2
    minpin = np.unravel_index(np.nanargmin(distsq), distsq.shape)
    xp = xp - xp[minpin] + xpos_arr[minpin]
    yp = yp - yp[minpin] + ypos_arr[minpin]

    # Flatten arrays and remove NaNs
    idx = np.logical_and(~np.isnan(xpos_arr), ~np.isnan(ypos_arr))
    results['xpos'] = xpos_arr[idx].flatten()
    results['ypos'] = ypos_arr[idx].flatten()
    results['xmodel'] = xp[idx].flatten()
    results['ymodel'] = yp[idx].flatten()

    return results


def view_model(x, y, fwhm=1.5, amplitude=3000,
               write_file=None, force=False, show=False):
    """
    View the pinhole model and optionally write to FITS file.

    Image will effectively be convolved with a gaussian for
    visibility

    Parameters
    ----------
    x : array-like
        x-pixel pinhole locations
    y : array-like
        y-pixel pinhole locations
    fwhm : float, optional
        fwhm of gaussian convolution kernel (pixels)
    amplitude : float, optional
        Amplitude of gaussian convolution kernel
    force : bool, optional
        Force overwrite of `write_file` if it already exists
    write_file : str, optional
        Write the image in FITS format to this location
    show : bool, optional
        Show plot with matplotlib.

    Returns
    -------
    None
    """
    from astropy.modeling.models import Gaussian2D
    from astropy.stats import gaussian_sigma_to_fwhm

    nx, ny = 256, 256
    yy, xx = np.mgrid[:ny, :nx]
    fwhm = gaussian_sigma_to_fwhm * fwhm
    gmodel = Gaussian2D(amplitude=amplitude, x_stddev=fwhm, y_stddev=fwhm)
    image = np.zeros((ny, nx), dtype=float)
    for gy, gx in zip(y, x):
        gmodel.x_mean = gx
        gmodel.y_mean = gy
        image += gmodel(xx, yy)

    if show:  # pragma: no cover
        import matplotlib.pyplot as plt
        plt.ion()
        plt.figure()
        plt.imshow(image, origin='lower')
        plt.title('Pinhole Model')
        plt.show()
        plt.pause(0.001)

    if isinstance(write_file, str):
        if os.path.isfile(write_file):
            if not force:
                log.warning("will not overwrite: %s" % write_file)
                return
            else:
                os.remove(write_file)
        fits.writeto(write_file, image)


def distcorr_model(pinhole=None,
                   viewpin=False, basehead=None, order=None):
    """
    Generate model array of pin holes base on input file

    Read the positions of the pinholes and average the distance
    between positions; creates a model where the pinholes are
    regularly spaced.  The model is then shifted and rotated in
    order to match the observed pinhole mask.

    Parameters
    ----------
    pinhole : str or pandas.dataframe
        File path to a text file containing a list of the positions
        of the pinholes or dataframe object.  Columns are
        xid, yid, xpos, ypos.
    viewpin : bool or str, optional
        If set to True, will display the pin model.
        If a string is provided, it will be interpreted
        as a filename, and the model will be written to the
        file instead of shown.
    basehead : astropy.io.fits.header.Header, optional
        Header array to update with pin model array
        (PIN_MOD=[dx,dy,angle,order]).
    order : int, optional
        Order of distortion model.  If not provided, the default
        value of 4 will be used.

    Notes
    -----
    A value of None for pinhole or order will cause default
    values specified in the drip configuration file to be used.

    Returns
    -------
    dict
        model -> numpy.ndarray
            N(x, y) model reference positions of shape (N, 2)
        pins -> numpy.ndarray
            N(x, y) pin positions of shape (N, 2)
        nx, ny -> int
            define number of pixels for both pins and image in x and y
        dx, dy -> float
            model x, y spacing
        angle -> float
            clockwise rotation of the model about the center
            in degrees.
        order -> int
            order to be used if using the "polynomial" method in
            sofia_redux.instruments.forcast.undistort based upon this model.
    """
    defaults = pinhole_defaults()
    if pinhole is None:
        pinhole = defaults.get('fpinhole')
    if order is None:
        order = defaults.get('order')

    if isinstance(pinhole, pandas.DataFrame):
        table = pinhole.copy()
    else:
        table = read_pinhole_file(pinhole)
    if not isinstance(table, pandas.DataFrame):
        log.error("Invalid pinhole data")
        return

    xpos = table.xpos.values.copy()
    ypos = table.ypos.values.copy()
    xid = table.xid.values.copy()
    yid = table.yid.values.copy()
    positions = pinhole_model(xpos, ypos, xid, yid)

    log.info("avgdx=%f, avgdy=%f, angle=%f" %
             (positions['avgdx'],
              positions['avgdy'],
              np.rad2deg(positions['angle'])))

    if viewpin:
        if isinstance(viewpin, str):
            fout = viewpin
            show = False
        else:  # pragma: no cover
            fout = None
            show = True
        view_model(positions['xmodel'], positions['ymodel'],
                   write_file=fout, show=show)

    # Update header
    if isinstance(basehead, fits.header.Header):
        value = '[%f,%f,%f,%s]' % (
            positions['avgdx'], positions['avgdy'],
            np.rad2deg(positions['angle']), int(order))
        comment = "pinhole model coeffs"
        hdinsert(basehead, 'PIN_MOD', value, comment=comment, refkey=kref)

    return {
        'model': np.stack((positions['xmodel'], positions['ymodel']), axis=1),
        'pins': np.stack((positions['xpos'], positions['ypos']), axis=1),
        'dx': positions['avgdx'],
        'dy': positions['avgdy'],
        'nx': 256,
        'ny': 256,
        'order': int(order),
        'angle': np.rad2deg(positions['angle'])
    }
