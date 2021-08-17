# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings

from astropy import log
from astropy.convolution import convolve, Box2DKernel
from astropy.visualization import AsymmetricPercentileInterval
from astropy.wcs import WCS

from matplotlib.backends.backend_agg \
    import FigureCanvasAgg as FigureCanvas
from matplotlib.cm import get_cmap
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.anchored_artists \
    import AnchoredEllipse
import numpy as np

from sofia_redux.toolkit.utilities.fits import gethdul

__all__ = ['make_image', 'make_spectral_plot']


def make_image(filename, extension=0, colormap='viridis', scale=None,
               n_contour=0, contour_color='gray', fill_contours=False,
               title='', subtitle='', subsubtitle='',
               crop_region=None, crop_unit='wcs', grid=False,
               beam=False, plot_layout=None,
               figure_size=None, cube_slice=None, decimal=False,
               watermark=None):
    """
    Generate a map image from a FITS file.

    This function generates a matplotlib Figure from imaging data in the
    input, for quick-look purposes.  The Figure can then be displayed, added
    to, or saved to disk.

    Parameters
    ----------
    filename : str or fits.HDUList
        FITS file to display.  May be either a file path or an astropy
        HDUList.
    extension : str or int, optional
        Image extension name or number to display.  Default is first
        extension.
    colormap : str, optional
        Matplotlib color map name.
    scale : list of float, optional
        Low and high percentile values to use to set the image scale.
        Default is [0.25, 99.75].
    n_contour : int, optional
        Number of contours to overlay.  Set to zero to turn off contour
        overlay.
    contour_color : str, optional
        Matplotlib color name for contours.
    fill_contours : bool, optional
        Set to fill contours.
    title : str, optional
        Title for the plot.
    subtitle : str, optional
        Subtitle for the plot.
    subsubtitle : str, optional
        Sub-subtitle for the plot.
    crop_region : list of float, optional
        4-element list of (center_x, center_y, x_width, y_height),
        specified in `crop_unit` units (WCS units by default).
        If not specified, the full image will be displayed.
    crop_unit : {'wcs', 'pixel'}, optional
        If 'wcs', `crop_region` should be specified in WCS units.  If
        'pixel', `crop_region` should be specified in image units.
    grid : bool, optional
        If set, a grid will be overlaid on the image.
    beam : bool, optional
        If set, and the BMAJ, BMIN, and BPA keywords are set in the
        FITS header, a beam marker is displayed on the image.
    plot_layout : tuple, optional
        If specified, should be (nrow, ncol).  The image is placed
        in the first subplot in the figure; any others are left empty,
        for later additions to the figure.
    figure_size : tuple, optional
        If specified, should be (width, height) in inches. Default is
        (8, 8).
    cube_slice : int, optional
        If 3D data is supplied (e.g. a spectral cube), this parameter
        must be provided to specify the slice of the cube to display.
        Provided value should be a value in the first numpy index in
        a (nw, ny, nx) cube.  This should be the last index in the
        associated WCS (NAXIS3).
    decimal : bool, optional
        If set, celestial coordinates will be displayed in decimal
        degrees instead of sexagesimal.
    watermark : str, optional
        If provided, the string will be added as semi-transparent text
        in the lower-right corner of the image.

    Returns
    -------
    matplotlib.figure.Figure
        A figure containing the image map and any additional overlays
        specified.  The first axis contains the primary image plot.
    """
    hdul = gethdul(filename)
    if hdul is None:
        raise ValueError('No input file')

    # get figure
    if figure_size is None:
        figure_size = (8, 8)
    fig = Figure(figsize=figure_size)
    FigureCanvas(fig)

    # set gridspec if desired
    if plot_layout:
        nrow, ncol = plot_layout
    else:
        nrow, ncol = 1, 1
    fig.add_gridspec(nrow, ncol)

    # get background image
    try:
        map_hdu = hdul[extension]
    except (IndexError, ValueError):
        raise ValueError(f'No extension {extension} present') from None

    # check for cube slice
    if cube_slice is not None:
        slices = ['x', 'y', cube_slice]
        data = map_hdu.data[cube_slice, :, :]
    else:
        slices = None
        data = map_hdu.data

    # set image in WCS projection
    try:
        hwcs = WCS(map_hdu)
    except (ValueError, IndexError, KeyError, MemoryError):
        log.warning('Unreadable WCS; using pixel coordinates')
        hwcs = WCS()
    wcs_dim = hwcs.wcs.naxis
    if wcs_dim > 2 and slices is None:
        raise ValueError('Slice must be set for cube data')

    ax = fig.add_subplot(nrow, ncol, 1, projection=hwcs, slices=slices)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if hwcs.wcs.has_cd():
            # approximate pixel scale for CD-based WCS
            pixscal = np.sqrt(np.linalg.det(hwcs.wcs.cd))
            pix_x, pix_y = pixscal, pixscal
            mat = hwcs.wcs.cd
        else:
            pix_x = np.abs(hwcs.wcs.cdelt[0])
            pix_y = np.abs(hwcs.wcs.cdelt[1])
            mat = hwcs.pixel_scale_matrix

    # check for rotated data: off diagonal elements close to zero
    if not np.allclose(np.extract(1 - np.eye(*mat.shape), mat), 0):
        # in this case, always turn the grid on - the
        # coordinates are difficult to read otherwise
        grid = True

    if not scale:
        scale = [0.25, 99.75]
    interval = AsymmetricPercentileInterval(scale[0], scale[1])
    vmin, vmax = interval.get_limits(data)
    log.debug(f'Scale: {vmin} -> {vmax}')

    # Set up the figure
    img = ax.imshow(data, origin='lower',
                    cmap=colormap, vmin=vmin,
                    vmax=vmax)

    if crop_region:
        log.debug(f'Using center cropping: {crop_region}')
        if crop_unit == 'wcs':
            racent, deccent, awidth, aheight = crop_region

            if wcs_dim == 2:
                xcent, ycent = hwcs.wcs_world2pix(racent, deccent, 0)
            else:
                # assume 3 dim, first two are in plot
                xcent, ycent, _ = hwcs.wcs_world2pix(racent, deccent,
                                                     hwcs.wcs.crval[2], 0)
            xwid = awidth / 2 / pix_x
            ywid = aheight / 2 / pix_y
        else:
            xcent, ycent, xwid, ywid = crop_region

        log.debug(f'Crop region: x={xcent - xwid},{xcent + xwid}, '
                  f'y={ycent - ywid},{ycent + ywid}')
        ax.set_xlim(xcent - xwid, xcent + xwid)
        ax.set_ylim(ycent - ywid, ycent + ywid)

    ax.set_autoscale_on(False)
    if grid:
        ax.grid(alpha=0.25, color='w')
    if title:
        fig.suptitle(title)
    if subtitle and subsubtitle:
        ax.set_title('\n'.join([subtitle, subsubtitle]), size='small')
    elif subtitle:
        ax.set_title(subtitle)
    elif subsubtitle:
        ax.set_title(subsubtitle)

    try:
        equinox = int(hwcs.wcs.equinox)
    except ValueError:
        # non-celestial WCS
        xname = hwcs.wcs.cname[0]
        yname = hwcs.wcs.cname[1]
        xname = 'x' if not xname else xname
        yname = 'y' if not yname else yname
        xunit = hwcs.wcs.cunit[0]
        yunit = hwcs.wcs.cunit[1]
        if not xunit or str(xunit).strip() == '':
            ax.set_xlabel(xname)
        else:
            ax.set_xlabel(f'{xname} ({xunit})')
        if not yunit or str(yunit).strip() == '':
            ax.set_ylabel(yname)
        else:
            ax.set_ylabel(f'{yname} ({yunit})')
    else:
        ax.set_xlabel(f'RA (J{equinox})')
        ax.set_ylabel(f'Dec (J{equinox})')
        if decimal:
            ax.coords[0].set_format_unit('degree', decimal=True)
            ax.coords[1].set_format_unit('degree', decimal=True)

    if n_contour > 0:
        # get contour data
        # similar to aplpy.convolve_util.convolve,
        # aplpy.FITSFigure.show_contours
        smooth = 1
        kernel = Box2DKernel(smooth, x_size=smooth * 5, y_size=smooth * 5)
        cdata = convolve(data, kernel, boundary='extend')

        # levels
        levels = np.linspace(vmin, vmax, n_contour)
        log.debug(f'Contours: {levels}')

        if fill_contours:
            try:
                # filled contours
                ax.contourf(cdata, levels, transform=ax.get_transform(hwcs),
                            cmap=colormap)
                # contour lines
                ax.contour(cdata, levels, transform=ax.get_transform(hwcs),
                           colors=contour_color, linewidths=0.3)
            except ValueError:  # pragma: no cover
                # may be raised for pathological data (e.g. all zeros)
                pass
        else:
            try:
                ax.contour(cdata, levels, transform=ax.get_transform(hwcs),
                           colors=contour_color, linewidths=0.5)
            except ValueError:  # pragma: no cover
                pass

    # Beam marker (needs BMAJ/BMIN/BPA keywords)
    # from aplpy.FITSFigure.show_beam
    if beam:
        major = map_hdu.header.get('BMAJ', None)
        minor = map_hdu.header.get('BMIN', None)
        angle = map_hdu.header.get('BPA', 0.0)
        if None in [major, minor]:
            # try the primary if not in the mapping extension
            major = hdul[0].header.get('BMAJ', None)
            minor = hdul[0].header.get('BMIN', None)
            angle = hdul[0].header.get('BPA', 0.0)
        if major and minor:
            pixscale = np.mean([pix_x, pix_y])
            major /= pixscale
            minor /= pixscale
            log.debug(f'Beam major, minor, angle: {major} {minor} {angle}')
            beam = AnchoredEllipse(ax.transData, width=minor,
                                   height=major, angle=angle,
                                   loc=3, pad=1, borderpad=0.4,
                                   frameon=False)
            face = get_cmap(colormap)(1.0)
            beam.ellipse.set(facecolor=face, edgecolor='black',
                             linewidth=2)
            ax.add_artist(beam)
            ax.text(0.02, 0.01, 'Beam FWHM', transform=ax.transAxes,
                    horizontalalignment='left', weight='bold')
        else:
            log.warning('Beam keywords not found')

    if watermark:
        ax.text(0.98, 0.01, watermark, transform=ax.transAxes,
                horizontalalignment='right', alpha=0.2,
                color='gray', size='x-large')

    # add space for plots below, before making color bar
    if nrow > 1:
        fig.subplots_adjust(hspace=0.3)

    # Color bar
    bunit = map_hdu.header.get('BUNIT', None)
    label = f'Flux ({bunit})' if bunit else 'Flux'
    cax = fig.add_axes([ax.get_position().x1 + 0.01,
                        ax.get_position().y0, 0.02,
                        ax.get_position().height])
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(label)

    return fig


def make_spectral_plot(axis, wavelength, spectral_flux,
                       spectral_error=None, labels=None, scale=None,
                       colormap='viridis', xunit=None, yunit=None,
                       title=None, marker=None, marker_color='gray',
                       overplot=None, overplot_label=None,
                       overplot_color='gray', watermark=None):
    """
    Generate a plot of spectral data.

    Given a Matplotlib axis, this function adds one or more spectral plots
    to it, with optional error shading, overplots, markers, and axis labels.

    The input data should be passes as arrays.  If labels are not provided,
    then wavelength and spectral flux are passed directly to the step
    function to plot, and should follow input rules for that function.

    If labels are passed, then it is assumed special handling is desired
    for multiple spectra.  In this case, the first dimension of labels,
    wavelength, spectral_flux, and spectral_error (if passed) must match,
    so that they can be iterated together.  If more than 15 spectra are
    passed in this manner, the labels are truncated to make the associated
    legend fit in the plot.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis instance to add the plot to.
    wavelength : array-like
        May be one-dimensional (nw), or (ns, nw) if ns labels are passed.
    spectral_flux : array-like
        Must match `wavelength` dimensions.
    spectral_error : array-like, optional
        Must match `wavelength` dimensions if provided.
    labels : list of str, optional
        Labels for multiple spectra, to display in a legend. Must have
        length ns, matching spectral data with dimensions (ns, nw).
    scale : list of float, optional
        Low and high percentile values to use to set the plot scale.
        If not provided, Matplotlib defaults are used.
    colormap : str, optional
        Matplotlib color map name.  If a single spectrum is passed
        colormap[0] is used for the color.  If multiple are passed,
        their colors are distributed evenly across the colormap.
    xunit : str, optional
        Wavelength units for the x-axis label.  If xunit = cm-1, then
        the label is set to 'Wavenumber'.  Otherwise, it is 'Wavelength'.
    yunit : str, optional
        Flux units for the y-axis label.
    title : str, optional
        Title for the plot
    marker : list of list, optional
        If provided, should be [x, y], where x and y are matching lists
        of x- and y- coordinate values to place a marker at.
    marker_color : str, optional
        Matplotlib color name for the marker, if provided.
    overplot : array-like, optional
        If provided, will be plotted as a thin line on a secondary
        y-axis.  Should be provided as [x, y], where x is in the same
        units as `wavelength`.
    overplot_label : str, optional
        Label for the overplot y-axis.
    overplot_color : str, optional
        Matplotlib color name for the overplot line.
    watermark : str, optional
        If provided, the string will be added as semi-transparent text
        in the lower-right corner of the plot.
    """

    # number of spectra to plot
    if spectral_flux.ndim > 1 and spectral_flux.shape[0] > 1:
        nspec = spectral_flux.shape[0]
        color_index = np.arange(nspec) / (nspec - 1)
    else:
        color_index = [0]

    if spectral_error is None:
        do_error = False
        spectral_error = np.zeros_like(spectral_flux)
    else:
        do_error = True

    # plot flux and error
    color = get_cmap(colormap)(color_index)
    axis.set_prop_cycle('color', color)
    if labels:
        lines_display = []
        labels_display = []
        legend_limit = 15
        for j, (w, f, e) in enumerate(zip(wavelength, spectral_flux,
                                          spectral_error)):
            if j == legend_limit:
                labels[j] = '...'
            ln = axis.step(w, f, where='mid', label=labels[j])
            if do_error:
                axis.fill_between(w, f - e, f + e,
                                  step='mid', alpha=0.2)
            if j <= legend_limit or j == len(labels) - 1:
                lines_display.append(ln[0])
                labels_display.append(labels[j])
        axis.legend(lines_display, labels_display)
    else:
        axis.step(wavelength, spectral_flux, where='mid')
        if do_error:
            axis.fill_between(wavelength, spectral_flux - spectral_error,
                              spectral_flux + spectral_error,
                              step='mid', alpha=0.2)

    # set y limits if desired
    if scale is not None:
        interval = AsymmetricPercentileInterval(scale[0], scale[1])
        ymin, ymax = interval.get_limits(spectral_flux)
        axis.set_ylim(ymin, ymax)

    # mark positions on the spectrum if desired
    if marker:
        axis.scatter(marker[0], marker[1],
                     c=marker_color, marker='x', s=30,
                     alpha=0.8)

    # add an overplot if desired
    if overplot is not None:
        ax2 = axis.twinx()
        ax2.plot(overplot[0], overplot[1], linestyle=':',
                 color=overplot_color, alpha=0.8, linewidth=0.5)
        ax2.tick_params(axis='y', labelcolor=overplot_color)
        if overplot_label:
            ax2.set_ylabel(overplot_label, color=overplot_color)

    # set labels
    if xunit == 'cm-1':
        axis.set_xlabel(f'Wavenumber ({xunit})')
    elif not xunit:
        axis.set_xlabel('Wavelength')
    else:
        axis.set_xlabel(f'Wavelength ({xunit})')
    if not yunit:
        axis.set_ylabel('Spectral flux')
    else:
        axis.set_ylabel(f'Spectral flux ({yunit})')
    if title:
        axis.set_title(title)

    if watermark:
        axis.text(0.98, 0.01, watermark, transform=axis.transAxes,
                  horizontalalignment='right', alpha=0.4,
                  color='gray', size='x-large')
