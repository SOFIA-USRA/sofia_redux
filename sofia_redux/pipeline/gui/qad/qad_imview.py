# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""DS9 Image Viewer for QAD."""

import contextlib
import io
import os
import re
import tempfile
import warnings

from astropy.io import fits
from astropy import log, modeling, stats, table, wcs
import numpy as np
from scipy.stats import gmean

from sofia_redux.pipeline.gui.matplotlib_viewer import MatplotlibPlot
from sofia_redux.toolkit.utilities.fits import set_log_level

try:
    from sofia_redux.calibration.pipecal_photometry import pipecal_photometry
    from sofia_redux.calibration.pipecal_error import PipeCalError
except ImportError:
    HAS_PIPECAL = False
    pipecal_photometry, PipeCalError = None, None
else:
    HAS_PIPECAL = True

try:
    from sofia_redux.visualization.redux_viewer import EyeViewer
except ImportError:
    HAS_EYE = False
    EyeViewer = None
else:
    HAS_EYE = True

try:
    from PyQt5 import QtCore
except ImportError:
    HAS_PYQT5 = False

    # duck type parents to allow class definition
    class QtCore:
        class QObject:
            pass
else:
    HAS_PYQT5 = True

try:
    import regions as ar
except ImportError:
    HAS_REGIONS = False
    ar = None
else:
    HAS_REGIONS = True


class ViewerSignals(QtCore.QObject):
    make_radial_plot = QtCore.pyqtSignal()
    make_histogram_plot = QtCore.pyqtSignal()
    make_p2p_plot = QtCore.pyqtSignal()


class QADImView(object):
    """
    View FITS images in DS9.

    Attributes
    ----------
    ds9 : pyds9.DS9
        DS9 instance to display images to.
    specviewer : EyeViewer
        Eye instance to display spectra to.
    plotviewer : MatplotlibPlot
        Plot viewer widget.
    files : list of str
        Currently displayed FITS files.
    regions : list of str
        Currently displayed region files.
    headers : dict
        FITS file headers for currently displayed files.
        Keys are filenames; values are lists of headers
        (astropy.io.fits.Header)
    radial_data : list of dict
        Radial flux data, for the radial_plot method to display.
    histogram_data : list of dict
        Histogram data, for the histogram_plot method to display.
    p2p_data : list of dict
        Pixel-to-pixel data, for the p2p_plot method to display.
    ptable : astropy.table.Table
        Photometry data table.
    disp_parameters : dict
        Display settings.
    phot_parameters : dict
        Photometry settings.
    plot_parameters : dict
        Plot settings.
    cs : {'image', 'wcs', 'none'}
        Coordinate system for DS9 display and frame locks.
    """
    def __init__(self):
        """Initialize the viewer."""
        self.ds9 = None
        self.specviewer = None
        self.plotviewer = None
        self.break_loop = False
        self.files = []
        self.regions = []
        self.headers = {}
        self.radial_data = []
        self.histogram_data = []
        self.p2p_data = []
        self.ptable = None
        self.disp_parameters = self.default_parameters('display')
        self.phot_parameters = self.default_parameters('photometry')
        self.plot_parameters = self.default_parameters('plot')
        self.cs = None

        # useful for testing: flag to always break imexam loop
        self.break_loop = False

        # flag for viewer availability
        self.HAS_DS9 = True
        self.HAS_EYE = HAS_EYE
        if not self.HAS_EYE:  # pragma: no cover
            log.warning('Eye viewer is not available. Install '
                        'sofia_redux.visualization for spectral display.')

        # flag for photometry availability
        self.HAS_PIPECAL = HAS_PIPECAL
        if not self.HAS_PIPECAL:
            log.warning('Photometry tools are not available. Install '
                        'sofia_redux.calibration to enable photometry.')

        # always set XPA_METHOD to local
        os.environ["XPA_METHOD"] = "local"

        # signals for plot events
        self.HAS_PYQT5 = HAS_PYQT5
        if self.HAS_PYQT5:
            self.signals = ViewerSignals()
            self.signals.make_radial_plot.connect(self.radial_plot)
            self.signals.make_histogram_plot.connect(self.histogram_plot)
            self.signals.make_p2p_plot.connect(self.pix2pix_plot)
        else:
            log.warning('Plotting tools are not available. Install PyQt5 '
                        'to enable plotting displays.')
            self.signals = None

    def _loaded_data(self):
        """Check for loaded FITS data in current frame."""
        try:
            dsize = [int(d) for d
                     in self.run('fits size', via='get').split()]
        except (ValueError, TypeError, AttributeError) as err:
            log.debug(f'  FITS size error: {err}')
            return False
        else:
            if 0 in dsize:
                return False
            else:
                return True

    def _run_internal(self, cmd, buf=None, via='set'):
        """
        Format commands to send to PyDS9.

        Parameters
        ----------
        cmd : str
            Command to send.
        buf : list, str, or file-like; optional
            Additional buffers to send.  May be a string or a file handler,
            or a list of those.
        via : {'get', 'set'}, optional
            Command method.

        Returns
        -------
        str
            DS9 command exit status.
        """
        log.debug('  Command: {}'.format(str(cmd)))
        retval = None
        if str(via).lower().strip() == 'set':
            if type(buf) is list:
                retval = self.ds9.set(cmd, *buf)
            else:
                retval = self.ds9.set(cmd, buf)
        elif str(via).lower().strip() == 'get':
            # workaround for DS9 seg fault with empty frame
            # and wcs requests
            if 'wcs' in cmd and not self._loaded_data():
                log.debug('  No loaded data; skipping frame')
                retval = ''
            else:
                try:
                    retval = self.ds9.get(cmd)
                except TypeError as err:
                    log.error('Error in pyds9')
                    log.debug(err)
        else:
            log.error('Unknown ds9 interaction command: ' + str(via))
        return retval

    def _region_mask(self, cs, all_regions, xctr, yctr, hwcs):
        """Compute a region mask at a cursor position."""
        if not HAS_REGIONS:
            return None
        ctr_coord = ar.PixCoord(xctr, yctr)
        mask = None
        for reg_str in all_regions:
            # read ds9 string into a region class
            try:
                with set_log_level('CRITICAL'):
                    frame_regions = ar.Regions.parse(reg_str, format='ds9')
            except Exception as err:
                log.debug(f'Region parser error: {err}')
                continue
            for fr in frame_regions:
                if cs == 'wcs':
                    # convert to a pixel region first
                    try:
                        with set_log_level('CRITICAL'):
                            fr = fr.to_pixel(hwcs)
                    except Exception as err:  # pragma: no cover
                        # error could be anything, since regions package
                        # is in early development state
                        log.debug(f'Region WCS conversion error: {err}')
                        continue

                # check if cursor is contained in a region
                # in any frame
                with set_log_level('CRITICAL'):
                    contained = fr.contains(ctr_coord)
                if hasattr(contained, '__len__'):
                    # PolygonPixelRegion returns an array, currently
                    # (regions v0.4)
                    contained = contained[0]

                if contained:
                    # get mask from first matching region
                    try:
                        with set_log_level('CRITICAL'):
                            mask = fr.to_mask()
                    except Exception as err:  # pragma: no cover
                        # error could be anything, since regions package
                        # is in early development state
                        log.debug(f'Region mask error: {err}')
                        continue
                    else:
                        log.info(f'Contained in {type(fr).__name__}')
                        break
            if mask is not None:
                break

        # reset active frame
        return mask

    def default_parameters(self, ptype):
        """
        Retrieve default display, photometry, and plot parameters.

        Parameters
        ----------
        ptype : {'display', 'photometry', 'plot'}
            Parameter type to retrieve.

        Returns
        -------
        dict
            Parameter settings dictionary.
        """
        ptype = str(ptype).lower()
        if ptype == 'photometry':
            defaults = {'model': 'moffat',
                        'window': 50.0,
                        'window_units': 'pixels',
                        'fwhm': 3.0,
                        'fwhm_units': 'arcsec',
                        'psf_radius': 12.0,
                        'aperture_units': 'pixels',
                        'bg_inner': 15.0,
                        'bg_width': 10.0,
                        'show_plots': False}
        elif ptype == 'display':
            defaults = {'extension': 'first',
                        'lock_image': 'wcs',
                        'lock_slice': 'image',
                        'scale': 'zscale',
                        'cmap': 'none',
                        'zoom_fit': True,
                        'tile': True,
                        's2n_range': None,
                        'eye_viewer': True,
                        'overplots': True,
                        'ds9_viewer': True,
                        'ds9_viewer_pipeline': False,
                        'ds9_viewer_qad': True}
        elif ptype == 'plot':
            defaults = {'window': 50.0,
                        'window_units': 'pixels',
                        'color': 'tab10',
                        'share_axes': 'none',
                        'separate_plots': True,
                        'bin': 'fd',
                        'hist_limits': None,
                        'p2p_reference': 1,
                        'summary_stat': 'clipped median'}
        else:
            defaults = {}

        return defaults

    def histogram(self, ctr1, ctr2):
        """
        Create a histogram of pixel values at the selected location.

        If the cursor is over an enclosed region (circle, box, etc.),
        then the histogram is computed for the enclosed data.  Otherwise,
        the histogram is computed for an analysis window centered on the
        cursor position.  The window width can be set in the plot parameter
        dialog; if set to a blank value, the entire image is used to compute
        the histogram.  The binning strategy can also be set in the
        parameters.  Accepted values are any string accepted by the
        matplotlib hist function, or else an integer number of bins. The
        range of data for the histogram can also be set in parameters,
        as min,max values.

        Parameters
        ----------
        ctr1 : float
            RA (if DS9 coordinate system is 'wcs') or
            x-position (if DS9 coordinate system is 'image')
            at selected image location.
        ctr2 : float
            Dec (if DS9 coordinate system is 'wcs') or
            y-position (if DS9 coordinate system is 'image')
            at selected image location.
        """
        # check for the current status of the viewer
        # (tiling, aligned by wcs)
        if self.run('tile', via='get') == 'yes':
            allframes = True
            frames = self.run('frame active', via='get').split()
        else:
            allframes = False
            frames = [self.run('frame', via='get')]
        if self.run('wcs align', via='get') == 'yes':
            cs = 'wcs'
        else:
            cs = 'image'

        # get any currently available regions
        all_regions = self.run(f'regions -system {cs}',
                               allframes=allframes, via='get')
        if not allframes:
            all_regions = [all_regions]

        param = self.plot_parameters
        for frame in frames:
            log.info('')
            if allframes:
                log.info('Frame ' + frame)
                self.run('frame ' + frame)
                # check for loaded data
                if not self._loaded_data():
                    continue

            try:
                results = self.retrieve_data(ctr1, ctr2, photometry=False)
            except (ValueError, TypeError) as err:
                log.debug(f'Error in retrieving data: {err}')
                continue
            fulldata = results['fulldata']
            data = results['data']
            wdw = results['window']
            hwcs = results['wcs']
            xctr = results['xctr']
            yctr = results['yctr']

            log.info(f'Histogram at: {ctr1},{ctr2}')

            # get data from region mask or window
            mask = self._region_mask(cs, all_regions, xctr, yctr, hwcs)
            if mask is None:
                if param['window'] is None:
                    log.info('Using the full image')
                    reg_name = 'full image'
                    short_reg_name = 'full'
                    hist_data = fulldata
                else:
                    log.info(f'Using the analysis window '
                             f'(width: {wdw} pixels)')
                    reg_name = f'{wdw} pixel window'
                    short_reg_name = f'x={xctr:.0f} y={yctr:.0f} {wdw}pix'
                    hist_data = data
            else:
                reg_name = 'DS9 region'
                short_reg_name = f'x={xctr:.0f} y={yctr:.0f} region'
                hist_data = mask.multiply(fulldata)
                if hist_data is None:  # pragma: no cover
                    # condition occasionally but unreliably encountered
                    # in testing
                    log.warning('Region is too small; skipping histogram')
                    continue
                hist_data[hist_data == 0] = np.nan

            hist_data = hist_data.ravel()
            hist_minmax = (np.nanmin(hist_data), np.nanmax(hist_data),
                           np.nansum(hist_data))
            hist_stats = (np.nanmean(hist_data),
                          np.nanmedian(hist_data),
                          np.nanstd(hist_data))
            nnan = np.isfinite(hist_data)
            clip_stats = stats.sigma_clipped_stats(hist_data[nnan])
            text_stats = [f'Total pixels: {np.sum(nnan)}',
                          f'Min, max, sum: '
                          f'{hist_minmax[0]:.5g}, {hist_minmax[1]:.5g}, '
                          f'{hist_minmax[2]:.5g}',
                          f'Mean, median, stddev: '
                          f'{hist_stats[0]:.5g}, {hist_stats[1]:.5g}, '
                          f'{hist_stats[2]:.5g}',
                          f'Clipped mean, median, stddev: '
                          f'{clip_stats[0]:.5g}, {clip_stats[1]:.5g}, '
                          f'{clip_stats[2]:.5g}']
            for t in text_stats:
                log.info(t)

            title = f'Frame {frame}, x={xctr:.0f} y={yctr:.0f} in {reg_name}'
            l1 = f'F{frame} {short_reg_name}'
            hist_kwargs = {'bins': param['bin'], 'label': l1, 'alpha': 0.8}
            if param['hist_limits'] is not None:
                hist_kwargs['range'] = (param['hist_limits'][0],
                                        param['hist_limits'][1])
            new_hist = {'plot_type': 'histogram', 'args': [hist_data],
                        'kwargs': hist_kwargs}

            if param['separate_plots'] or len(self.histogram_data) < 1:
                # summary stat (mean, median, clipped mean, or clipped median)
                summary_stat = str(param.get('summary_stat', 'mean')).lower()
                if 'clip' in summary_stat:
                    se = clip_stats[2]
                    if 'median' in summary_stat:
                        ss = clip_stats[1]
                        ss_label = 'Clipped median'
                    else:
                        ss = clip_stats[0]
                        ss_label = 'Clipped mean'
                else:
                    se = hist_stats[2]
                    if 'median' in summary_stat:
                        ss = hist_stats[1]
                        ss_label = 'Median'
                    else:
                        ss = hist_stats[0]
                        ss_label = 'Mean'
                l2 = f'{ss_label} {ss:.3g} +/- {se:.3g}'

                overplots = [new_hist]
                vlines = [ss, ss - se, ss + se]
                vlabels = [l2, None, None]
                vstyles = ['-', ':', ':']
                for vdata, vlabel, vstyle in zip(vlines, vlabels, vstyles):
                    overplots.append({'plot_type': 'vline',
                                      'args': [vdata],
                                      'kwargs': {'label': vlabel,
                                                 'color': 'gray',
                                                 'linewidth': 1,
                                                 'linestyle': vstyle}})
                overplots.append({'plot_type': 'legend', 'args': []})

                plot_data = {'args': [],
                             'kwargs': {'title': title,
                                        'xlabel': 'Flux',
                                        'ylabel': 'Count',
                                        'colormap': param['color']},
                             'plot_kwargs': {},
                             'overplot': overplots}
                self.histogram_data.append(plot_data)
            else:
                # append new histogram to existing ones
                plot_data = self.histogram_data[-1]
                overplots = []
                for plot in plot_data['overplot']:
                    if plot['plot_type'] == 'histogram':
                        overplots.append(plot)
                overplots.append(new_hist)
                overplots.append({'plot_type': 'legend', 'args': []})
                plot_data['overplot'] = overplots
                plot_data['kwargs']['title'] = 'All histogram regions'

        if self.signals is not None:
            self.signals.make_histogram_plot.emit()

    def histogram_plot(self):
        """Plot radial fluxes in a separate window."""
        if not self.HAS_PYQT5:
            return
        data = self.histogram_data
        if data is None or len(data) == 0:
            return

        # start up plot viewer if needed
        if self.plotviewer is None or not self.plotviewer.isVisible():
            self.plotviewer = MatplotlibPlot()

        self.plotviewer.setWindowTitle('Histogram')
        self.plotviewer.plot_layout = 'rows'
        self.plotviewer.share_axes = self.plot_parameters['share_axes']
        self.plotviewer.plot(data)
        self.plotviewer.set_scroll('bottom')
        self.plotviewer.show()
        self.plotviewer.raise_()

    def imexam(self):
        """
        Start event loop for photometry in DS9.

        The following keypress values (over the DS9 window) are
        accepted:

        * a: Perform photometry at the cursor position.
        * p: Plot a pixel-to-pixel comparison of all loaded
          frames at the cursor location.
        * s: Compute statistics and plot a histogram of the data
          at the cursor location.
        * c: Clear active plots and photometry data.
        * h: Show a help message.
        * q: Clear previous results and close the imexam event loop.
        """
        if not self.HAS_DS9:  # pragma: no cover
            return

        usage_message = [
            '', 'Available options for ImExam:',
            '  a: Perform photometry at the cursor location.',
            '  p: Plot a pixel-to-pixel comparison of all '
            'frames at the cursor location.',
            '  s: Compute statistics and plot a histogram of the data at the '
            'cursor location.',
            '  c: Clear active plots and photometry data.',
            '  h: Show this message.',
            '  q: Quit ImExam.', '']

        for msg in usage_message:
            log.info(msg)

        keypress = 'none'
        while keypress != 'q':
            # reset ptable if necessary
            if self.ptable is None:
                self.reset_ptable()

            # get new imexam cursor
            try:
                if self.run('wcs align', via='get') == 'yes':
                    cs = 'wcs'
                else:
                    cs = 'image'
                uval = self.run(f'imexam any coordinate {cs}', via='get')
            except (ValueError, AttributeError):
                log.debug('Error in ds9 wcs or imexam command')
                log.error('Error in ImExam loop')
                break

            if str(uval).strip() == '0' or str(uval).strip() == '':
                # error condition -- break loop
                log.debug(f'ImExam returned: {uval}')
                log.error('Error in ImExam loop')
                break

            if str(uval).lower().strip().startswith('q'):
                keypress = 'q'
                xcoord = None
                ycoord = None
            else:
                uvallist = str(uval).split()
                if len(uvallist) != 3:
                    log.debug(f'Imexam returned: {uval}')
                    log.error('Error in ImExam loop')
                    break
                keypress, xcoord, ycoord = uvallist
                keypress = keypress.lower().strip()
                xcoord = float(xcoord)
                ycoord = float(ycoord)

            if keypress == 'a':
                # perform profile fits/aperture photometry
                self.photometry(xcoord, ycoord)

                # show radial plot if desired
                if self.phot_parameters['show_plots']:
                    if self.signals is not None:
                        self.signals.make_radial_plot.emit()
            elif keypress == 's':
                self.histogram(xcoord, ycoord)
            elif keypress == 'p':
                self.pix2pix(xcoord, ycoord)
            elif keypress == 'h':
                # show help message
                for msg in usage_message:
                    log.info(msg)
            elif keypress == 'c' or keypress == 'q':
                # reset imexam data
                self.radial_data = []
                self.histogram_data = []
                self.p2p_data = []
                self.ptable = None
                if self.plotviewer is not None:
                    self.plotviewer.clear()

                # check for tiling -- if tiled, delete regions
                # from all active frames
                if self.run('tile', via='get') == 'yes':
                    tile = True
                else:
                    tile = False
                self.run('regions group imexam delete', allframes=tile)

            if self.break_loop:
                # break loop if required
                break

    def run(self, cmd, buf=None, via='set', allframes=False):
        """
        Send an XPA command to DS9.

        Parameters
        ----------
        cmd : str
            Command to send.
        buf : list, str, or file-like; optional
            Additional buffers to send.  May be a string or a file handler,
            or a list of those.
        via : {'get', 'set'}, optional
            Command method.
        allframes : bool
            If True, the command should be run once for each active frame.

        Returns
        -------
        str
            DS9 command exit status.
        """
        if self.ds9 is None:
            self.startup()
        if not self.HAS_DS9:  # pragma: no cover
            return
        if allframes:
            fcmd = 'frame active'
        else:
            fcmd = 'frame'
        try:
            framenums = self._run_internal(fcmd, via='get').split()
        except (TypeError, ValueError, AttributeError) as err:
            msg = str(err)
            if (msg.lower().startswith('ds9 is no longer running')
                    or msg.lower().startswith('no response')):
                self.startup()
                framenums = self._run_internal(fcmd, via='get').split()
            elif msg.lower().startswith("'nonetype' object has no attribute"):
                framenums = []
            else:
                raise

        if allframes:
            retvalarr = []
            for frm in framenums:
                fcmd = 'frame ' + frm
                self._run_internal(fcmd)
                retval = self._run_internal(cmd, buf, via)
                retvalarr.append(retval)
            return retvalarr
        else:
            retval = self._run_internal(cmd, buf, via)
            return retval

    def get_extension_param(self):
        """Retrieve the extension setting from the display parameters."""
        if 'frame' in self.disp_parameters['extension']:
            exten = 'all'
        elif 'cube' in self.disp_parameters['extension']:
            exten = 'all'
        elif 'first' in self.disp_parameters['extension']:
            exten = 0
        else:
            try:
                exten = int(self.disp_parameters['extension'])
            except ValueError:
                exten = str(self.disp_parameters['extension']).strip()
        return exten

    def load(self, fitsfiles, regfiles=None):
        """
        Load FITS files into DS9.

        Spectral files will be loaded into the Eye viewer
        if possible.  Spectra are determined by the `spec_test` method.

        If only headers are provided, they are just stored in the `headers`
        attribute.

        Otherwise, all images and region files are loaded into DS9.

        Parameters
        ----------
        fitsfiles : list of str, `astropy.io.fits.HDUList`, \
            or `astropy.io.fits.Header`
            FITS file paths to load.  May also be HDUList instances
            or just Header instances, if the data is already in memory.
        regfiles : list of str, optional
            DS9 region file paths to load.
        """
        if type(fitsfiles) is not list:
            fitsfiles = [fitsfiles]
        if regfiles is None:
            regfiles = []
        elif type(regfiles) is not list:
            regfiles = [regfiles]

        # read fits files, categorizing into image and spectral files
        good_files = []
        img_files = []
        imgdata = []
        spec_files = []
        specdata = []
        regions = []
        headers = {}
        is_data = {}
        data_type = {}
        for j, ffile in enumerate(fitsfiles):
            if isinstance(ffile, list) and \
                    isinstance(ffile[0], fits.Header):
                # header only: set aside for header viewing
                hdr = ffile
                try:
                    ffile = hdr[0]["FILENAME"]
                except KeyError:
                    ffile = "Array {}".format(j)
                headers[ffile] = hdr
                continue
            elif isinstance(ffile, fits.HDUList):
                hdul = ffile
                hdr = hdul[0].header
                try:
                    ffile = hdr["FILENAME"]
                except KeyError:
                    ffile = "Array {}".format(j)
                is_data[ffile] = True
            elif os.path.exists(str(ffile)):
                try:
                    hdul = fits.open(ffile)
                except IOError:
                    log.warning('Cannot load {}; ignoring.'.format(ffile))
                    continue
                is_data[ffile] = False
            else:
                log.warning('Cannot load {}; ignoring.'.format(ffile))
                continue
            data_type[ffile] = self.spec_test(hdul)
            if 'spec' in data_type[ffile]:
                spec_files.append(ffile)
                specdata.append(hdul)
            if data_type[ffile] != 'spectrum_only':
                # Data that is not spectrum-only should also go to DS9
                img_files.append(ffile)
                imgdata.append(hdul)

        # load spectral files into Eye if possible
        if len(spec_files) > 0:
            if self.disp_parameters['eye_viewer'] and self.HAS_EYE:
                if self.specviewer is None:
                    self.specviewer = EyeViewer()
                    self.specviewer.start()

                # send as data if any are not available as files,
                # otherwise send as files
                if any([is_data[ffile] for ffile in spec_files]):
                    self.specviewer.update(specdata)
                else:
                    self.specviewer.update(spec_files)
                self.specviewer.display()

            # keep the headers even if the eye is closed
            for j, ffile in enumerate(spec_files):
                if data_type[ffile] == 'spectrum_only':
                    # other data will be tracked in img_files section
                    good_files.append(ffile)
                    headers[ffile] = specdata[j][0].header

        # keep headers for image files even if ds9 is unavailable
        if len(img_files) > 0:
            for j, ffile in enumerate(img_files):
                if is_data[ffile]:
                    good_files.append(imgdata[j])
                else:
                    good_files.append(ffile)

                headers[ffile] = []
                for ext in imgdata[j]:
                    headers[ffile].append(ext.header)

        # also track good region files, regardless
        for rfile in regfiles:
            if os.path.exists(rfile):
                regions.append(rfile)

        # load image files into ds9
        i = 0
        if len(img_files) > 0 and self.disp_parameters['ds9_viewer'] \
                and self.HAS_DS9:
            self.set_defaults()
            exten = ''
            if 'frame' in self.disp_parameters['extension']:
                cmd = 'multiframe'
            elif 'cube' in self.disp_parameters['extension']:
                cmd = 'mecube'
            elif 'first' in self.disp_parameters['extension']:
                cmd = 'fits'
            else:
                try:
                    exten = int(self.disp_parameters['extension'])
                except ValueError:
                    exten = str(self.disp_parameters['extension']).strip()
                cmd = 'fits'

            for j, ffile in enumerate(img_files):
                ds9_cmd = cmd

                # automagic an S/N extension if desired
                if str(exten).upper() == 'S/N':
                    try:
                        s2n = self._make_s2n(ffile, imgdata[j])
                    except ValueError as err:
                        log.error(err)
                        continue
                    is_data[ffile] = True
                    imgdata[j] = s2n

                try:
                    if not is_data[ffile]:
                        if exten != '':
                            extenstr = '[{}]'.format(exten)
                        else:
                            extenstr = ''
                        if cmd == 'multiframe':
                            ds9_cmd = "{} {}{}".format(cmd, ffile, extenstr)
                        else:
                            ds9_cmd = "{} new {}{}".format(cmd, ffile,
                                                           extenstr)
                        log.debug("Running DS9 command: {}".format(ds9_cmd))
                        status = self.run(ds9_cmd)
                    else:
                        self.run('frame new')

                        # display FILENAME keyword in addition to filename
                        self.run('view keyvalue "{}"'.format("'FILENAME'"))
                        self.run('view keyword yes')

                        if exten != '' and str(exten).upper() != 'S/N':
                            try:
                                phu = imgdata[j][0].copy()
                                phu.data = None
                                data = fits.HDUList([phu, imgdata[j][exten]])
                            except KeyError as err:
                                log.error(err)
                                raise ValueError(err)
                        else:
                            data = imgdata[j]

                        # try to stream the data to DS9.  If that fails,
                        # try to write it to disk, then load it
                        try:
                            log.debug("Loading from memory")
                            status = self._load_from_memory(cmd, ffile, data)
                        except ValueError:
                            log.debug("Loading from tempfile")
                            status = self._load_from_tempfile(cmd, ffile, data)

                except ValueError as err:
                    msg = str(err)
                    if exten != '':
                        if str(exten).lower() in msg.lower():
                            log.warning(f'Error loading extension '
                                        f'{exten} for file '
                                        f'{os.path.basename(ffile)}')
                            continue
                        else:
                            status = 0
                            log.error(f'Error in XPA command: {ds9_cmd}')
                            log.error(msg)
                    else:
                        status = 0
                        log.error(f'Error in XPA command: {ds9_cmd}')
                        log.error(msg)
                if status == 1 and self.disp_parameters['overplots']:
                    # overlay photometric and/or spectroscopic apertures
                    self.overlay()
                    if len(specdata) > 0:
                        nspec = len(specdata)
                        if i >= nspec:
                            hdr = specdata[0][0].header
                        else:
                            hdr = specdata[i][0].header
                        try:
                            with set_log_level('ERROR'):
                                hwcs = wcs.WCS(headers[ffile][0])
                        except (ValueError, IndexError, MemoryError,
                                AttributeError, TypeError):
                            hwcs = None
                        try:
                            self.overlay_aperture(hdr, hwcs=hwcs)
                        except ValueError:  # pragma: no cover
                            # may be encountered with extensions with
                            # unexpected WCSs
                            pass
                i += 1

            if str(self.cs).strip().lower() != 'none':
                self.run('match frame ' + self.cs)
            if self.disp_parameters['zoom_fit']:
                self.run('zoom to fit')
            if self.disp_parameters['cmap'].lower() != 'none':
                self.run('cmap ' + self.disp_parameters['cmap'].lower())
            if self.disp_parameters['scale'].lower() != 'none':
                self.run('scale ' + self.disp_parameters['scale'].lower())

            if len(regfiles) > 0:
                frames = self.run('frame active', via='get').split()
                nframes = len(frames)
                if nframes == len(regfiles):
                    for reg_i, rfile in enumerate(regfiles):
                        if os.path.exists(rfile):
                            self.run('frame {}'.format(frames[reg_i]))
                            self.run('region load {}'.format(rfile))
                else:
                    for rfile in regfiles:
                        if os.path.exists(rfile):
                            self.run('region load all ' + rfile)

            # reset the photometry table
            self.ptable = None
            self.radial_data = []
            self.histogram_data = []
            self.p2p_data = []
            self.run('raise')

        # keep track of the files loaded into ds9, for reloading
        # if ds9 is closed
        self.files = good_files
        self.regions = regions
        self.headers = headers

    def _load_from_tempfile(self, cmd, ffile, data):
        """Write a tempfile and load it into DS9."""
        with tempfile.NamedTemporaryFile(delete=False) as \
                new_file:
            new_name = new_file.name
        try:
            data.writeto(new_name, overwrite=True)
            ds9_cmd = "{} {}".format(cmd, new_name)
            log.debug("Running DS9 command: {}".format(ds9_cmd))
            status = self.run(ds9_cmd)
            os.remove(new_name)
        except (TypeError, OSError, ValueError):
            log.warning("Cannot load image {} "
                        "from tempfile".format(ffile))
            status = 0
        return status

    def _load_from_memory(self, cmd, ffile, data):
        """Use BytesIO to stream HDU data to DS9."""
        status = 0
        with contextlib.closing(io.BytesIO()) as new_file:
            new_file.name = ffile
            try:
                data.writeto(new_file, overwrite=True)
                new_fits = new_file.getvalue()

                log.debug("Running DS9 command: {}".format(cmd))
                status = self.run(cmd, buf=[new_fits,
                                            len(new_fits)])
            except (TypeError, ValueError):
                msg = "Cannot load image {} " \
                      "from memory".format(ffile)
                log.warning(msg)
                raise ValueError(msg)
        return status

    def _make_s2n(self, ffile, hdul):
        """Retrieve or make an S/N image."""
        bname = os.path.basename(ffile)
        phu = hdul[0].copy()
        phu.data = None
        if 'S/N' in hdul:
            hdu = hdul['S/N']
        else:
            if 'FLUX' in hdul and 'ERROR' in hdul:
                log.debug(f'Making S/N image from FLUX and ERROR '
                          f'extensions for {bname}.')
                hdu = fits.ImageHDU(hdul['FLUX'].data,
                                    hdul['FLUX'].header)
                s = hdul['FLUX'].data
                n = hdul['ERROR'].data
            elif 'FLUX' in hdul and 'STDDEV' in hdul:
                log.debug(f'Making S/N image from FLUX and STDDEV '
                          f'extensions for {bname}.')
                hdu = fits.ImageHDU(hdul['FLUX'].data,
                                    hdul['FLUX'].header)
                s = hdul['FLUX'].data
                n = hdul['STDDEV'].data
            elif 'STOKES I' in hdul and 'ERROR I' in hdul:
                log.debug(f'Making S/N image from STOKES I and '
                          f'ERROR I extensions for {bname}.')
                hdu = fits.ImageHDU(hdul['STOKES I'].data,
                                    hdul['STOKES I'].header)
                s = hdul['STOKES I'].data
                n = hdul['ERROR I'].data
            else:
                raise ValueError(f'Cannot determine S/N from extensions '
                                 f'in file {bname}')

            hdu.data = s / n
            hdu.header['EXTNAME'] = 'S/N'
            hdu.header['BUNIT'] = ''

        # blank out data outside of range
        try:
            low, high = self.disp_parameters['s2n_range']
            hdu.data[hdu.data < float(low)] = np.nan
            hdu.data[hdu.data > float(high)] = np.nan
        except (ValueError, AttributeError, IndexError, TypeError):
            pass

        s2n = fits.HDUList([phu, hdu])
        return s2n

    def lock(self, cs=None, ltype=None, off=False):
        """
        Lock DS9 frames to the desired coordinate system.

        Parameters
        ----------
        cs : {'image', 'wcs', 'none'}, optional
            Coordinate system to lock to.  Display parameters
            will be used if not provided.
        ltype : {'frame', 'crosshair', 'crop', 'slice', \
            'bin', 'scale', 'colorbar', 'smooth'}, optional
            Lock type.  If not provided, all will be locked.
        off : bool
            If True, locks will be turned off.
        """
        icstypes = ['frame', 'crosshair', 'crop']
        scstypes = ['slice']
        booltypes = ['bin', 'scale', 'colorbar', 'smooth']
        if off:
            cs = 'none'
            bval = 'no'
        else:
            bval = 'yes'

        if cs is None:
            ics = self.disp_parameters['lock_image']
            scs = self.disp_parameters['lock_slice']
        else:
            ics = cs
            scs = cs

        if ltype is not None:
            ltypes = [ltype.lower().strip()]
        else:
            # lock all
            ltypes = icstypes + scstypes + booltypes
        for ltype in ltypes:
            if ltype in icstypes:
                self.run('lock ' + ltype + ' ' + ics)
            elif ltype in scstypes:
                self.run('lock ' + ltype + ' ' + scs)
            elif ltype in booltypes:
                self.run('lock ' + ltype + ' ' + bval)
        self.cs = ics

    def overlay(self):
        """
        Overlay photometry aperture regions.

        Aperture parameters are read from the SRCPOSX/Y,
        STCENTX/Y, PHOTAPER, PHOTSKAP, STAPFLX, and
        STAPSKY header keywords in the FITS data in the
        currently displayed frame.
        """
        # retrieve header for photometry keywords
        # from current frame only
        hdr_str = self.run('fits header', via='get')

        # read it in to a fits header
        phdr = fits.Header()
        hdr = phdr.fromstring(hdr_str, sep='\n')

        try:
            srcposx = hdr['SRCPOSX'] + 1
            srcposy = hdr['SRCPOSY'] + 1
            s1 = 'point({:f} {:f}) # ' \
                 'point=x ' \
                 'color=blue tag={{srcpos}} '\
                 'text=SRCPOS'.format(srcposx, srcposy)
            self.run('regions', s1)
        except (KeyError, ValueError):
            pass
        try:
            stcentx = hdr['STCENTX'] + 1
            stcenty = hdr['STCENTY'] + 1
            photaper = hdr['PHOTAPER']
            photskap = [float(x) for x in hdr['PHOTSKAP'].split(',')]
            s1 = 'point({:f} {:f}) # ' \
                 'point=x ' \
                 'color=cyan tag={{srcpos}}'.format(stcentx, stcenty)
            self.run('regions', s1)
            s2 = 'circle({:f} {:f} {:f}) # ' \
                 'color=cyan tag={{srcpos}}'.format(
                     stcentx, stcenty, photaper)
            self.run('regions', s2)
            s3 = 'annulus({:f} {:f} {:f} {:f}) # ' \
                 'color=cyan tag={{srcpos}} text=STCENT'.format(
                     stcentx, stcenty, photskap[0], photskap[1])
            self.run('regions', s3)
        except (KeyError, ValueError):
            pass
        try:
            stcentx = hdr['STCENTX'] + 1
            stcenty = hdr['STCENTY'] + 1
            flux = hdr['STAPFLX']
            sky = hdr['STAPSKY']
            s1 = 'text({:f} {:f}) # color=cyan ' \
                 'text="Flux={:.2f}, Sky={:.2f}"'.format(
                     stcentx, stcenty - 40, flux, sky)
            self.run('regions', s1)
        except (KeyError, ValueError):
            pass

        # try overlaying apertures as well
        try:
            self.overlay_aperture(hdr)
        except ValueError:  # pragma: no cover
            # may be encountered with extensions with
            # unexpected WCSs
            pass

    def overlay_aperture(self, hdr, hwcs=None):
        """
        Overlay spectroscopic aperture regions.

        Aperture parameters are determined from the FITS header
        keywords APPOSO01, APRADO01, PSFRAD01, and BGR. A primary
        spectral WCS is required.

        Parameters
        ----------
        hdr : `astropy.io.fits.Header`
            FITS header to read apertures from.
        """
        # test for appropriately sized data
        if hwcs is None:
            try:
                with set_log_level('ERROR'):
                    hwcs = wcs.WCS(hdr)
            except (ValueError, IndexError, MemoryError,
                    AttributeError, TypeError):
                return
        naxis = hwcs.pixel_shape
        if not naxis or len(naxis) != 2:
            return

        minval = hwcs.wcs_pix2world([[1, 1]], 1)[0][0]
        maxval = hwcs.wcs_pix2world([[naxis[0], 1]], 1)[0][0]

        regions = []
        width = 1
        try:
            appos = hdr['APPOSO01'].split(',')
            if 'APRADO01' in hdr:
                aprad = hdr['APRADO01'].split(',')
            else:
                aprad = None
            if 'PSFRAD01' in hdr:
                psfrad = hdr['PSFRAD01'].split(',')
            else:
                psfrad = None
            line_template = 'wcs; linear; line({:f} {:f} {:f} {:f}) # ' \
                            'color={:s} tag={{aperture}} '\
                            'width={:d}'
            for j in range(len(appos)):
                pos = float(appos[j])
                s1 = line_template.format(minval, pos,
                                          maxval, pos, 'cyan', width)
                regions.extend([s1])

                if aprad is not None:
                    try:
                        apr = float(aprad[j])
                        s2 = line_template.format(minval, pos + apr, maxval,
                                                  pos + apr, 'green', width)
                        s3 = line_template.format(minval, pos - apr, maxval,
                                                  pos - apr, 'green', width)
                        regions.extend([s2, s3])
                    except IndexError:
                        pass

                if psfrad is not None:
                    try:
                        psf = float(psfrad[j])
                        s4 = line_template.format(minval, pos + psf, maxval,
                                                  pos + psf, 'blue', width)
                        s5 = line_template.format(minval, pos - psf, maxval,
                                                  pos - psf, 'blue', width)
                        regions.extend([s4, s5])
                    except IndexError:
                        pass
            if 'BGR' in hdr:
                bgreg = re.split('[,;]', hdr['BGR'])
                for bg in bgreg:
                    r1, r2 = bg.split('-')
                    s6 = line_template.format(minval, float(r1), maxval,
                                              float(r1), 'red', width)
                    s7 = line_template.format(minval, float(r2), maxval,
                                              float(r2), 'red', width)
                    regions.extend([s6, s7])

        except (KeyError, ValueError):
            pass

        for reg in regions:
            try:
                self.run('regions', reg)
            except ValueError:
                # A bad WCS in the ds9 image will cause the
                # region set to fail -- in this case, don't continue
                # to try region
                break

    def photometry(self, ctr1, ctr2):
        """
        Perform aperture photometry at a selected position.

        For each active frame, the following steps are performed:

        * A window around the selected position is extracted from the
          FITS image.
        * NaNs in the extracted data are removed, the absolute value
          of the extracted data is taken, and the median value of
          the extracted data is subtracted.
        * The brightest peak in the window is located.
        * A model is fit to the extracted data, centered at the
          peak location.  The model is either a 2-D Moffat or
          Gaussian, depending on the current photometry parameters.
        * A circular aperture and sky annulus are placed at the fit
          position, and fluxes are extracted.
        * Model fit parameters (centroid, FWHM, etc.) and fluxes
          are printed to the log at INFO level.

        Parameters
        ----------
        ctr1 : float
            RA (if DS9 coordinate system is 'wcs') or
            x-position (if DS9 coordinate system is 'image')
            at selected image location.
        ctr2 : float
            Dec (if DS9 coordinate system is 'wcs') or
            y-position (if DS9 coordinate system is 'image')
            at selected image location.
        """
        if not self.HAS_PIPECAL:
            return

        # reset photometry table if necessary
        if self.ptable is None:
            self.reset_ptable()

        # get photometry parameters
        param = self.phot_parameters

        # check for the current status of the viewer
        # (tiling, aligned by wcs)
        if self.run('tile', via='get') == 'yes':
            allframes = True
            frames = self.run('frame active', via='get').split()
        else:
            allframes = False
            frames = [self.run('frame', via='get')]

        for frame in frames:
            if allframes:
                log.info('Frame ' + frame)
                self.run('frame ' + frame)

            try:
                results = self.retrieve_data(ctr1, ctr2)
            except (ValueError, TypeError):
                continue
            ps = results['pix_scale']
            data = results['data']
            fulldata = results['fulldata']
            hwcs = results['wcs']
            wdw = results['window']
            xstart = results['xstart']
            ystart = results['ystart']
            xctr = results['xctr']
            yctr = results['yctr']

            # check for reasonable data
            if np.sum(np.isfinite(data)) < 3:
                continue

            default_fwhm = param['fwhm']
            if param['fwhm_units'] == 'arcsec':
                default_fwhm /= ps
            try:
                psfr = float(param['psf_radius'])
                if param['aperture_units'] == 'arcsec':
                    psfr /= ps
            except ValueError:
                # auto radius
                psfr = 2.15 * default_fwhm

            if (param['bg_inner'] is None
                    or param['bg_width'] is None):
                do_bg = False
                skyrad = (0., 0.)
            else:
                do_bg = True
                try:
                    bgrin = float(param['bg_inner'])
                    if param['aperture_units'] == 'arcsec':
                        bgrin /= ps
                except ValueError:
                    bgrin = psfr + 0.2 * default_fwhm
                try:
                    bgwid = float(param['bg_width'])
                    if param['aperture_units'] == 'arcsec':
                        bgwid /= ps
                    bgrout = bgrin + bgwid
                except ValueError:
                    bgrout = bgrin + 2.0 * default_fwhm

                if bgrout > bgrin:
                    skyrad = (bgrin, bgrout)
                else:
                    skyrad = (0., 0.)

            try:
                phot_par = pipecal_photometry(
                    fulldata, np.full_like(fulldata, np.nan),
                    srcpos=(xctr, yctr), fitsize=wdw, fwhm=default_fwhm,
                    profile=param['model'], aprad=psfr,
                    skyrad=skyrad, stamp_center=False, allow_badfit=True)
            except PipeCalError as err:
                log.warning('  Bad fit.')
                log.warning(f'  {err}')
                continue

            peak, xcent, ycent, ra, dec, xfwhm, yfwhm, ellip, \
                pa, pw_law, final_sum, bg_avg, bg_std = [np.nan] * 13
            bg_fit = 0.0
            for pp in phot_par:
                if pp['key'] == 'STPEAK':
                    peak = pp['value'][0]
                elif pp['key'] == 'STCENTX':
                    xcent = pp['value'][0]
                elif pp['key'] == 'STCENTY':
                    ycent = pp['value'][0]
                elif pp['key'] == 'STFWHMX':
                    xfwhm = pp['value'][0]
                elif pp['key'] == 'STFWHMY':
                    yfwhm = pp['value'][0]
                elif pp['key'] == 'STANGLE':
                    pa = pp['value'][0]
                elif pp['key'] == 'STPWLAW':
                    pw_law = pp['value'][0]
                elif pp['key'] == 'STAPFLX':
                    final_sum = pp['value'][0]
                elif pp['key'] == 'STAPSKY' and do_bg:
                    bg_avg = pp['value'][0]
                elif pp['key'] == 'STAPSSTD' and do_bg:
                    bg_std = pp['value']
                elif pp['key'] == 'STBKG':
                    bg_fit = pp['value'][0]

            # check whether source is already in table
            limit = 2. * default_fwhm
            present = (int(frame) == self.ptable['Frame']) \
                & (np.abs(self.ptable['X'] - (xcent + 1)) < limit) \
                & (np.abs(self.ptable['Y'] - (ycent + 1)) < limit)
            if np.any(present):
                log.info('  Source already measured.')
                continue

            # check whether source is unreasonably large or small
            badfit = False
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mfwhm = gmean([xfwhm, yfwhm])
            if np.isnan(mfwhm) or mfwhm > 20 or mfwhm < 1.0:
                log.warning('  Bad fit.')
                log.warning('  Calculated FWHM: {:.2f} pixels'.format(mfwhm))
                badfit = True
                mfwhm = np.nan
                ellip = np.nan
                pa = np.nan
            else:
                # calculate ellipticity and fix PA
                if xfwhm >= yfwhm:
                    ellip = 1 - yfwhm / xfwhm
                else:
                    ellip = 1 - xfwhm / yfwhm
                    if pa <= 0:
                        pa += 90
                    else:
                        pa -= 90

            # track flux by radial distance
            if param['show_plots']:
                y, x = np.mgrid[:wdw, :wdw]
                r = np.sqrt((x - xcent + xstart) ** 2
                            + (y - ycent + ystart) ** 2)
                if badfit:
                    moddata = None
                else:
                    # get the equivalent 1D model from the profile fit
                    # for plotting
                    if param['model'] == 'gaussian':
                        eqw = mfwhm * stats.gaussian_fwhm_to_sigma
                        rmodel = modeling.models.Gaussian1D(peak, 0.0, eqw)
                    else:
                        n_1 = 1 / pw_law
                        eqw = mfwhm / (2 * np.sqrt(2 ** n_1 - 1))
                        rmodel = modeling.models.Moffat1D(
                            peak, 0.0, eqw, pw_law)
                    rmodel += modeling.models.Const1D(bg_fit)
                    moddata = rmodel(r)

                # data for matplotlib viewer: primary is model;
                # scatter data and h/v lines are overplots
                rflat = r.ravel()
                dflat = data.ravel()
                sortidx = np.argsort(rflat)
                xdata = rflat[sortidx]
                overplots = [{'plot_type': 'scatter',
                              'args': [rflat, dflat],
                              'kwargs': {'marker': '*',
                                         'c': dflat,
                                         'label': 'Flux data'}},
                             {'plot_type': 'hline',
                              'args': [0.0],
                              'kwargs': {'linestyle': ':',
                                         'linewidth': 1,
                                         'color': 'lightgray'}}]
                if moddata is not None:
                    ydata = moddata.ravel()[sortidx]
                    overplots.append({'plot_type': 'vline',
                                      'args': [mfwhm / 2.0],
                                      'kwargs': {
                                          'linestyle': ':',
                                          'linewidth': 1,
                                          'color': '#ff7f0e',
                                          'label': 'Fit HWHM'}})
                    overplots.append({'plot_type': 'vline',
                                      'args': [mfwhm],
                                      'kwargs': {
                                          'linestyle': ':',
                                          'linewidth': 1,
                                          'color': '#d62728',
                                          'label': 'Fit FWHM'}})
                else:
                    ydata = np.full_like(xdata, np.nan)

                title = f'Frame {frame}, x={xcent:.0f} y={ycent:.0f}'
                overplots.append({'plot_type': 'legend',
                                  'args': []})
                plot_data = {'args': [xdata, ydata],
                             'kwargs': {
                                 'title': title,
                                 'xlabel': 'Distance (pixels)',
                                 'ylabel': 'Flux'},
                             'plot_kwargs': {
                                 'linestyle': '-',
                                 'color': 'gray',
                                 'label': f"{param['model'].title()} profile"},
                             'overplot': overplots}
                self.radial_data.append(plot_data)

            # add DS9 start index back into centroid and convert to RA/Dec
            xcent += 1
            ycent += 1
            if hwcs is not None:
                try:
                    radec = hwcs.wcs_pix2world([[xcent, ycent, 1]], 1)
                except ValueError:
                    try:
                        radec = hwcs.wcs_pix2world([[xcent, ycent]], 1)
                    except ValueError:
                        radec = np.array([[None, None]])
            else:
                radec = np.array([[None, None]])

            # set region
            b0 = 'point({:f} {:f}) # ' \
                 'point=x ' \
                 'color=green tag={{imexam}}'.format(xcent, ycent)
            self.run('regions', b0)
            b1 = 'circle({:f} {:f} {:f}) # ' \
                'color=green tag={{imexam}}'.format(xcent, ycent, psfr)
            self.run('regions', b1)
            if do_bg:
                b2 = 'annulus({:f} {:f} {:f} {:f}) # ' \
                    'color=red ' \
                     'tag={{imexam}}'.format(xcent, ycent,
                                             skyrad[0], skyrad[1])
                self.run('regions', b2)

            self.ptable.add_row([frame, peak, xcent, ycent,
                                 radec[0, 0], radec[0, 1],
                                 mfwhm, mfwhm * ps, ellip, pa,
                                 final_sum, bg_avg, bg_std])

        self.ptable.sort(['Frame', 'Peak'])
        print_str = '\n'.join(
            self.ptable.pformat(max_lines=-1, max_width=-1))
        log.info('\n{}\n'.format(print_str))

    def pix2pix(self, ctr1, ctr2):
        """
        Create a comparison of pixel values at the selected location.

        Two or more frames must be loaded to generate a pixel comparison
        plot.  The reference frame is Frame 1 by default, but can be
        changed in the plot parameters.

        If the cursor is over an enclosed region (circle, box, etc.),
        then the comparison is computed for the enclosed data.  Otherwise,
        the comparison is computed for an analysis window centered on the
        cursor position.  The window width can be set in the plot parameter
        dialog; if set to a blank value, the entire image is used to compute
        the comparison.

        Parameters
        ----------
        ctr1 : float
            RA (if DS9 coordinate system is 'wcs') or
            x-position (if DS9 coordinate system is 'image')
            at selected image location.
        ctr2 : float
            Dec (if DS9 coordinate system is 'wcs') or
            y-position (if DS9 coordinate system is 'image')
            at selected image location.
        """
        # check for the current status of the viewer
        # (tiling, aligned by wcs)
        if self.run('tile', via='get') == 'yes':
            allframes = True
            frames = self.run('frame active', via='get').split()
        else:
            allframes = False
            frames = [self.run('frame', via='get')]
        if self.run('wcs align', via='get') == 'yes':
            cs = 'wcs'
        else:
            cs = 'image'

        if len(frames) < 2:
            log.info('')
            log.info('Pixel comparison requires 2 or more '
                     'frames to be shown.')
            return

        # get any currently available regions
        all_regions = self.run(f'regions -system {cs}',
                               allframes=allframes, via='get')
        if not allframes:  # pragma: no cover
            # this shouldn't be reachable, since frames >= 2
            all_regions = [all_regions]

        param = self.plot_parameters
        p2p_data_sets = {}
        label = {}
        reg_name = {}
        for frame in frames:
            log.info('')
            if allframes:
                log.info('Frame ' + frame)
                self.run('frame ' + frame)
                # check for loaded data
                if not self._loaded_data():
                    continue

            try:
                results = self.retrieve_data(ctr1, ctr2, photometry=False)
            except (ValueError, TypeError) as err:
                log.debug(f'Error in retrieving data: {err}')
                continue
            fulldata = results['fulldata']
            data = results['data']
            wdw = results['window']
            hwcs = results['wcs']
            xctr = results['xctr']
            yctr = results['yctr']

            log.info(f'Pixel comparison at: {ctr1},{ctr2}')

            # get data from region mask or window
            mask = self._region_mask(cs, all_regions, xctr, yctr, hwcs)
            if mask is None:
                if param['window'] is None:
                    log.info('Using the full image')
                    reg_name[frame] = 'full image'
                    short_reg_name = 'full'
                    p2p_data = fulldata
                else:
                    log.info(f'Using the analysis window '
                             f'(width: {wdw} pixels)')
                    reg_name[frame] = f'{wdw} pixel window'
                    short_reg_name = f'x={xctr:.0f} y={yctr:.0f} {wdw}pix'
                    p2p_data = data
            else:
                reg_name[frame] = 'DS9 region'
                short_reg_name = f'x={xctr:.0f} y={yctr:.0f} region'
                p2p_data = mask.multiply(fulldata)
                p2p_data[p2p_data == 0] = np.nan

            p2p_data = p2p_data.ravel()
            p2p_data_sets[frame] = p2p_data
            label[frame] = f'F{frame} {short_reg_name}'

        log.info('')
        ref_frame = str(param['p2p_reference'])
        log.info(f'Reference frame: {ref_frame}')
        if ref_frame in p2p_data_sets:

            ref_data = p2p_data_sets[ref_frame]
            mp = (np.nanmean(ref_data), np.nanmean(ref_data))
            overplots = []
            for f in p2p_data_sets:
                if f != ref_frame:
                    if p2p_data_sets[f].size == ref_data.size:
                        new_p2p = {'plot_type': 'plot',
                                   'args': [ref_data, p2p_data_sets[f]],
                                   'kwargs': {'label': label[f],
                                              'alpha': 0.8,
                                              'linestyle': '',
                                              'marker': '.'}}
                        overplots.append(new_p2p)
                    else:  # pragma: no cover
                        # todo: this clause and the next are unreachable
                        #  with current mock test fixture, since it
                        #  returns the same data array for all frames
                        log.warning(f'Data pixelation mismatch; '
                                    f'skipping frame {f}')
            if len(overplots) < 1:  # pragma: no cover
                log.info('No data to plot.')
                return

            title = f'Pixel comparison to Frame {ref_frame}, ' \
                    f'{reg_name[ref_frame]}'
            legend = {'plot_type': 'legend', 'args': []}
            line = {'plot_type': 'line', 'args': [mp],
                    'kwargs': {'linestyle': ':', 'color': 'gray',
                               'slope': 1.0}}
            if param['separate_plots'] or len(self.p2p_data) < 1:
                overplots.extend([line, legend])
                plot_data = {'args': [],
                             'kwargs': {'title': title,
                                        'xlabel': 'Reference flux',
                                        'ylabel': 'Comparison flux',
                                        'colormap': param['color']},
                             'plot_kwargs': {},
                             'overplot': overplots}
                self.p2p_data.append(plot_data)
            else:
                # append new dataset to existing ones
                plot_data = self.p2p_data[-1]
                old_overplots = []
                for plot in plot_data['overplot']:
                    if plot['plot_type'] == 'plot':
                        old_overplots.append(plot)
                old_overplots.extend(overplots)
                old_overplots.extend([line, legend])
                plot_data['overplot'] = old_overplots
                plot_data['kwargs']['title'] = 'All pixel comparisons'

            if self.signals is not None:
                self.signals.make_p2p_plot.emit()
        else:
            log.warning(f'Reference frame {ref_frame} is not loaded')

    def pix2pix_plot(self):
        """Plot pixel-to-pixel comparison in a separate window."""
        if not self.HAS_PYQT5:
            return
        data = self.p2p_data
        if data is None or len(data) == 0:
            return

        # start up plot viewer if needed
        if self.plotviewer is None or not self.plotviewer.isVisible():
            self.plotviewer = MatplotlibPlot()

        self.plotviewer.setWindowTitle('Pixel-to-Pixel Comparison')
        self.plotviewer.plot_layout = 'rows'
        self.plotviewer.share_axes = self.plot_parameters['share_axes']
        self.plotviewer.plot(data)
        self.plotviewer.set_scroll('bottom')
        self.plotviewer.show()
        self.plotviewer.raise_()

    def radial_plot(self):
        """Plot radial fluxes in a separate window."""
        if not self.HAS_PYQT5:
            return
        data = self.radial_data
        if data is None or len(data) == 0:
            return

        # start up plot viewer if needed
        if self.plotviewer is None or not self.plotviewer.isVisible():
            self.plotviewer = MatplotlibPlot()

        self.plotviewer.setWindowTitle('Radial Profiles')
        self.plotviewer.plot_layout = 'rows'
        self.plotviewer.share_axes = self.plot_parameters['share_axes']
        self.plotviewer.plot(data)
        self.plotviewer.set_scroll('bottom')
        self.plotviewer.show()
        self.plotviewer.raise_()

    def reload(self):
        """Reload previously loaded FITS files."""
        if len(self.files) > 0:
            self.load(self.files, regfiles=self.regions)

    def reset(self):
        """Reset to starting state."""
        self.files = []
        self.regions = []
        self.headers = {}
        self.radial_data = []
        self.histogram_data = []
        self.p2p_data = []
        self.ptable = None

    def reset_ptable(self):
        self.ptable = table.Table(names=['Frame', 'Peak', 'X', 'Y',
                                         'RA', 'Dec',
                                         'FWHM (px)', 'FWHM (")',
                                         'Ellip.', 'PA', 'Flux', 'BG/pix',
                                         'BG Std.'],
                                  dtype=[int, float, float, float,
                                         float, float, float, float,
                                         float, float, float, float,
                                         float])
        flt_col = ['Peak', 'X', 'Y', 'FWHM (px)', 'FWHM (")',
                   'Ellip.', 'PA', 'Flux', 'BG/pix', 'BG Std.']
        for col in flt_col:
            self.ptable[col].format = '.4g'
        self.ptable['RA'].format = '.7g'
        self.ptable['Dec'].format = '.7g'

    def retrieve_data(self, ctr1, ctr2, cube=False, photometry=True):
        """
        Extract a sub region from the active frame.

        Parameters
        ----------
        ctr1 : float
            RA (if DS9 coordinate system is 'wcs') or
            x-position (if DS9 coordinate system is 'image')
            at selected image location.
        ctr2 : float
            Dec (if DS9 coordinate system is 'wcs') or
            y-position (if DS9 coordinate system is 'image')
            at selected image location.
        cube : bool, optional
            If True, extract a region from the full data cube.
            Otherwise, extract from the current slice only.
        photometry : bool, optional
            If True, the photometry parameters are used to determine
            the data window. Otherwise, the plot parameters are used.

        Returns
        -------
        dict
            Extracted data for the current frame.  Keys are:
            'cs', 'pix_scale', 'wcs', 'window', 'xstart',
            'ystart', 'xctr', 'yctr', 'data', 'fulldata',
            'header'
        """
        if photometry:
            param = self.phot_parameters
        else:
            param = self.plot_parameters
        if self.run('wcs align', via='get') == 'yes':
            cs = 'wcs'
        else:
            cs = 'image'

        # retrieve header for wcs from current frame
        hdr_str = self.run('fits header', via='get')
        phdr = fits.Header()
        hdr = phdr.fromstring(hdr_str, sep='\n')
        try:
            with set_log_level('ERROR'):
                hwcs = wcs.WCS(hdr)
            psd2 = wcs.utils.proj_plane_pixel_scales(hwcs.celestial)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ps = np.mean(psd2) * 3600
        except (ValueError, IndexError, MemoryError,
                AttributeError, TypeError):
            log.warning('Error reading WCS.  Setting plate scale to 1.0.')
            hwcs = None
            ps = 1.0
        if np.isnan(ps):
            log.warning('Error reading WCS.  Setting plate scale to 1.0.')
            hwcs = None
            ps = 1.0

        # convert RA/Dec to pix if necessary
        if cs == 'wcs' and hwcs is not None:
            if hwcs.naxis == 2:
                ctr = [[ctr1, ctr2]]
            else:
                ctr = [[ctr1, ctr2, 1]]
            pixctr = hwcs.wcs_world2pix(ctr, 1)

            # occasionally, values get returned as pixels instead
            # of RA/Dec - eg. when there is an empty frame around
            if np.any(np.isnan(pixctr)):
                log.debug('WCS position retrieval failed; '
                          'assuming pixel postions')
                xctr = ctr1
                yctr = ctr2
            else:
                xctr = pixctr[0, 0]
                yctr = pixctr[0, 1]
        else:
            xctr = ctr1
            yctr = ctr2

        # subtract 1 for numpy indexing
        xctr -= 1
        yctr -= 1

        # retrieve data from current slice from viewer
        try:
            pdata = self.ds9.get_arr2np()
        except ValueError:
            log.warning('Displayed data cannot be retrieved as an array.')
            log.warning('Try turning off cube display for '
                        'multi-extension files.')
            raise
        dim = pdata.shape
        if len(dim) > 2:
            zdim, ydim, xdim = dim
        else:
            ydim, xdim = dim
            zdim = 0
        dslice = int(self.run('cube', via='get'))

        # set fitting window and retrieve subimage
        if param['window'] is None:
            pix_wdw = np.inf
        elif param['window_units'] == 'arcsec':
            pix_wdw = param['window'] / ps
        else:
            pix_wdw = param['window']
        wdw = int(np.min([pix_wdw, xdim, ydim]))
        xstart = int(xctr - wdw / 2.0)
        ystart = int(yctr - wdw / 2.0)
        if xstart < 0:
            xstart = 0
        elif xstart + wdw > xdim:
            xstart = xdim - wdw
        if ystart < 0:
            ystart = 0
        elif ystart + wdw > ydim:
            ystart = ydim - wdw
        if zdim >= dslice:
            if not cube:
                log.debug(f'Retrieving slice {dslice}')
                fulldata = pdata[dslice - 1, :, :]
                data = pdata[dslice - 1,
                             ystart:ystart + wdw,
                             xstart:xstart + wdw]
            else:
                fulldata = pdata
                data = pdata[:, ystart:ystart + wdw, xstart:xstart + wdw]
        elif zdim > 0:
            fulldata = pdata[0, :, :]
            data = pdata[0, ystart:ystart + wdw, xstart:xstart + wdw]
        else:
            fulldata = pdata
            data = pdata[ystart:ystart + wdw, xstart:xstart + wdw]

        results = {'cs': cs,
                   'pix_scale': ps,
                   'wcs': hwcs,
                   'window': wdw,
                   'xstart': xstart,
                   'ystart': ystart,
                   'xctr': xctr,
                   'yctr': yctr,
                   'data': data,
                   'fulldata': fulldata,
                   'header': hdr}

        return results

    def set_defaults(self):
        """Reset DS9 configuration from display parameters."""
        if not self.HAS_DS9:  # pragma: no cover
            return
        self.run('frame delete all')
        self.run('wcs degrees')
        if self.disp_parameters['tile']:
            self.run('tile yes')
        else:
            self.run('tile no')
        self.cs = str(self.disp_parameters['lock_image']).lower()
        self.lock()

    def spec_test(self, hdul):
        """
        Test for a spectrum the Eye can display.

        The data is considered spectral if a 'spectral_flux'
        extension is present.  It is 'spectrum_only'
        if NAXIS1 is greater than 0 and NAXIS2 is less than 6.
        Otherwise, it is 'image'.

        Parameters
        ----------
        header : `astropy.io.fits.Header`
            FITS header to test.

        Returns
        -------
        str
            'spectrum', 'spectrum_only', or 'image'.
        """
        if 'spectral_flux' in hdul:
            return 'spectrum'
        header = hdul[0].header
        if 'NAXIS1' in header:
            xdim = header['NAXIS1']
        else:
            xdim = 0
        if 'NAXIS2' in header:
            ydim = header['NAXIS2']
        else:
            ydim = 0
        if xdim > 0 and ydim < 6:
            return 'spectrum_only'
        else:
            return 'image'

    def startup(self):
        """Start up DS9."""
        log.debug('Starting DS9.')

        # lazy import pyds9 because it has non-trivial startup behavior
        try:
            import pyds9
        except (ImportError, ValueError):
            # PyDS9 sometimes fails to import for internal reasons.
            log.error('Cannot import PyDS9. DS9 display '
                      'will not be available.')
            self.HAS_DS9 = False
            return
        else:
            self.HAS_DS9 = True

        try:
            self.ds9 = pyds9.DS9()
        except (TypeError, ValueError):
            raise ValueError('DS9 is not accessible.') from None

        # reset files and regions instead
        self.files = []
        self.regions = []

    def quit(self):
        """Quit DS9."""
        self.break_loop = True
        try:
            self._run_internal('quit')
        except Exception:
            pass
