# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Redux Viewer using QAD modules for display to DS9."""

import os

import configobj
from astropy import log

from sofia_redux.pipeline.viewer import Viewer
from sofia_redux.pipeline.gui.widgets import GeneralRunnable

try:
    from PyQt5 import QtWidgets, QtCore
    from sofia_redux.pipeline.gui.ui import ui_qad_settings
except ImportError:
    HAS_PYQT5 = False
    QtCore = None

    # duck type parents to allow class definition
    class QtWidgets:
        class QWidget:
            pass

    class ui_qad_settings:
        class Ui_Form:
            pass
else:
    HAS_PYQT5 = True


class QADViewerSettings(QtWidgets.QWidget, ui_qad_settings.Ui_Form):
    """
    Settings widget for QAD Viewer.

    All attributes and methods for this class are intended for internal
    use, in response to user actions within a Qt5 application.
    """
    def __init__(self, imviewer, parent=None):
        """
        Start up the settings widget.

        Parameters
        ----------
        imviewer : `sofia_redux.pipeline.gui.qad.qad_imview.QADImView`
            Associated QAD Image Viewer.
        parent : `QWidget`, optional
            Parent widget.  May be any Qt Widget.
        """
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for Redux GUI.')

        # parent initialization
        super().__init__()

        self.setupUi(self)

        self.imviewer = imviewer
        self.headviewer = None
        self.cfg_dir = None
        self.load_worker = None
        self.imexam_worker = None

        self.dispInitial = self.imviewer.disp_parameters.copy()
        self.photInitial = self.imviewer.phot_parameters.copy()
        self.plotInitial = self.imviewer.plot_parameters.copy()

        # set initial widget values from viewer parameters
        self.setDispValue(self.dispInitial)
        self.setPhotValue(self.photInitial)
        self.setPlotValue(self.plotInitial)

        # connect signals to slots
        self.disableDS9Box.stateChanged.connect(self.getDispValue)
        self.disableOverplotsBox.stateChanged.connect(self.getDispValue)
        self.extensionBox.currentIndexChanged.connect(self.getDispValue)
        self.extensionBox.lineEdit().editingFinished.connect(self.getDispValue)
        self.lockImageBox.currentIndexChanged.connect(self.getDispValue)
        self.lockSliceBox.currentIndexChanged.connect(self.getDispValue)
        self.scaleBox.currentIndexChanged.connect(self.getDispValue)
        self.colorMapBox.editingFinished.connect(self.getDispValue)
        self.zoomBox.stateChanged.connect(self.getDispValue)
        self.tileBox.stateChanged.connect(self.getDispValue)
        self.snRangeBox.editingFinished.connect(self.getDispValue)

        self.modelTypeBox.currentIndexChanged.connect(self.getPhotValue)
        self.windowUnitsBox.currentIndexChanged.connect(self.getPhotValue)
        self.fwhmUnitsBox.currentIndexChanged.connect(self.getPhotValue)
        self.apradUnitsBox.currentIndexChanged.connect(self.getPhotValue)
        self.windowSizeBox.editingFinished.connect(self.getPhotValue)
        self.fwhmBox.editingFinished.connect(self.getPhotValue)
        self.apradBox.editingFinished.connect(self.getPhotValue)
        self.bgrinBox.editingFinished.connect(self.getPhotValue)
        self.bgwidBox.editingFinished.connect(self.getPhotValue)
        self.radialPlotBox.stateChanged.connect(self.getPhotValue)

        self.plotWindowUnitsBox.currentIndexChanged.connect(self.getPlotValue)
        self.shareAxesBox.currentIndexChanged.connect(self.getPlotValue)
        self.plotWindowSizeBox.editingFinished.connect(self.getPlotValue)
        self.plotColorBox.editingFinished.connect(self.getPlotValue)
        self.histBinBox.editingFinished.connect(self.getPlotValue)
        self.histLimitsBox.editingFinished.connect(self.getPlotValue)
        self.summaryStatBox.currentIndexChanged.connect(self.getPlotValue)
        self.p2pReferenceBox.editingFinished.connect(self.getPlotValue)
        self.separatePlotsBox.stateChanged.connect(self.getPlotValue)

        self.resetDispButton.clicked.connect(self.resetDisp)
        self.restoreDispButton.clicked.connect(self.restoreDisp)
        self.resetPhotButton.clicked.connect(self.resetPhot)
        self.restorePhotButton.clicked.connect(self.restorePhot)
        self.resetPlotButton.clicked.connect(self.resetPlot)
        self.restorePlotButton.clicked.connect(self.restorePlot)

        self.imexamButton.clicked.connect(self.onImExam)
        self.headerButton.clicked.connect(self.onDisplayHeader)
        self.saveButton.clicked.connect(self.onSave)

    def getDispValue(self):
        """Get current display settings from widgets."""
        parameters = self.imviewer.disp_parameters.copy()

        parameters['extension'] = str(self.extensionBox.currentText()).lower()
        parameters['lock_image'] = str(self.lockImageBox.currentText()).lower()
        parameters['lock_slice'] = str(self.lockSliceBox.currentText()).lower()
        parameters['scale'] = str(self.scaleBox.currentText()).lower()
        parameters['cmap'] = str(self.colorMapBox.text()).lower()
        parameters['zoom_fit'] = self.zoomBox.isChecked()
        parameters['tile'] = self.tileBox.isChecked()
        parameters['overplots'] = not self.disableOverplotsBox.isChecked()
        parameters['ds9_viewer_pipeline'] = not self.disableDS9Box.isChecked()
        parameters['ds9_viewer'] = parameters['ds9_viewer_pipeline']

        try:
            lim = str(self.snRangeBox.text()).split(',')
            parameters['s2n_range'] = [float(lim[0]), float(lim[1])]
        except (ValueError, TypeError, AttributeError, IndexError):
            parameters['s2n_range'] = None

        # attempt redisplay if any parameters have changed
        if parameters != self.imviewer.disp_parameters:
            self.imviewer.disp_parameters = parameters
            self.onLoad(None)

    def getPhotValue(self):
        """Get current photometry settings from widgets."""
        parameters = self.imviewer.phot_parameters

        parameters['model'] = \
            str(self.modelTypeBox.currentText()).lower()
        parameters['fwhm_units'] = \
            str(self.fwhmUnitsBox.currentText()).lower()
        parameters['window_units'] = \
            str(self.windowUnitsBox.currentText()).lower()
        parameters['aperture_units'] = \
            str(self.apradUnitsBox.currentText()).lower()
        try:
            parameters['window'] = float(self.windowSizeBox.text())
        except ValueError:
            pass
        try:
            parameters['fwhm'] = float(self.fwhmBox.text())
        except ValueError:
            pass
        try:
            parameters['psf_radius'] = float(self.apradBox.text())
        except ValueError:
            parameters['psf_radius'] = 'auto'
        bgi = self.bgrinBox.text()
        if str(bgi).lower() == 'none':
            parameters['bg_inner'] = None
        else:
            try:
                parameters['bg_inner'] = float(bgi)
            except ValueError:
                parameters['bg_inner'] = 'auto'
        bgw = self.bgwidBox.text()
        if str(bgw).lower() == 'none':
            parameters['bg_width'] = None
        else:
            try:
                parameters['bg_width'] = float(bgw)
            except ValueError:
                parameters['bg_width'] = 'auto'

        parameters['show_plots'] = self.radialPlotBox.isChecked()

        self.imviewer.phot_parameters = parameters

    def getPlotValue(self):
        """Get current plot settings from widgets."""
        parameters = self.imviewer.plot_parameters

        parameters['window_units'] = \
            str(self.plotWindowUnitsBox.currentText()).lower()

        try:
            parameters['window'] = float(self.plotWindowSizeBox.text())
        except ValueError:
            parameters['window'] = None

        parameters['color'] = str(self.plotColorBox.text()).lower()
        parameters['share_axes'] = str(self.shareAxesBox.currentText()).lower()
        parameters['separate_plots'] = self.separatePlotsBox.isChecked()

        try:
            parameters['bin'] = int(self.histBinBox.text())
        except ValueError:
            bin_val = str(self.histBinBox.text()).lower()
            if bin_val in ['auto', 'fd', 'doane', 'scott', 'stone',
                           'rice', 'sturges', 'sqrt']:
                parameters['bin'] = bin_val
            else:
                parameters['bin'] = 'fd'

        try:
            lim = str(self.histLimitsBox.text()).split(',')
            parameters['hist_limits'] = [float(lim[0]), float(lim[1])]
        except (ValueError, TypeError, AttributeError, IndexError):
            parameters['hist_limits'] = None

        parameters['summary_stat'] = str(
            self.summaryStatBox.currentText()).lower()

        try:
            parameters['p2p_reference'] = int(self.p2pReferenceBox.text())
        except ValueError:
            parameters['p2p_reference'] = 1

        self.imviewer.plot_parameters = parameters

    def onDisplayHeader(self):
        """Display header data in a separate widget."""
        if self.headviewer is None or not self.headviewer.isVisible():
            try:
                from .qad import qad_headview
            except ImportError:
                log.error("QAD not found; Viewer will not display headers.")
                return
            self.headviewer = qad_headview.HeaderViewer(self)

        headers = self.imviewer.headers
        if len(headers) == 0:
            self.setStatus("No headers to display.")
        else:
            title = os.path.basename(sorted(list(headers.keys()))[0])
            if len(headers) > 1:
                title += '...'

            # check the display settings -- if a particular extension
            # is specified, only retrieve its header.  Otherwise,
            # get them all
            try:
                exten = self.imviewer.get_extension_param()
            except (ValueError, TypeError, IndexError,
                    AttributeError, KeyError):
                exten = 'all'
            if str(exten) != 'all':
                new_headers = {}
                for fpath in headers:
                    try:
                        # integer extension
                        new_headers[fpath] = [headers[fpath][exten]]
                    except TypeError:
                        # string extension
                        exten = str(exten).lower()
                        for ext in headers[fpath]:
                            if 'EXTNAME' in ext and \
                                    ext['EXTNAME'].strip().lower() == exten:
                                new_headers[fpath] = [ext]
                                break
                        if fpath not in new_headers:
                            log.warning(f'No extension {exten} found for '
                                        f'{fpath}; displaying all headers')
                            new_headers[fpath] = headers[fpath]
                    except (ValueError, IndexError, TypeError, KeyError,
                            AttributeError):
                        # unrecognized extension
                        log.warning(f'No extension {exten} found for '
                                    f'{fpath}; displaying all headers')
                        new_headers[fpath] = headers[fpath]
                headers = new_headers

            self.headviewer.load(headers)
            self.headviewer.show()
            self.headviewer.raise_()
            self.headviewer.setTitle("Header for: {}".format(title))
            self.setStatus("FITS headers displayed.")

    def onImExam(self):
        """Start imexam in a new thread."""
        self.imexamButton.setEnabled(False)
        self.setStatus("Starting imexam in DS9; press 'q' to quit.")
        self.imviewer.break_loop = False
        threadpool = QtCore.QThreadPool.globalInstance()
        self.imexam_worker = GeneralRunnable(self.imviewer.imexam)
        self.imexam_worker.signals.finished.connect(self.imexamFinish)
        threadpool.start(self.imexam_worker)

    def imexamFinish(self, status):
        """
        ImExam callback.

        Parameters
        ----------
        status : None or tuple
            If not None, contains an error message to log.
        """
        if status is not None:
            # log the error
            log.error("\n{}".format(status[2]))
        self.imexam_worker = None
        self.imviewer.break_loop = True
        self.setStatus('')
        self.imexamButton.setEnabled(True)

    def onLoad(self, data, regfiles=None):
        """Start DS9 load in a new thread."""
        if self.load_worker is not None:
            self.setStatus("Skipping display; DS9 is busy")
            return
        else:
            self.setStatus("Loading QAD data.")
        threadpool = QtCore.QThreadPool.globalInstance()
        if data is None:
            self.load_worker = GeneralRunnable(self.imviewer.reload)
        else:
            self.load_worker = GeneralRunnable(self.imviewer.load, data,
                                               regfiles=regfiles)
        self.load_worker.signals.finished.connect(self.loadFinish)
        threadpool.start(self.load_worker)

    def loadFinish(self, status):
        """
        DS9 load callback.

        Parameters
        ----------
        status : None or tuple
            If not None, contains an error message to log.
        """
        if status is not None:
            # log the error
            log.error("\n{}".format(status[2]))
        self.load_worker = None
        self.setStatus('')

    def onSave(self):
        """Save QAD parameters."""
        if self.cfg_dir is None or not os.path.isdir(self.cfg_dir):
            log.error('No config directory available; not saving parameters')
            return

        # display parameters
        config = configobj.ConfigObj(self.imviewer.disp_parameters,
                                     unrepr=True)
        config.filename = os.path.join(self.cfg_dir, 'display.cfg')
        config.write()

        # photometry parameters
        config = configobj.ConfigObj(self.imviewer.phot_parameters,
                                     unrepr=True)
        config.filename = os.path.join(self.cfg_dir, 'photometry.cfg')
        config.write()

        # plot parameters
        config = configobj.ConfigObj(self.imviewer.plot_parameters,
                                     unrepr=True)
        config.filename = os.path.join(self.cfg_dir, 'plot.cfg')
        config.write()

        self.setStatus('Settings saved to {:s}'.format(self.cfg_dir))

    def resetDisp(self):
        """Reset display settings to initial values."""
        self.setDispValue(self.dispInitial)
        self.getDispValue()

    def resetPhot(self):
        """Reset photometry settings to initial values."""
        self.setPhotValue(self.photInitial)
        self.getPhotValue()

    def resetPlot(self):
        """Reset plot settings to initial values."""
        self.setPlotValue(self.plotInitial)
        self.getPlotValue()

    def restoreDisp(self):
        """Restore display settings to default values."""
        self.setDispValue(self.imviewer.default_parameters('display'))
        self.getDispValue()

    def restorePhot(self):
        """Restore photometry settings to default values."""
        self.setPhotValue(self.imviewer.default_parameters('photometry'))
        self.getPhotValue()

    def restorePlot(self):
        """Restore plot settings to default values."""
        self.setPlotValue(self.imviewer.default_parameters('plot'))
        self.getPlotValue()

    def setDispValue(self, fromdict):
        """
        Set widget values for display settings.

        Parameters
        ----------
        fromdict : dict
            Display parameter dictionary (`QADImView.disp_parameters`).
        """
        # extension combo box
        if 'extension' in fromdict:
            ext = str(fromdict['extension']).lower()
            idx = self.extensionBox.findText(ext, QtCore.Qt.MatchFixedString)
            if idx != -1:
                self.extensionBox.setCurrentIndex(idx)
            else:
                # set text in last entry (extnum)
                idx = 3
                self.extensionBox.setCurrentIndex(idx)
                self.extensionBox.setItemText(idx, ext.upper())

        # lock type combo boxes
        if 'lock_image' in fromdict:
            lock = fromdict['lock_image'].lower()
            idx = self.lockImageBox.findText(lock, QtCore.Qt.MatchFixedString)
            if idx != -1:
                self.lockImageBox.setCurrentIndex(idx)
        if 'lock_slice' in fromdict:
            lock = fromdict['lock_slice'].lower()
            idx = self.lockSliceBox.findText(lock, QtCore.Qt.MatchFixedString)
            if idx != -1:
                self.lockSliceBox.setCurrentIndex(idx)

        # scale combo box
        if 'scale' in fromdict:
            scale = fromdict['scale'].lower()
            idx = self.scaleBox.findText(scale, QtCore.Qt.MatchFixedString)
            if idx != -1:
                self.scaleBox.setCurrentIndex(idx)

        # colormap text
        try:
            self.colorMapBox.setText(str(fromdict['cmap']))
        except KeyError:
            pass

        # zoom to fit check box
        try:
            self.zoomBox.setChecked(fromdict['zoom_fit'])
        except KeyError:
            pass

        # tile check box
        try:
            self.tileBox.setChecked(fromdict['tile'])
        except KeyError:
            pass

        # s2n range text
        try:
            str_lim = [str(f) for f in fromdict['s2n_range']]
            self.snRangeBox.setText(','.join(str_lim))
        except (KeyError, ValueError, TypeError, AttributeError, IndexError):
            self.snRangeBox.setText('')

        # overplots check box
        try:
            self.disableOverplotsBox.setChecked(not fromdict['overplots'])
        except KeyError:
            pass

        # DS9 disable check box
        try:
            self.disableDS9Box.setChecked(not fromdict['ds9_viewer_pipeline'])
        except KeyError:
            pass

        self.repaint()

    def setPhotValue(self, fromdict):
        """
        Set widget values for photometry settings.

        Parameters
        ----------
        fromdict : dict
            Photometry parameter dictionary (`QADImView.phot_parameters`).
        """
        # model list
        if 'model' in fromdict:
            mname = fromdict['model'].lower()
            idx = self.modelTypeBox.findText(mname, QtCore.Qt.MatchFixedString)
            if idx != -1:
                self.modelTypeBox.setCurrentIndex(idx)

        # window text
        try:
            self.windowSizeBox.setText(str(fromdict['window']))
        except KeyError:
            pass

        # window units list
        if 'window_units' in fromdict:
            units = fromdict['window_units'].lower()
            idx = self.windowUnitsBox.findText(
                units, QtCore.Qt.MatchFixedString)
            if idx != -1:
                self.windowUnitsBox.setCurrentIndex(idx)

        # fwhm text
        try:
            self.fwhmBox.setText(str(fromdict['fwhm']))
        except KeyError:
            pass

        # fwhm units list
        if 'fwhm_units' in fromdict:
            units = fromdict['fwhm_units'].lower()
            idx = self.fwhmUnitsBox.findText(units, QtCore.Qt.MatchFixedString)
            if idx != -1:
                self.fwhmUnitsBox.setCurrentIndex(idx)

        # aperture (PSF) radius text
        try:
            self.apradBox.setText(str(fromdict['psf_radius']))
        except KeyError:
            pass

        # aperture units list
        if 'aperture_units' in fromdict:
            units = fromdict['aperture_units'].lower()
            idx = self.apradUnitsBox.findText(units,
                                              QtCore.Qt.MatchFixedString)
            if idx != -1:
                self.apradUnitsBox.setCurrentIndex(idx)

        # bg inner text
        try:
            self.bgrinBox.setText(str(fromdict['bg_inner']))
        except KeyError:
            pass

        # bg width text
        try:
            self.bgwidBox.setText(str(fromdict['bg_width']))
        except KeyError:
            pass

        # show plots check box
        try:
            self.radialPlotBox.setChecked(fromdict['show_plots'])
        except KeyError:
            pass

        self.repaint()

    def setPlotValue(self, fromdict):
        """
        Set widget values for plot settings.

        Parameters
        ----------
        fromdict : dict
            Plot parameter dictionary (`QADImView.plot_parameters`).
        """
        # window text
        try:
            self.plotWindowSizeBox.setText(str(fromdict['window']))
        except KeyError:
            pass

        # window units list
        if 'window_units' in fromdict:
            units = fromdict['window_units'].lower()
            idx = self.plotWindowUnitsBox.findText(
                units, QtCore.Qt.MatchFixedString)
            if idx != -1:
                self.plotWindowUnitsBox.setCurrentIndex(idx)

        # color cycle text
        try:
            self.plotColorBox.setText(str(fromdict['color']))
        except KeyError:
            pass

        # share axes list
        if 'share_axes' in fromdict:
            ax = fromdict['share_axes'].lower()
            idx = self.shareAxesBox.findText(ax, QtCore.Qt.MatchFixedString)
            if idx != -1:
                self.shareAxesBox.setCurrentIndex(idx)

        # separate plots check box
        try:
            self.separatePlotsBox.setChecked(fromdict['separate_plots'])
        except KeyError:
            pass

        # histogram binning text
        try:
            self.histBinBox.setText(str(fromdict['bin']))
        except KeyError:
            pass

        # histogram limits text
        try:
            str_lim = [str(f) for f in fromdict['hist_limits']]
            self.histLimitsBox.setText(','.join(str_lim))
        except (KeyError, ValueError, TypeError, AttributeError, IndexError):
            self.histLimitsBox.setText('')

        # summary stat list
        if 'summary_stat' in fromdict:
            ax = fromdict['summary_stat'].lower()
            idx = self.summaryStatBox.findText(ax, QtCore.Qt.MatchFixedString)
            if idx != -1:
                self.summaryStatBox.setCurrentIndex(idx)

        # P2P reference frame text
        try:
            self.p2pReferenceBox.setText(str(fromdict['p2p_reference']))
        except KeyError:
            pass

        self.repaint()

    def setStatus(self, msg):
        """
        Set a status message.

        Parameters
        ----------
        msg : str
            Status message to display.
        """
        self.status.setText(msg)
        self.status.repaint()


class QADViewer(Viewer):
    """
    Redux Viewer interface to DS9 and the Eye of SOFIA.

    Uses pyds9 to control DS9 for image and cube display,
    and the Eye viewer for spectra.

    Attributes
    ----------
    imviewer : `sofia_redux.pipeline.gui.qad.qad_imview.QADImView`
        QAD Image and Spectrum Viewer.
    settings : `QADViewerSettings`
        Control widgets for the QADViewer.

    See Also
    --------
    sofia_redux.pipeline.gui.qad.qad_app : standalone QAD application
    """
    def __init__(self):
        """Initialize the QAD Viewer."""
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for Redux GUI.')

        super().__init__()
        self.name = "QADViewer"
        self.embedded = True

        self.imviewer = None
        self.settings = None

    def start(self, parent=None):
        """
        Start up the viewer.

        Parameters
        ----------
        parent : QtWidgets.QSplitter
            Widget to add the viewer settings to.  May be any Qt
            widget with an `addWidget` method.
        """
        self.parent = parent

        # import at last minute to avoid pyds9
        # startup behavior until necessary
        try:
            from .qad.qad_imview import QADImView
            self.imviewer = QADImView()
        except ImportError:
            log.error("QAD not found; Viewer will not display data.")
            self.embedded = False
            return
        try:
            import pyds9
            assert pyds9
        except ImportError:
            log.warning('DS9 not found. Images will not display.')
            HAS_DS9 = False
        else:
            HAS_DS9 = True

        # read settings if available
        cfg_dir = os.path.join(os.path.expanduser('~'), '.qad')
        os.makedirs(cfg_dir, exist_ok=True)

        disp_cfg = os.path.join(cfg_dir, 'display.cfg')
        if os.path.isfile(disp_cfg):
            config = configobj.ConfigObj(disp_cfg, unrepr=True)
            self.imviewer.disp_parameters.update(config.dict())
        phot_cfg = os.path.join(cfg_dir, 'photometry.cfg')
        if os.path.isfile(phot_cfg):
            config = configobj.ConfigObj(phot_cfg, unrepr=True)
            self.imviewer.phot_parameters.update(config.dict())
        plot_cfg = os.path.join(cfg_dir, 'plot.cfg')
        if os.path.isfile(plot_cfg):
            config = configobj.ConfigObj(plot_cfg, unrepr=True)
            self.imviewer.plot_parameters.update(config.dict())

        # set ds9 disable preference to pipeline version
        # (separate from standalone preference)
        self.imviewer.disp_parameters['ds9_viewer'] = \
            self.imviewer.disp_parameters['ds9_viewer_pipeline']

        # disable EyeViewer -- it is run separately for the
        # Redux interface
        self.imviewer.HAS_EYE = False
        # set initial value for ds9 check -- it will check itself as well
        self.imviewer.HAS_DS9 = HAS_DS9

        self.settings = QADViewerSettings(self.imviewer, parent=parent)
        self.settings.cfg_dir = cfg_dir

        # add widget to parent
        parent.addWidget(self.settings)
        log.debug('QAD Viewer started.')

    def display(self):
        """
        Display data.

        Data items should be set in the `display_data` attribute,
        by the `update` method.

        This method dispatches FITS files, DS9 region files, and
        data arrays to the QAD image viewer.  Any other files are
        ignored.
        """
        if self.imviewer is None:
            return

        # break imexam loop on new load
        self.imviewer.break_loop = True

        imview_items = []
        reg_files = []
        for item in self.display_data:
            if os.path.isfile(str(item)):
                if item.endswith('.reg'):
                    # ds9 region files: pass to imviewer
                    reg_files.append(item)
                elif item.endswith('.fits'):
                    # FITS files: pass to imviewer
                    imview_items.append(item)
            else:
                # probably data arrays -- pass to imviewer
                imview_items.append(item)

        # load imviewable items in a separate thread
        if len(imview_items) > 0:
            self.settings.onLoad(imview_items, regfiles=reg_files)

    def reset(self):
        """Reset the viewer."""
        if self.imviewer is None:
            return
        self.imviewer.reset()

    def close(self):
        """Close the viewer."""
        if self.imviewer is not None:
            log.debug("Quitting DS9.")
            self.imviewer.break_loop = True
            self.imviewer.quit()
