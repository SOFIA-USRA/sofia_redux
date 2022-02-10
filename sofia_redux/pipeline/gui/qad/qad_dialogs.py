# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Dialogs for the QAD standalone GUI."""

try:
    from PyQt5 import QtWidgets, QtCore
    from sofia_redux.pipeline.gui.qad.ui import ui_qad_disp_settings
    from sofia_redux.pipeline.gui.qad.ui import ui_qad_phot_settings
    from sofia_redux.pipeline.gui.qad.ui import ui_qad_plot_settings
except ImportError:
    HAS_PYQT5 = False
    QtCore = None

    # duck type parents to allow class definition
    class QtWidgets:
        class QDialog:
            pass

    class ui_qad_disp_settings:
        class Ui_DisplayDialog:
            pass

    class ui_qad_phot_settings:
        class Ui_PhotometryDialog:
            pass

    class ui_qad_plot_settings:
        class Ui_PlotSettingsDialog:
            pass

else:
    HAS_PYQT5 = True


class DispSettingsDialog(QtWidgets.QDialog,
                         ui_qad_disp_settings.Ui_DisplayDialog):
    """Retrieve user preferences for display parameters."""
    def __init__(self, parent=None, current=None, default=None):
        """
        Build the settings dialog.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        current : dict, optional
            Current display parameter dictionary
            (`QADImView.disp_parameters`).
        default : dict, optional
            Default display parameter dictionary
            (`QADImView.default_parameters`).
        """
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for QAD.')

        # parent initialization
        QtWidgets.QDialog.__init__(self, parent)

        # set up UI from Designer generated file
        self.setupUi(self)

        # connect signals to slots
        self.buttonBox.button(
            QtWidgets.QDialogButtonBox.Reset).clicked.connect(self.reset)
        self.buttonBox.button(
            QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(
                self.restore)

        # set values from current, if available
        try:
            self.current = current.copy()
        except AttributeError:
            self.current = current
        if self.current is not None:
            self.setValue(self.current)
        try:
            self.default = default.copy()
        except AttributeError:
            self.default = default

    def getValue(self):
        """Get current display settings from widgets."""
        if self.current is not None:
            parameters = self.current.copy()
        else:
            parameters = {}

        parameters['extension'] = str(self.extensionBox.currentText()).lower()
        parameters['lock_image'] = str(self.lockImageBox.currentText()).lower()
        parameters['lock_slice'] = str(self.lockSliceBox.currentText()).lower()
        parameters['scale'] = str(self.scaleBox.currentText()).lower()
        parameters['cmap'] = str(self.colorMapBox.text()).lower()
        parameters['zoom_fit'] = self.zoomBox.isChecked()
        parameters['tile'] = self.tileBox.isChecked()
        parameters['eye_viewer'] = not self.disableEyeBox.isChecked()
        parameters['overplots'] = not self.disableOverplotsBox.isChecked()
        parameters['ds9_viewer_qad'] = not self.disableDS9Box.isChecked()
        parameters['ds9_viewer'] = parameters['ds9_viewer_qad']

        try:
            lim = str(self.snRangeBox.text()).split(',')
            parameters['s2n_range'] = [float(lim[0]), float(lim[1])]
        except (ValueError, TypeError, AttributeError, IndexError):
            parameters['s2n_range'] = None

        return parameters

    def reset(self):
        """Reset values to initial settings."""
        if self.current is not None:
            self.setValue(self.current)
            self.repaint()

    def restore(self):
        """Restore values from default settings."""
        if self.default is not None:
            self.setValue(self.default)
            self.repaint()

    def setValue(self, fromdict):
        """
        Set widget values for display settings.

        Parameters
        ----------
        fromdict : dict
            Display parameter dictionary (`QADImView.disp_parameters`).
        """
        # extension combo box
        if 'extension' in fromdict:
            ext = fromdict['extension'].lower()
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

        # DS9, Eye, and overplots disable check boxes
        try:
            self.disableDS9Box.setChecked(not fromdict['ds9_viewer_qad'])
        except KeyError:
            pass
        try:
            self.disableEyeBox.setChecked(not fromdict['eye_viewer'])
        except KeyError:
            pass
        try:
            self.disableOverplotsBox.setChecked(not fromdict['overplots'])
        except KeyError:
            pass


class PhotSettingsDialog(QtWidgets.QDialog,
                         ui_qad_phot_settings.Ui_PhotometryDialog):
    """Retrieve user preferences for photometry parameters."""
    def __init__(self, parent=None, current=None, default=None):
        """
        Build the settings dialog.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        current : dict, optional
            Current photometry parameter dictionary
            (`QADImView.phot_parameters`).
        default : dict, optional
            Default photometry parameter dictionary
            (`QADImView.default_parameters`).
        """
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for QAD.')

        # parent initialization
        QtWidgets.QDialog.__init__(self, parent)

        # set up UI from Designer generated file
        self.setupUi(self)

        # connect signals to slots
        self.buttonBox.button(
            QtWidgets.QDialogButtonBox.Reset).clicked.connect(self.reset)
        self.buttonBox.button(
            QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(
                self.restore)

        # set values from current, if available
        try:
            self.current = current.copy()
        except AttributeError:
            self.current = current
        if self.current is not None:
            self.setValue(self.current)
        try:
            self.default = default.copy()
        except AttributeError:
            self.default = default

    def getValue(self):
        """Get current photometry settings from widgets."""
        if self.current is not None:
            parameters = self.current.copy()
        else:
            parameters = {}

        parameters['model'] = str(self.modelTypeBox.currentText()).lower()
        parameters['fwhm_units'] = str(self.fwhmUnitsBox.currentText()).lower()
        parameters['aperture_units'] = str(
            self.apradUnitsBox.currentText()).lower()
        parameters['window_units'] = str(
            self.windowUnitsBox.currentText()).lower()
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

        return parameters

    def reset(self):
        """Reset values to initial settings."""
        if self.current is not None:
            self.setValue(self.current)
            self.repaint()

    def restore(self):
        """Restore values from default settings."""
        if self.default is not None:
            self.setValue(self.default)
            self.repaint()

    def setValue(self, fromdict):
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
            idx = self.windowUnitsBox.findText(units,
                                               QtCore.Qt.MatchFixedString)
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


class PlotSettingsDialog(QtWidgets.QDialog,
                         ui_qad_plot_settings.Ui_PlotSettingsDialog):
    """Retrieve user preferences for plot parameters."""
    def __init__(self, parent=None, current=None, default=None):
        """
        Build the settings dialog.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        current : dict, optional
            Current plot parameter dictionary
            (`QADImView.plot_parameters`).
        default : dict, optional
            Default plot parameter dictionary
            (`QADImView.default_parameters`).
        """
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for QAD.')

        # parent initialization
        QtWidgets.QDialog.__init__(self, parent)

        # set up UI from Designer generated file
        self.setupUi(self)

        # connect signals to slots
        self.buttonBox.button(
            QtWidgets.QDialogButtonBox.Reset).clicked.connect(self.reset)
        self.buttonBox.button(
            QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(
                self.restore)

        # set values from current, if available
        try:
            self.current = current.copy()
        except AttributeError:
            self.current = current
        if self.current is not None:
            self.setValue(self.current)
        try:
            self.default = default.copy()
        except AttributeError:
            self.default = default

    def getValue(self):
        """Get current plot settings from widgets."""
        if self.current is not None:
            parameters = self.current.copy()
        else:
            parameters = {}

        try:
            parameters['window'] = float(self.windowSizeBox.text())
        except ValueError:
            parameters['window'] = None
        parameters['window_units'] = str(
            self.windowUnitsBox.currentText()).lower()

        parameters['color'] = str(self.colorBox.text()).lower()
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

        return parameters

    def reset(self):
        """Reset values to initial settings."""
        if self.current is not None:
            self.setValue(self.current)
            self.repaint()

    def restore(self):
        """Restore values from default settings."""
        if self.default is not None:
            self.setValue(self.default)
            self.repaint()

    def setValue(self, fromdict):
        """
        Set widget values for plot settings.

        Parameters
        ----------
        fromdict : dict
            Plot parameter dictionary (`QADImView.plot_parameters`).
        """
        # window text
        try:
            self.windowSizeBox.setText(str(fromdict['window']))
        except KeyError:
            pass

        # window units list
        if 'window_units' in fromdict:
            units = fromdict['window_units'].lower()
            idx = self.windowUnitsBox.findText(units,
                                               QtCore.Qt.MatchFixedString)
            if idx != -1:
                self.windowUnitsBox.setCurrentIndex(idx)

        # color cycle text
        try:
            self.colorBox.setText(str(fromdict['color']))
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
