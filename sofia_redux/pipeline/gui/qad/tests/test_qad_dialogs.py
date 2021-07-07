# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the QAD Dialogs classes."""

import types

import pytest

from sofia_redux.pipeline.gui.qad.qad_imview import QADImView
from sofia_redux.pipeline.gui.qad.qad_dialogs \
    import DispSettingsDialog, PhotSettingsDialog, PlotSettingsDialog
from sofia_redux.pipeline.gui.tests.test_qad_viewer import MockDS9

try:
    from PyQt5 import QtWidgets
except ImportError:
    QtWidgets = None
    HAS_PYQT5 = False
else:
    HAS_PYQT5 = True


@pytest.mark.skipif("not HAS_PYQT5")
class TestQADDialogs(object):
    """Test the QAD Settings dialogs."""
    @pytest.fixture(autouse=True, scope='function')
    def mock_app(self, qapp, mocker):
        mocker.patch.object(QtWidgets, 'QApplication',
                            return_value=qapp)

    def mock_ds9(self, mocker):
        """Mock the pyds9 DS9 class."""
        mock_pyds9 = types.ModuleType('pyds9')
        mock_pyds9.DS9 = MockDS9
        mocker.patch.dict('sys.modules', {'pyds9': mock_pyds9})

    def test_settings_noinit(self, qtbot, mocker):
        self.mock_ds9(mocker)

        # mock show and exec
        mocker.patch.object(QtWidgets.QDialog, 'show',
                            return_value=None)
        mocker.patch.object(QtWidgets.QDialog, 'exec_',
                            return_value=None)

        disp_dialog = DispSettingsDialog()
        phot_dialog = PhotSettingsDialog()
        plot_dialog = PlotSettingsDialog()
        qtbot.addWidget(disp_dialog)
        qtbot.addWidget(phot_dialog)
        qtbot.addWidget(plot_dialog)

        imviewer = QADImView()
        dkeys = list(imviewer.default_parameters('display').keys())
        pkeys = list(imviewer.default_parameters('photometry').keys())
        plkeys = list(imviewer.default_parameters('plot').keys())

        # at least some defaults should be set without error
        new_disp = disp_dialog.getValue()
        new_phot = phot_dialog.getValue()
        new_plot = plot_dialog.getValue()
        assert len(new_disp) > 0
        assert len(new_phot) > 0
        assert len(new_plot) > 0
        for key in new_disp:
            assert key in dkeys
        for key in new_phot:
            assert key in pkeys
        for key in new_plot:
            assert key in plkeys

    def test_settings_getvalue(self, qtbot, mocker):
        self.mock_ds9(mocker)

        # mock show and exec
        mocker.patch.object(QtWidgets.QDialog, 'show',
                            return_value=None)
        mocker.patch.object(QtWidgets.QDialog, 'exec_',
                            return_value=None)

        imviewer = QADImView()

        orig_disp = imviewer.disp_parameters.copy()
        orig_phot = imviewer.phot_parameters.copy()
        orig_plot = imviewer.plot_parameters.copy()
        disp = imviewer.disp_parameters.copy()
        phot = imviewer.phot_parameters.copy()
        plot = imviewer.plot_parameters.copy()

        disp_dialog = DispSettingsDialog(
            current=disp,
            default=imviewer.default_parameters('display'))
        phot_dialog = PhotSettingsDialog(
            current=phot,
            default=imviewer.default_parameters('photometry'))
        plot_dialog = PlotSettingsDialog(
            current=plot,
            default=imviewer.default_parameters('plot'))
        qtbot.addWidget(disp_dialog)
        qtbot.addWidget(phot_dialog)
        qtbot.addWidget(plot_dialog)

        # call getvalue; verify unchanged
        new_disp = disp_dialog.getValue()
        new_phot = phot_dialog.getValue()
        new_plot = plot_dialog.getValue()
        assert new_disp == orig_disp
        assert new_phot == orig_phot
        assert new_plot == orig_plot

        # change some widget values and verify change is passed on
        disp_dialog.tileBox.toggle()
        phot_dialog.radialPlotBox.toggle()
        plot_dialog.colorBox.setText('test value')
        new_disp = disp_dialog.getValue()
        new_phot = phot_dialog.getValue()
        new_plot = plot_dialog.getValue()
        assert new_disp != orig_disp
        assert new_phot != orig_phot
        assert new_plot != orig_plot
        assert new_disp['tile'] == disp_dialog.tileBox.isChecked()
        assert new_phot['show_plots'] == phot_dialog.radialPlotBox.isChecked()
        assert new_plot['color'] == \
            str(plot_dialog.colorBox.text()).lower()

        # reset values to original
        disp_dialog.reset()
        phot_dialog.reset()
        plot_dialog.reset()
        new_disp = disp_dialog.getValue()
        new_phot = phot_dialog.getValue()
        new_plot = plot_dialog.getValue()
        assert new_disp == orig_disp
        assert new_phot == orig_phot
        assert new_plot == orig_plot

        # restore defaults
        disp_dialog.restore()
        phot_dialog.restore()
        plot_dialog.restore()
        new_disp = disp_dialog.getValue()
        new_phot = phot_dialog.getValue()
        new_plot = plot_dialog.getValue()
        assert new_disp == imviewer.default_parameters('display')
        assert new_phot == imviewer.default_parameters('photometry')
        assert new_plot == imviewer.default_parameters('plot')

        # put bad values in float boxes; verify unchanged
        phot_dialog.reset()
        phot_dialog.windowSizeBox.setText('test')
        phot_dialog.fwhmBox.setText('test')
        new_phot = phot_dialog.getValue()
        assert new_phot == orig_phot
        plot_dialog.reset()
        plot_dialog.histBinBox.setText('test')
        plot_dialog.histLimitsBox.setText('test')
        plot_dialog.p2pReferenceBox.setText('test')
        new_plot = plot_dialog.getValue()
        assert new_plot == orig_plot

        # verify these get set to auto or None if not float
        phot_dialog.apradBox.setText('test')
        phot_dialog.bgrinBox.setText('test')
        phot_dialog.bgwidBox.setText('test')
        new_phot = phot_dialog.getValue()
        assert new_phot['psf_radius'] == 'auto'
        assert new_phot['bg_inner'] == 'auto'
        assert new_phot['bg_width'] == 'auto'
        phot_dialog.bgrinBox.setText('none')
        phot_dialog.bgwidBox.setText('none')
        new_phot = phot_dialog.getValue()
        assert new_phot['bg_inner'] is None
        assert new_phot['bg_width'] is None
        plot_dialog.windowSizeBox.setText('test')
        new_plot = plot_dialog.getValue()
        assert new_plot['window'] is None

    def test_settings_setvalue(self, qtbot, mocker):
        self.mock_ds9(mocker)

        # mock show and exec
        mocker.patch.object(QtWidgets.QDialog, 'show',
                            return_value=None)
        mocker.patch.object(QtWidgets.QDialog, 'exec_',
                            return_value=None)

        imviewer = QADImView()

        disp = imviewer.disp_parameters.copy()
        phot = imviewer.phot_parameters.copy()
        plot = imviewer.plot_parameters.copy()

        disp_dialog = DispSettingsDialog(
            current=disp,
            default=imviewer.default_parameters('display'))
        phot_dialog = PhotSettingsDialog(
            current=phot,
            default=imviewer.default_parameters('photometry'))
        plot_dialog = PlotSettingsDialog(
            current=plot,
            default=imviewer.default_parameters('plot'))
        qtbot.addWidget(disp_dialog)
        qtbot.addWidget(phot_dialog)
        qtbot.addWidget(plot_dialog)

        # set non-default extension
        disp['extension'] = '2'
        disp_dialog.setValue(disp)
        assert disp_dialog.extensionBox.itemText(3) == '2'

        new_disp = disp_dialog.getValue()
        assert new_disp == disp

        # remove some keys from the settings
        prev = disp.copy()
        del disp['cmap']
        del disp['zoom_fit']
        del disp['tile']
        del disp['ds9_viewer']
        del disp['ds9_viewer_qad']
        del disp['eye_viewer']
        del disp['overplots']
        disp['extension'] = 's/n'
        disp['s2n_range'] = 'test'
        disp_dialog.setValue(disp)
        new_disp = disp_dialog.getValue()
        for key, val in new_disp.items():
            if key in disp:
                if key == 's2n_range':
                    # invalid value set to default
                    assert val is None
                else:
                    assert val == disp[key]
            else:
                assert val == prev[key]

        # same for phot -- set only passed ones
        prev = phot.copy()
        phot = {'model': 'gaussian'}
        phot_dialog.setValue(phot)
        new_phot = phot_dialog.getValue()
        for key, val in new_phot.items():
            if key in phot:
                assert val == phot[key]
            else:
                assert val == prev[key]

        # same for plot -- set only passed ones
        prev = plot.copy()
        plot = {'share_axes': 'x', 'hist_limits': 'test'}
        plot_dialog.setValue(plot)
        new_plot = plot_dialog.getValue()
        for key, val in new_plot.items():
            if key in plot and key != 'hist_limits':
                assert val == plot[key]
            else:
                assert val == prev[key]
