import os
import logging

import pytest

from sofia_redux.visualization.display.text_view import TextView
from sofia_redux.visualization.display.reference_window import ReferenceWindow
from sofia_redux.visualization.models.reference_model import ReferenceData

PyQt5 = pytest.importorskip('PyQt5')


class TestReferenceWindow(object):

    def test_connections(self, empty_view, mocker):
        mock_load_line = mocker.patch.object(ReferenceWindow, 'load_lines',
                                             return_value=None)

        mock_line = mocker.patch.object(ReferenceWindow, 'toggle_visibility',
                                        return_value=None)

        mock_clear_lists = mocker.patch.object(ReferenceWindow, 'clear_lists',
                                               return_value=None)
        mock_show_text = mocker.patch.object(ReferenceWindow, 'show_text',
                                             return_value=None)

        parent = empty_view

        rw = ReferenceWindow(parent)
        assert isinstance(rw, PyQt5.QtWidgets.QDialog)

        rw.load_file_button.clicked.emit()
        assert mock_load_line.called

        rw.show_lines_box.toggled.emit(False)
        mock_line.assert_called_with('ref_line')

        rw.show_labels_box.toggled.emit(False)
        mock_line.assert_called_with('ref_label')

        rw.clear_lists_button.clicked.emit()
        assert mock_clear_lists.called

        # mock some item - QT specific class
        rw.loaded_files_list.itemDoubleClicked.emit(
            PyQt5.QtWidgets.QListWidgetItem())
        assert mock_show_text.called

    def test_load_lines(self, empty_view, line_list_csv, qtbot, mocker):
        mocker.patch.object(PyQt5.QtWidgets.QFileDialog, 'getOpenFileName',
                            return_value=[line_list_csv])
        parent = empty_view
        rw = ReferenceWindow(parent)

        mock_add_line_list = mocker.patch.object(rw.ref_models,
                                                 'add_line_list',
                                                 return_value=True)

        with qtbot.wait_signal(rw.signals.update_reference_lines):
            rw.load_lines()

        assert mock_add_line_list.called

        assert rw.visibility['ref_line']
        assert rw.visibility['ref_label']

        buttons = [rw.show_lines_box, rw.show_labels_box]
        for button in buttons:
            assert button.isChecked()

        assert 'Loaded' in rw.status.text()

    @pytest.mark.parametrize('output',
                             [([]), ([False]), ()]
                             )
    def test_load_lines_nofile(self, output, empty_view, line_list_csv, qtbot,
                               mocker):
        mocker.patch.object(PyQt5.QtWidgets.QFileDialog, 'getOpenFileName',
                            return_value=output)
        parent = empty_view
        rw = ReferenceWindow(parent)

        assert rw.load_lines() is False

    def test_load_lines_alt(self, empty_view, line_list_csv, qtbot, mocker):
        mocker.patch.object(PyQt5.QtWidgets.QFileDialog, 'getOpenFileName',
                            return_value=[line_list_csv])
        mocker.patch.object(ReferenceData, 'add_line_list',
                            side_effect=IOError)

        parent = empty_view
        rw = ReferenceWindow(parent)

        assert rw.load_lines() is False

    def test_load_lines_false(self, empty_view, line_list_csv, mocker):
        mocker.patch.object(PyQt5.QtWidgets.QFileDialog, 'getOpenFileName',
                            return_value=[line_list_csv])
        parent = empty_view
        rw = ReferenceWindow(parent)

        mocker.patch.object(rw.ref_models, 'add_line_list',
                            return_value=False)
        rw.load_lines()
        assert 'Unable to parse' in rw.status.text()

    @pytest.mark.parametrize('target', ['ref_label', 'ref_line'])
    def test_toggle_visibility(self, target, empty_view, qtbot,
                               mocker, caplog):
        caplog.set_level(logging.DEBUG)
        parent = empty_view
        rw = ReferenceWindow(parent)
        mock_visibility = mocker.patch.object(parent.reference_models,
                                              'set_visibility')

        with qtbot.wait_signal(rw.signals.update_reference_lines):
            rw.toggle_visibility(target)

        assert mock_visibility.call_count == 1
        assert 'Updated visibility of' in caplog.text

    def test_toggle_visibility_empty(self, empty_view, qtbot, mocker, caplog):
        caplog.set_level(logging.DEBUG)
        parent = empty_view
        rw = ReferenceWindow(parent)
        mock_visibility = mocker.patch.object(parent.reference_models,
                                              'set_visibility')
        target = 'ref'

        rw.toggle_visibility(target)
        assert mock_visibility.call_count == 0
        assert 'Invalid visibility target' in caplog.text

    def test_show_text(self, empty_view, line_list_csv, mocker, qtbot):
        mocker.patch.object(PyQt5.QtWidgets.QListWidgetItem,
                            'data', return_value=line_list_csv)
        mock_textview_load = mocker.patch.object(TextView,
                                                 'load', return_value=None)
        mocker.patch.object(TextView, 'show', return_value=None)

        QW = PyQt5.QtWidgets.QListWidgetItem()
        filename = QW.data(line_list_csv)
        assert filename == line_list_csv

        parent = empty_view
        rw = ReferenceWindow(parent)

        assert rw.textview is None
        rw.show_text(QW)

        assert isinstance(rw.textview, TextView)
        assert mock_textview_load.called
        assert os.path.basename(line_list_csv) == rw.textview.windowTitle()

    def test_set_status(self, empty_view):
        parent = empty_view
        rw = ReferenceWindow(parent)
        message = 'random text'
        assert rw.status.text() == ''
        rw.set_status(message)
        assert rw.status.text() == message

    def test_clear_status(self, empty_view):
        parent = empty_view
        rw = ReferenceWindow(parent)
        message = 'random text'
        rw.set_status(message)
        rw.clear_status()
        assert rw.status.text() == ''

    def test_clear_lists(self, empty_view, qtbot, mocker):
        parent = empty_view
        rw = ReferenceWindow(parent)

        # check for unload signal
        with qtbot.wait_signal(rw.signals.unload_reference_model):
            rw.clear_lists()
