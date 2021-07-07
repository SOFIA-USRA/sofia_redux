# Licensed under a 3-clause BSD style license - see LICENSE.rst

import logging
import pytest
import numpy.testing as npt
import matplotlib.backend_bases as bb

from sofia_redux.visualization import eye
from sofia_redux.visualization.display import (view, pane, fitting_results,
                                               cursor_location)

PyQt5 = pytest.importorskip('PyQt5')


class TestEye(object):

    def test_init(self, empty_view, mocker, capsys, log_args):
        mocker.patch.object(view, 'View', return_value=empty_view)
        eye.Eye(log_args)
        log_ = capsys.readouterr().out
        assert 'Applying command line' in log_
        assert 'Setup Eye' in log_

    def test_open_eye(self, mocker, qapp, log_args, capsys):
        opening = mocker.patch.object(view.View, 'open_eye',
                                      return_value=None)
        show = mocker.patch.object(PyQt5.QtWidgets.QMainWindow, 'show',
                                   return_value=None)
        mocker.patch.object(PyQt5.QtWidgets, 'QApplication',
                            return_value=qapp)
        exec_ = mocker.patch.object(PyQt5.QtWidgets.QApplication, 'exec_',
                                    return_value=None)

        eye_app = eye.Eye(log_args)
        eye_app.open_eye()
        log_ = capsys.readouterr().out

        assert opening.called_once()
        assert show.called_once()
        assert exec_.called_once()
        assert 'Opening Eye' in log_

    def test_load_file(self, mocker, qtbot, qapp, spectral_filenames,
                       caplog):
        mocker.patch.object(PyQt5.QtWidgets.QMainWindow, 'show',
                            return_value=None)
        mocker.patch.object(PyQt5.QtWidgets, 'QApplication',
                            return_value=qapp)
        window = mocker.patch.object(PyQt5.QtWidgets.QFileDialog,
                                     'getOpenFileNames',
                                     return_value=[spectral_filenames])

        app = eye.Eye()

        assert app.view.file_table_widget.rowCount() == 0
        qtbot.mouseClick(app.view.add_file_button, PyQt5.QtCore.Qt.LeftButton)
        app.view.refresh_controls()
        assert window.called_once()

        assert app.view.file_table_widget.rowCount() == len(spectral_filenames)

        assert app.view.pane_tree_display.topLevelItemCount() == 0
        assert not app.view.figure.populated()
        app.view.file_table_widget.selectRow(0)
        mocker.patch.object(app.view.file_table_widget, 'hasFocus',
                            return_value=True)
        qtbot.keyClick(app.view.file_table_widget,
                       PyQt5.QtCore.Qt.Key_Return)

        app.view.refresh_controls()
        assert app.view.figure.populated()
        assert app.view.pane_tree_display.topLevelItemCount() == 1
        assert app.view.order_list_widget.count() == 1

        # Add pane
        with qtbot.wait_signal(app.signals.current_pane_changed):
            qtbot.mouseClick(app.view.add_pane_button,
                             PyQt5.QtCore.Qt.LeftButton)

        count = app.view.pane_tree_display.topLevelItemCount()
        assert app.view.figure.pane_count() == 2
        assert count == 2
        assert all([isinstance(p, pane.OneDimPane)
                    for p in app.view.figure.panes])
        assert len(app.view.figure.panes[0].models) == 1
        assert len(app.view.figure.panes[-1].models) == 0

        # Remove pane
        root = app.view.pane_tree_display.invisibleRootItem()
        item = root.child(count - 1)
        app.view.pane_tree_display.setCurrentItem(item)

        with qtbot.wait_signal(app.signals.atrophy_bg_full):
            qtbot.mouseClick(app.view.remove_pane_button,
                             PyQt5.QtCore.Qt.LeftButton)
        app.view.refresh_controls()
        count = app.view.pane_tree_display.topLevelItemCount()
        assert app.view.figure.pane_count() == 1
        assert count == 1
        assert len(app.view.figure.panes[0].models) == 1

    def test_change_scales(self, loaded_eye, qtbot):
        assert not loaded_eye.view.x_scale_log_button.isChecked()
        assert loaded_eye.view.x_scale_linear_button.isChecked()
        with qtbot.wait_signals([loaded_eye.signals.atrophy_bg_partial,
                                 loaded_eye.signals.axis_scale_changed],
                                order='none'):
            loaded_eye.view.x_scale_log_button.setChecked(True)

        qtbot.wait(1000)
        assert loaded_eye.view.x_scale_log_button.isChecked()
        assert not loaded_eye.view.x_scale_linear_button.isChecked()
        assert all([p.get_scale('x') == 'log'
                    for p in loaded_eye.view.figure.panes])
        assert all([p.get_scale('y') == 'linear'
                    for p in loaded_eye.view.figure.panes])

        assert not loaded_eye.view.y_scale_log_button.isChecked()
        assert loaded_eye.view.y_scale_linear_button.isChecked()
        with qtbot.wait_signals([loaded_eye.signals.atrophy_bg_partial,
                                 loaded_eye.signals.axis_scale_changed],
                                order='none'):
            loaded_eye.view.y_scale_log_button.setChecked(True)

        qtbot.wait(1000)
        assert loaded_eye.view.y_scale_log_button.isChecked()
        assert not loaded_eye.view.y_scale_linear_button.isChecked()
        assert all([p.get_scale('x') == 'log'
                    for p in loaded_eye.view.figure.panes])
        assert all([p.get_scale('y') == 'log'
                    for p in loaded_eye.view.figure.panes])

        # test with alt axis
        with qtbot.wait_signal(loaded_eye.view.signals.atrophy_bg_full):
            qtbot.mouseClick(loaded_eye.view.enable_overplot_checkbox,
                             PyQt5.QtCore.Qt.LeftButton)
        loaded_eye.view.axes_selector.setCurrentText('Current Overplot')
        with qtbot.wait_signals([loaded_eye.signals.atrophy_bg_partial,
                                 loaded_eye.signals.axis_scale_changed],
                                order='none'):
            loaded_eye.view.y_scale_log_button.setChecked(True)
        qtbot.wait(1000)
        assert loaded_eye.view.y_scale_log_button.isChecked()
        assert not loaded_eye.view.y_scale_linear_button.isChecked()
        assert all([p.get_scale('y_alt') == 'log'
                    for p in loaded_eye.view.figure.panes])

    def test_change_fields(self, loaded_eye, qtbot):
        selectors = {'x': [loaded_eye.view.x_property_selector, 'wavepos'],
                     'y': [loaded_eye.view.y_property_selector,
                           'spectral_flux']}
        for axis, prop_selector in selectors.items():
            selector = prop_selector[0]
            assert selector.currentText() == prop_selector[1]
            last_index = selector.count() - 1
            last_field = selector.itemText(last_index)
            selector.setCurrentIndex(last_index)
            loaded_eye.signals.axis_field_changed.emit()
            qtbot.wait(1000)
            assert all([p.get_field(axis) == last_field
                        for p in loaded_eye.view.figure.panes])

    def test_change_fields_with_alt(self, loaded_eye, qtbot):
        with qtbot.wait_signal(loaded_eye.view.signals.atrophy_bg_full):
            qtbot.mouseClick(loaded_eye.view.enable_overplot_checkbox,
                             PyQt5.QtCore.Qt.LeftButton)
        loaded_eye.view.axes_selector.setCurrentText('Current Overplot')

        selectors = {'x': [loaded_eye.view.x_property_selector, 'wavepos'],
                     'y_alt': [loaded_eye.view.y_property_selector,
                               'transmission']}
        for axis, prop_selector in selectors.items():
            selector = prop_selector[0]
            assert selector.currentText() == prop_selector[1]
            last_index = selector.count() - 1
            last_field = selector.itemText(last_index)
            selector.setCurrentIndex(last_index)
            loaded_eye.signals.axis_field_changed.emit()
            qtbot.wait(1000)
            assert all([p.get_field(axis) == last_field
                        for p in loaded_eye.view.figure.panes])

    def test_change_units(self, mocker, qtbot, qapp, spectral_filenames,
                          caplog):
        mocker.patch.object(PyQt5.QtWidgets.QMainWindow, 'show',
                            return_value=None)
        mocker.patch.object(PyQt5.QtWidgets, 'QApplication',
                            return_value=qapp)
        mocker.patch.object(PyQt5.QtWidgets.QFileDialog,
                            'getOpenFileNames',
                            return_value=[spectral_filenames])

        app = eye.Eye()
        qtbot.mouseClick(app.view.add_file_button, PyQt5.QtCore.Qt.LeftButton)
        app.view.refresh_controls()
        app.view.file_table_widget.selectRow(0)
        mocker.patch.object(app.view.file_table_widget, 'hasFocus',
                            return_value=True)
        qtbot.keyClick(app.view.file_table_widget,
                       PyQt5.QtCore.Qt.Key_Return)
        app.view.refresh_controls()

        selectors = {'x': [app.view.x_unit_selector, 'um'],
                     'y': [app.view.y_unit_selector, 'Jy']}
        for axis, prop_selector in selectors.items():
            selector = prop_selector[0]
            assert selector.currentText() == prop_selector[1]
            last_index = selector.count() - 1
            last_field = selector.itemText(last_index)
            selector.setCurrentIndex(last_index)
            app.signals.axis_unit_changed.emit()
            qtbot.wait(1000)

            assert all([p.get_unit(axis) == last_field
                        for p in app.view.figure.panes])

    def test_change_units_2(self, loaded_eye, qtbot):
        selectors = {'x': [loaded_eye.view.x_unit_selector, 'um'],
                     'y': [loaded_eye.view.y_unit_selector, 'Jy']}
        for axis, prop_selector in selectors.items():
            selector = prop_selector[0]
            assert selector.currentText() == prop_selector[1]
            last_index = selector.count() - 1
            last_field = selector.itemText(last_index)
            selector.setCurrentIndex(last_index)
            loaded_eye.signals.axis_unit_changed.emit()
            qtbot.wait(1000)

            assert all([p.get_unit(axis) == last_field
                        for p in loaded_eye.view.figure.panes])

    def test_change_units_with_alt(self, loaded_eye_with_alt, qtbot):
        loaded_eye = loaded_eye_with_alt
        selectors = {'y_alt': [loaded_eye.view.y_unit_selector, 'Jy']}
        for axis, prop_selector in selectors.items():
            selector = prop_selector[0]
            assert selector.currentText() == prop_selector[1]
            last_index = selector.count() - 1
            selector.setCurrentIndex(last_index)
            loaded_eye.signals.axis_unit_changed.emit()
            qtbot.wait(1000)

            assert all([p.get_unit(axis) == ''
                        for p in loaded_eye.view.figure.panes])

    def test_change_limits(self, loaded_eye, qtbot):
        assert loaded_eye.view.x_limit_min.text() == '4.805'
        assert loaded_eye.view.x_limit_max.text() == '7.995'

        limits = [5, 10]
        widgets = [(loaded_eye.view.x_limit_min, str(limits[0])),
                   (loaded_eye.view.x_limit_max, str(limits[1])),
                   (loaded_eye.view.y_limit_min, str(limits[0])),
                   (loaded_eye.view.y_limit_max, str(limits[1]))]
        signals = [loaded_eye.signals.axis_limits_changed,
                   loaded_eye.signals.atrophy_bg_partial]
        qtbot.wait(2000)
        for widget, value in widgets:
            with qtbot.wait_signals(signals):
                qtbot.mouseClick(widget, PyQt5.QtCore.Qt.LeftButton)
                widget.setText(value)
                widget.editingFinished.emit()
        qtbot.wait(2000)
        qtbot.wait_until(loaded_eye.view.timer.isActive)
        ax = loaded_eye.view.figure.panes[0].ax
        npt.assert_array_equal(ax.get_xlim(), limits)
        npt.assert_array_equal(ax.get_ylim(), limits)

    @pytest.mark.parametrize(
        'key,cid_name,deltax,deltay,changed,same,guide_count',
        [(PyQt5.QtCore.Qt.Key_X, 'x_zoom', [1 / 4, 3 / 4], [1 / 2, 1 / 2],
          [0], [1], 1),
         (PyQt5.QtCore.Qt.Key_Y, 'y_zoom', [1 / 2, 1 / 2], [1 / 4, 3 / 4],
          [1], [0], 1),
         (PyQt5.QtCore.Qt.Key_Z, 'b_zoom', [1 / 4, 3 / 4], [1 / 4, 3 / 4],
          [0, 1], [], 2)])
    def test_zoom(self, loaded_eye, qtbot, caplog,
                  key, cid_name, deltax, deltay, changed, same, guide_count):
        caplog.set_level(logging.INFO)
        qtbot.wait(200)
        starting_limits = (loaded_eye.view.figure.panes[0].ax.get_xlim(),
                           loaded_eye.view.figure.panes[0].ax.get_ylim())
        with qtbot.wait_signal(loaded_eye.signals.atrophy):
            qtbot.keyClick(loaded_eye.view.figure_widget, key)

        cids = ['zoom_crosshair', cid_name]
        for cid in cids:
            assert cid in loaded_eye.view.cid.keys()
            assert isinstance(loaded_eye.view.cid[cid], int)
        assert f'Starting {cid_name} mode' in caplog.text

        p = loaded_eye.view.figure_widget.canvas.pos()
        w = loaded_eye.view.figure_widget.canvas.width()
        h = loaded_eye.view.figure_widget.canvas.height()

        guide_counts = [guide_count, 0]
        for i, count in enumerate(guide_counts):
            point = PyQt5.QtCore.QPoint(int(p.x() + deltax[i] * w),
                                        int(p.y() + deltay[i] * h))

            with qtbot.wait_signal(loaded_eye.signals.atrophy):
                qtbot.mouseClick(loaded_eye.view.figure_widget.canvas,
                                 PyQt5.QtCore.Qt.LeftButton,
                                 pos=point)

            qtbot.wait(1000)
            assert len(loaded_eye.view.figure.artists.arts['guide']) == count

        new_limits = (loaded_eye.view.figure.panes[0].ax.get_xlim(),
                      loaded_eye.view.figure.panes[0].ax.get_ylim())

        for index in changed:
            assert new_limits[index][0] > starting_limits[index][0]
            assert new_limits[index][1] < starting_limits[index][1]
        for index in same:
            npt.assert_array_equal(new_limits[index], starting_limits[index])

        with qtbot.wait_signal(loaded_eye.signals.atrophy_bg_partial):
            qtbot.mouseClick(loaded_eye.view.reset_zoom_button,
                             PyQt5.QtCore.Qt.LeftButton)

        qtbot.wait(100)
        new_limits = (loaded_eye.view.figure.panes[0].ax.get_xlim(),
                      loaded_eye.view.figure.panes[0].ax.get_ylim())
        for new, start in zip(new_limits, starting_limits):
            npt.assert_array_equal(new, start)
        assert 'Zoom reset' in caplog.text

    @pytest.mark.parametrize(
        'key,cid_name,deltax,deltay,changed,same,guide_count',
        [(PyQt5.QtCore.Qt.Key_Y, 'y_zoom', [1 / 2, 1 / 2], [1 / 4, 3 / 4],
          [1], [0], 1)])
    def test_zoom_with_alt(self, loaded_eye_with_alt, qtbot, caplog,
                           key, cid_name, deltax, deltay, changed,
                           same, guide_count):
        loaded_eye = loaded_eye_with_alt
        caplog.set_level(logging.INFO)
        qtbot.wait(200)
        starting_limits = (loaded_eye.view.figure.panes[0].ax.get_xlim(),
                           loaded_eye.view.figure.panes[0].ax.get_ylim())
        with qtbot.wait_signal(loaded_eye.signals.atrophy):
            qtbot.keyClick(loaded_eye.view.figure_widget, key)

        cids = ['zoom_crosshair', cid_name]
        for cid in cids:
            assert cid in loaded_eye.view.cid.keys()
            assert isinstance(loaded_eye.view.cid[cid], int)
        assert f'Starting {cid_name} mode' in caplog.text

        p = loaded_eye.view.figure_widget.canvas.pos()
        w = loaded_eye.view.figure_widget.canvas.width()
        h = loaded_eye.view.figure_widget.canvas.height()

        guide_counts = [guide_count, 0]
        for i, count in enumerate(guide_counts):
            point = PyQt5.QtCore.QPoint(int(p.x() + deltax[i] * w),
                                        int(p.y() + deltay[i] * h))

            with qtbot.wait_signal(loaded_eye.signals.atrophy):
                qtbot.mouseClick(loaded_eye.view.figure_widget.canvas,
                                 PyQt5.QtCore.Qt.LeftButton,
                                 pos=point)

            qtbot.wait(1000)
            assert len(loaded_eye.view.figure.artists.arts['guide']) == count

        new_limits = (loaded_eye.view.figure.panes[0].ax.get_xlim(),
                      loaded_eye.view.figure.panes[0].ax.get_ylim())

        for index in changed:
            assert new_limits[index][0] > starting_limits[index][0]
            assert new_limits[index][1] < starting_limits[index][1]
        for index in same:
            npt.assert_array_equal(new_limits[index], starting_limits[index])

        with qtbot.wait_signal(loaded_eye.signals.atrophy_bg_partial):
            qtbot.mouseClick(loaded_eye.view.reset_zoom_button,
                             PyQt5.QtCore.Qt.LeftButton)

        qtbot.wait(100)
        new_limits = (loaded_eye.view.figure.panes[0].ax.get_xlim(),
                      loaded_eye.view.figure.panes[0].ax.get_ylim())
        for new, start in zip(new_limits, starting_limits):
            npt.assert_array_equal(new, start)
        assert 'Zoom reset' in caplog.text

    @pytest.mark.parametrize('feature,baseline,mode',
                             [('Gaussian', 'Linear', 'fit_gauss_linear'),
                              ('Gaussian', 'Constant', 'fit_gauss_constant'),
                              ('Gaussian', '-', 'fit_gauss_-'),
                              ('Moffat', 'Linear', 'fit_moffat_linear'),
                              ('Moffat', 'Constant', 'fit_moffat_constant'),
                              ('Moffat', '-', 'fit_moffat_-'),
                              ('-', 'Linear', 'fit_-_linear'),
                              ('-', 'Constant', 'fit_-_constant'),
                              ('-', '-', 'fit_-_-')])
    def test_fit(self, loaded_eye, qtbot, caplog, mocker,
                 feature, baseline, mode):
        mocker.patch.object(fitting_results.FittingResults,
                            'show')
        caplog.set_level(logging.INFO)

        loaded_eye.view.feature_model_selection.setCurrentText(feature)
        loaded_eye.view.background_model_selection.setCurrentText(baseline)
        qtbot.keyClick(loaded_eye.view.figure_widget, PyQt5.QtCore.Qt.Key_F)

        cids = ['zoom_crosshair', mode]
        for cid in cids:
            assert cid in loaded_eye.view.cid.keys()
            assert isinstance(loaded_eye.view.cid[cid], int)
        assert f'Starting {mode} mode' in caplog.text

        p = loaded_eye.view.figure_widget.canvas.pos()
        w = loaded_eye.view.figure_widget.canvas.width()
        h = loaded_eye.view.figure_widget.canvas.height()

        deltax = [1 / 4, 3 / 4]
        deltay = [1 / 2, 1 / 2]
        assert len(loaded_eye.view.figure.artists.arts['fit']) == 0
        for x, y in zip(deltax, deltay):
            point = PyQt5.QtCore.QPoint(int(p.x() + x * w),
                                        int(p.y() + y * h))

            with qtbot.wait_signal(loaded_eye.signals.atrophy):
                qtbot.mouseClick(loaded_eye.view.figure_widget.canvas,
                                 PyQt5.QtCore.Qt.LeftButton,
                                 pos=point)

        qtbot.wait(1000)
        assert len(loaded_eye.view.figure.artists.arts['fit']) == 2
        assert len(loaded_eye.view.fit_results.model_fits) == 1

        with qtbot.wait_signal(loaded_eye.signals.atrophy):
            qtbot.keyClick(loaded_eye.view.figure_widget,
                           PyQt5.QtCore.Qt.Key_C)
        assert len(loaded_eye.view.figure.artists.arts['fit']) == 0

    def test_cursor_location(self, loaded_eye, qtbot, mocker):
        mocker.patch.object(cursor_location.CursorLocation, 'show')
        window_mock = mocker.patch.object(cursor_location.CursorLocation,
                                          'update_points')
        event = bb.MouseEvent(name='motion_notify_event',
                              canvas=loaded_eye.view.figure_widget.canvas,
                              x=1200, y=1200, button=None)

        loc_labels = [loaded_eye.view.cursor_x_label,
                      loaded_eye.view.cursor_y_label,
                      loaded_eye.view.cursor_wave_label,
                      loaded_eye.view.cursor_flux_label,
                      loaded_eye.view.cursor_column_label]

        assert loaded_eye.view.cursor_location_window is None
        assert not loaded_eye.view.cursor_checkbox.isChecked()
        assert all([label.text().strip() == '-' for label in loc_labels])

        loaded_eye.view.figure_widget.canvas.callbacks.process(
            'motion_notify_event', event)
        qtbot.wait(100)

        assert all([label.text().strip() == '-' for label in loc_labels])
        assert all([not art['artist'].get_visible()
                    for art in loaded_eye.view.figure.artists.arts['cursor']])

        # Turn on checkbox
        qtbot.mouseClick(loaded_eye.view.cursor_checkbox,
                         PyQt5.QtCore.Qt.LeftButton)
        qtbot.wait(1000)
        cids = ['cursor_loc', 'cursor_axis_leave']
        for cid in cids:
            assert cid in loaded_eye.view.cid.keys()
            assert isinstance(loaded_eye.view.cid[cid], int)

        # Wiggle mouse
        loaded_eye.view.figure_widget.canvas.callbacks.process(
            'motion_notify_event', event)
        qtbot.wait(100)
        assert all([label.text().strip() != '-' for label in loc_labels])
        arts = loaded_eye.view.figure.artists.arts['cursor']
        assert all([art['artist'].get_visible() for art in arts])

        # Turn off checkbox
        qtbot.mouseClick(loaded_eye.view.cursor_checkbox,
                         PyQt5.QtCore.Qt.LeftButton)
        qtbot.wait(200)
        assert all([label.text().strip() == '-' for label in loc_labels])
        assert all([not art['artist'].get_visible() for art in arts])

        # Pop out window
        qtbot.mouseClick(loaded_eye.view.cursor_popout_button,
                         PyQt5.QtCore.Qt.LeftButton)
        qtbot.wait(200)
        assert loaded_eye.view._cursor_popout

        # Wiggle mouse
        loaded_eye.view.figure_widget.canvas.callbacks.process(
            'motion_notify_event', event)
        qtbot.wait(100)
        assert all([label.text().strip() == '-' for label in loc_labels])
        assert all([art['artist'].get_visible() for art in arts])
        assert window_mock.called_once()

    @pytest.mark.parametrize('index,label,color',
                             [(0, 'Accessible', '#2848ad'),
                              (1, 'Spectral', '#66c2a5'),
                              (2, 'Tableau', '#1f77b4')])
    def test_color_cycle(self, loaded_eye, qtbot, index, label, color):

        cycle = loaded_eye.view.color_cycle_selector.currentText()
        assert cycle == 'Accessible'

        if index:
            line = loaded_eye.view.figure.artists.arts['line'][0]['artist']
            assert line.get_color() != color
            with qtbot.wait_signal(loaded_eye.view.signals.atrophy):
                loaded_eye.view.color_cycle_selector.setCurrentIndex(index)

        qtbot.wait(200)
        cycle = loaded_eye.view.color_cycle_selector.currentText()
        assert cycle == label

        line = loaded_eye.view.figure.artists.arts['line'][0]['artist']
        assert line.get_color() == color

    @pytest.mark.parametrize('index,label,drawstyle,linestyle,marker',
                             [(0, 'Step', 'steps-mid', '-', 'None'),
                              (1, 'Line', 'default', '-', 'None'),
                              (2, 'Scatter', 'default', 'None', 'x')])
    def test_plot_type_cycle(self, loaded_eye, qtbot, index, label,
                             drawstyle, linestyle, marker):
        current = loaded_eye.view.plot_type_selector.currentText()
        assert current == 'Step'

        if index:
            with qtbot.wait_signal(loaded_eye.view.signals.atrophy):
                loaded_eye.view.plot_type_selector.setCurrentIndex(index)

        qtbot.wait(200)

        current = loaded_eye.view.plot_type_selector.currentText()
        assert current == label

        line = loaded_eye.view.figure.artists.arts['line'][0]['artist']
        assert line.get_drawstyle() == drawstyle
        assert line.get_linestyle() == linestyle
        assert str(line.get_marker()) == marker

    def test_show_markers(self, loaded_eye, qtbot):
        assert not loaded_eye.view.marker_checkbox.isChecked()
        line = loaded_eye.view.figure.artists.arts['line'][0]['artist']
        assert str(line.get_marker()) == 'None'

        with qtbot.wait_signal(loaded_eye.view.signals.atrophy):
            qtbot.mouseClick(loaded_eye.view.marker_checkbox,
                             PyQt5.QtCore.Qt.LeftButton)
        assert str(line.get_marker()) == 'x'

        with qtbot.wait_signal(loaded_eye.view.signals.atrophy):
            qtbot.mouseClick(loaded_eye.view.marker_checkbox,
                             PyQt5.QtCore.Qt.LeftButton)
        assert str(line.get_marker()) == 'None'

    def test_show_grid(self, loaded_eye, qtbot):
        assert not loaded_eye.view.grid_checkbox.isChecked()
        ax = loaded_eye.view.figure.panes[0].ax
        assert not ax.xaxis._major_tick_kw['gridOn']
        assert not ax.yaxis._major_tick_kw['gridOn']
        assert not loaded_eye.view.figure.show_grid

        with qtbot.wait_signal(loaded_eye.view.signals.atrophy_bg_partial):
            qtbot.mouseClick(loaded_eye.view.grid_checkbox,
                             PyQt5.QtCore.Qt.LeftButton)
        assert ax.xaxis._major_tick_kw['gridOn']
        assert ax.yaxis._major_tick_kw['gridOn']
        assert loaded_eye.view.figure.show_grid

        with qtbot.wait_signal(loaded_eye.view.signals.atrophy_bg_partial):
            qtbot.mouseClick(loaded_eye.view.grid_checkbox,
                             PyQt5.QtCore.Qt.LeftButton)
        assert not ax.xaxis._major_tick_kw['gridOn']
        assert not ax.yaxis._major_tick_kw['gridOn']
        assert not loaded_eye.view.figure.show_grid

    def test_show_errors(self, loaded_eye, qtbot):
        assert loaded_eye.view.error_checkbox.isChecked()
        art = loaded_eye.view.figure.artists.arts['error_range'][0]['artist']
        assert art.get_visible()

        with qtbot.wait_signal(loaded_eye.view.signals.atrophy):
            qtbot.mouseClick(loaded_eye.view.error_checkbox,
                             PyQt5.QtCore.Qt.LeftButton)
        assert not loaded_eye.view.error_checkbox.isChecked()
        assert not art.get_visible()

        with qtbot.wait_signal(loaded_eye.view.signals.atrophy):
            qtbot.mouseClick(loaded_eye.view.error_checkbox,
                             PyQt5.QtCore.Qt.LeftButton)
        assert loaded_eye.view.error_checkbox.isChecked()
        assert art.get_visible()

    def test_dark_mode(self, loaded_eye, qtbot):
        white = (1, 1, 1, 1)
        black = (0, 0, 0, 1)

        assert not loaded_eye.view.dark_mode_checkbox.isChecked()
        assert loaded_eye.view.figure.fig.get_facecolor() == white

        with qtbot.wait_signal(loaded_eye.view.signals.atrophy_bg_partial):
            qtbot.mouseClick(loaded_eye.view.dark_mode_checkbox,
                             PyQt5.QtCore.Qt.LeftButton)
        assert loaded_eye.view.figure.fig.get_facecolor() == black

        with qtbot.wait_signal(loaded_eye.view.signals.atrophy_bg_partial):
            qtbot.mouseClick(loaded_eye.view.dark_mode_checkbox,
                             PyQt5.QtCore.Qt.LeftButton)
        assert loaded_eye.view.figure.fig.get_facecolor() == white

    def test_show_overplot(self, loaded_eye, qtbot):
        assert not loaded_eye.view.enable_overplot_checkbox.isChecked()
        assert len(loaded_eye.view.figure.artists.arts['line_alt']) == 0

        with qtbot.wait_signal(loaded_eye.view.signals.atrophy_bg_full):
            qtbot.mouseClick(loaded_eye.view.enable_overplot_checkbox,
                             PyQt5.QtCore.Qt.LeftButton)
        assert loaded_eye.view.enable_overplot_checkbox.isChecked()
        assert len(loaded_eye.view.figure.artists.arts['line_alt']) == 1
        art = loaded_eye.view.figure.artists.arts['line_alt'][0]['artist']
        assert art.get_visible()

        with qtbot.wait_signal(loaded_eye.view.signals.atrophy_bg_full):
            qtbot.mouseClick(loaded_eye.view.enable_overplot_checkbox,
                             PyQt5.QtCore.Qt.LeftButton)
        assert loaded_eye.view.error_checkbox.isChecked()
        assert len(loaded_eye.view.figure.artists.arts['line_alt']) == 0

    def test_fit_results_window(self, loaded_eye, qtbot, mocker):
        show_mock = mocker.patch.object(fitting_results.FittingResults,
                                        'show')
        vis_mock = mocker.patch.object(fitting_results.FittingResults,
                                       'isVisible', return_value=True)

        # pop out new fit results window
        assert loaded_eye.view.fit_results is None
        qtbot.mouseClick(loaded_eye.view.open_fit_results_button,
                         PyQt5.QtCore.Qt.LeftButton)
        qtbot.wait(200)
        assert loaded_eye.view.fit_results is not None
        assert show_mock.call_count == 1
        assert vis_mock.call_count == 0

        # click again, while visible - does nothing
        qtbot.mouseClick(loaded_eye.view.open_fit_results_button,
                         PyQt5.QtCore.Qt.LeftButton)
        qtbot.wait(200)
        assert loaded_eye.view.fit_results is not None
        assert show_mock.call_count == 1
        assert vis_mock.call_count == 1

        # click again, while hidden - shows again
        vis_mock = mocker.patch.object(fitting_results.FittingResults,
                                       'isVisible', return_value=False)
        qtbot.mouseClick(loaded_eye.view.open_fit_results_button,
                         PyQt5.QtCore.Qt.LeftButton)
        qtbot.wait(200)
        assert loaded_eye.view.fit_results is not None
        assert show_mock.call_count == 2
        assert vis_mock.call_count == 1
