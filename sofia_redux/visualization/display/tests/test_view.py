# Licensed under a 3-clause BSD style license - see LICENSE.rst

import logging
import pytest
import matplotlib.backend_bases as mbb
from sofia_redux.visualization.display import (view, figure, cursor_location,
                                               pane, fitting_results,
                                               reference_window)
from sofia_redux.visualization.utils import eye_error
from sofia_redux.visualization import signals, log
from sofia_redux.visualization.models import high_model
from sofia_redux.visualization.display.ui.mplwidget import MplWidget

try:
    from PyQt5 import QtWidgets, QtCore, QtGui
except ImportError:
    HAS_PYQT5 = False
    QtWidgets, QtGui = None, None

    class QtCore:
        class Qt:
            Key_X = None
            Key_Right = None
            Key_Escape = None
            Key_Home = None
            Key_Space = None
else:
    HAS_PYQT5 = True


@pytest.mark.skipif("not HAS_PYQT5")
class TestView(object):

    @pytest.mark.parametrize('key,name',
                             [(QtCore.Qt.Key_X, 'x'),
                              (QtCore.Qt.Key_A, 'a'),
                              (QtCore.Qt.Key_W, 'w'),
                              (QtCore.Qt.Key_Right, 'right'),
                              (QtCore.Qt.Key_Escape, 'esc'),
                              (QtCore.Qt.Key_Home, 'home'),
                              (QtCore.Qt.Key_Delete, 'del'),
                              (QtCore.Qt.Key_Space, 'space')])
    def test_key_press_event(self, key, name, qtbot, capsys, mocker, qapp):
        mocker.patch.object(QtWidgets.QMainWindow, 'show',
                            return_value=None)
        mocker.patch.object(QtWidgets, 'QApplication',
                            return_value=qapp)
        v = view.View(signals.Signals())
        qtbot.add_widget(v)

        qtbot.keyClick(v, key)

        log.setLevel('DEBUG')
        captured = capsys.readouterr()
        assert 'Key pushed in view' in captured.out
        assert name in captured.out.lower()

    def test_key_press_event_delete_table(self, blank_view, mocker,
                                          qtbot, caplog):
        caplog.set_level(logging.DEBUG)

        mocker.patch.object(QtWidgets.QTableWidget, 'hasFocus',
                            return_value=True)
        with qtbot.wait_signal(blank_view.signals.model_removed):
            qtbot.keyClick(blank_view, QtCore.Qt.Key_Delete)
        assert 'Key pushed in view: Del' in caplog.text

    def test_key_press_event_delete_pane(self, blank_view, mocker,
                                         qtbot, caplog):
        caplog.set_level(logging.DEBUG)

        mocker.patch.object(MplWidget, 'hasFocus', return_value=True)
        mock = mocker.patch.object(view.View, 'remove_pane')

        qtbot.keyClick(blank_view, QtCore.Qt.Key_Delete)

        assert 'Key pushed in view: Del' in caplog.text
        assert mock.call_count == 1

    def test_key_press_event_delete_file(self, blank_view, mocker, qtbot,
                                         caplog):
        caplog.set_level(logging.DEBUG)
        mocker.patch.object(QtWidgets.QTableWidget, 'hasFocus',
                            side_effect=[False, True])
        mock = mocker.patch.object(view.View, 'remove_file_from_pane')
        qtbot.keyClick(blank_view, QtCore.Qt.Key_Backspace)
        assert 'Key pushed in view: Backspace' in caplog.text
        assert mock.call_count == 1

    def test_atrophy_controls(self, blank_view, mocker, qtbot, caplog):
        caplog.set_level(logging.DEBUG)
        clear_mock = mocker.patch.object(view.View, 'clear_selection')
        update_mock = mocker.patch.object(view.View, 'update_controls')

        blank_view.atrophy_controls()

        assert clear_mock.call_count == 1
        assert update_mock.call_count == 1
        assert 'Controls no longer in sync' in caplog.text

    def test_hold_atrophy(self, blank_view, mocker):
        blank_view.signals.atrophy.connect(blank_view.atrophy)
        blank_view.signals.atrophy_bg_full.connect(
            blank_view.atrophy_background_full)
        blank_view.signals.atrophy_bg_partial.connect(
            blank_view.atrophy_background_partial)

        mock = mocker.patch.object(QtCore.pyqtBoundSignal, 'disconnect')

        sigs = [blank_view.signals.atrophy,
                blank_view.signals.atrophy_bg_full,
                blank_view.signals.atrophy_bg_partial]
        assert all([isinstance(signal, QtCore.pyqtBoundSignal)
                    for signal in sigs])
        blank_view.hold_atrophy()
        assert mock.call_count == 3

    def test_release_atrophy(self, blank_view, mocker):
        mock = mocker.patch.object(QtCore.pyqtBoundSignal, 'connect')
        blank_view.release_atrophy()
        assert mock.call_count == 3

    def test_axis_limits_changed_fail(self, blank_view, caplog, mocker,
                                      qtbot):
        mocker.patch.object(view.View, '_pull_limits_from_gui',
                            side_effect=ValueError)
        with qtbot.wait_signal(blank_view.signals.atrophy_bg_partial):
            blank_view.axis_limits_changed()

        assert 'Illegal limits entered' in caplog.text

    def test_axis_limits_changed_all(self, blank_view, mocker, qtbot):
        mocker.patch.object(view.View, 'selected_target_axis',
                            return_value={'axis': 'all'})
        mocker.patch.object(blank_view.x_limit_min, 'text',
                            return_value='0')
        mocker.patch.object(blank_view.x_limit_max, 'text',
                            return_value='1')
        mocker.patch.object(blank_view.y_limit_min, 'text',
                            return_value='2')
        mocker.patch.object(blank_view.y_limit_max, 'text',
                            return_value='3')
        mocker.patch.object(QtWidgets.QRadioButton, 'isChecked',
                            return_value=True)
        limit_mock = mocker.patch.object(figure.Figure,
                                         'change_axis_limits')

        with qtbot.wait_signal(blank_view.signals.atrophy_bg_partial):
            blank_view.axis_limits_changed()

        assert limit_mock.called_with({'panes': 'all'})

    def test_axis_scale_changed_all(self, blank_view, mocker, qtbot):
        mocker.patch.object(view.View, '_pull_scale_from_gui',
                            return_value='log')
        mocker.patch.object(QtWidgets.QRadioButton, 'isChecked',
                            return_value=True)
        scale_mock = mocker.patch.object(figure.Figure, 'set_scales')

        with qtbot.wait_signal(blank_view.signals.atrophy_bg_partial):
            blank_view.axis_scale_changed()

        assert scale_mock.called_with({'panes': 'all'})

    def test_axis_unit_changed_all(self, blank_view, mocker, qtbot):
        mocker.patch.object(view.View, '_pull_units_from_gui',
                            return_value='Jy')
        mocker.patch.object(QtWidgets.QRadioButton, 'isChecked',
                            return_value=True)
        unit_mock = mocker.patch.object(figure.Figure, 'change_axis_unit')

        with qtbot.wait_signal(blank_view.signals.atrophy_bg_full):
            blank_view.axis_unit_changed()

        assert unit_mock.called_with({'panes': 'all'})

    def test_axis_field_changed_all(self, blank_view, mocker, qtbot, caplog):
        mocker.patch.object(view.View, '_pull_fields_from_gui',
                            return_value='flux')
        mocker.patch.object(QtWidgets.QRadioButton, 'isChecked',
                            return_value=True)
        field_mock = mocker.patch.object(figure.Figure, 'change_axis_field')
        clear_mock = mocker.patch.object(view.View, 'clear_selection')

        with qtbot.wait_signal(blank_view.signals.atrophy_bg_full):
            blank_view.axis_field_changed()

        assert field_mock.called_with({'panes': 'all'})
        assert clear_mock.call_count == 1

    def test_update_cursor_loc_labels_blank(self, blank_view):
        data_coords = dict()
        cursor_coords = [1, 1]
        labels = [blank_view.cursor_wave_label,
                  blank_view.cursor_flux_label,
                  blank_view.cursor_column_label]
        for label in labels:
            label.setText('dummy')

        blank_view._update_cursor_loc_labels(data_coords, cursor_coords)

        for label in labels:
            assert label.text() == '-'

    def test_closed_cursor_popout(self, blank_view, mocker, qtbot):
        blank_view._cursor_popout = True
        blank_view.cursor_checkbox.setChecked(True)
        popout = mocker.Mock(spec_set=cursor_location.CursorLocation)
        blank_view.cursor_location_window = popout
        cid_mock = mocker.patch.object(view.View, 'clear_cids')

        with qtbot.wait_signal(blank_view.signals.atrophy):
            blank_view.closed_cursor_popout()

        assert not blank_view._cursor_popout
        assert not blank_view.cursor_checkbox.isChecked()
        assert blank_view.cursor_location_window is None
        assert cid_mock.called_with('cursor')

    @pytest.mark.parametrize('x_scale,y_scale,x_log,x_lin,y_log,y_lin',
                             [('lin', 'lin', False, True, False, True),
                              ('log', 'log', True, False, True, False),
                              ('log', 'loglin', True, False, False, True)])
    def test_setup_initial_scales(self, blank_view, mocker, x_scale, y_scale,
                                  x_log, x_lin, y_log, y_lin):
        pane_mock = mocker.Mock(spec_set=pane.OneDimPane)
        pane_mock.get_axis_scale.return_value = {'x': x_scale, 'y': y_scale}
        mocker.patch.object(figure.Figure, 'get_current_pane',
                            return_value=[pane_mock, pane_mock])

        # invalid values are ignored
        if x_lin:
            blank_view.setup_initial_scales()
        else:
            blank_view.setup_initial_scales(pane_mock)

        assert blank_view.x_scale_linear_button.isChecked() is x_lin
        assert blank_view.x_scale_log_button.isChecked() is x_log
        assert blank_view.y_scale_linear_button.isChecked() is y_lin
        assert blank_view.y_scale_log_button.isChecked() is y_log

    def test_setup_initial_scales_inconsistent(self, blank_view, mocker,
                                               one_dim_pane):
        mocker.patch.object(pane.OneDimPane, 'get_axis_scale',
                            side_effect=[{'x': 'linear', 'y': 'linear'},
                                         {'x': 'linear', 'y': 'log'}])
        blank_view.setup_initial_scales(panes=[one_dim_pane, one_dim_pane])
        assert blank_view.x_scale_linear_button.isChecked()
        assert not blank_view.x_scale_log_button.isChecked()
        assert not blank_view.y_scale_linear_button.isChecked()
        assert not blank_view.y_scale_log_button.isChecked()

    def test_setup_property_selectors(self, blank_view, grism_hdul):
        # no current pane, nothing is done
        blank_view.setup_property_selectors()
        assert blank_view.x_property_selector.currentText() == ''
        sigs = signals.Signals()

        # set a pane, no models
        pane_ = pane.OneDimPane(sigs)
        blank_view.setup_property_selectors(pane_)
        assert blank_view.x_property_selector.currentText() == '-'
        assert blank_view.y_property_selector.currentText() == '-'

        # set a model
        model_grism = high_model.Grism(grism_hdul)
        pane_.models = {1: model_grism}
        pane_.plot_kind = 'spectrum'
        blank_view.setup_property_selectors(pane_)
        assert blank_view.x_property_selector.currentText() == 'Wavelength'
        assert blank_view.y_property_selector.currentText() == 'Spectral Flux'

        # setup for alt
        pane_.fields['y_alt'] = 'transmission'
        blank_view.setup_property_selectors(pane_, {'axis': 'alt'})
        assert blank_view.x_property_selector.currentText() == 'Wavelength'
        assert blank_view.y_property_selector.currentText() == 'Transmission'

    def test_setup_unit_selectors(self, blank_view, grism_hdul, caplog):
        caplog.set_level(logging.DEBUG)
        # no current pane, nothing is done
        blank_view.setup_unit_selectors()
        assert blank_view.x_unit_selector.currentText() == ''

        # set a pane, no models
        sigs = signals.Signals()
        pane_ = pane.OneDimPane(sigs)
        blank_view.setup_unit_selectors(pane_)
        assert blank_view.x_unit_selector.currentText() == ''
        assert blank_view.y_unit_selector.currentText() == ''

        # set a model
        model_grism = high_model.Grism(grism_hdul)
        pane_.models = {1: model_grism}
        pane_.plot_kind = 'spectrum'
        blank_view.setup_unit_selectors(pane_)
        assert blank_view.x_unit_selector.currentText() == 'um'
        assert blank_view.y_unit_selector.currentText() == 'Jy'

        # setup for alt
        blank_view.setup_unit_selectors(pane_, {'axis': 'alt'})
        assert blank_view.x_unit_selector.currentText() == 'um'
        assert blank_view.y_unit_selector.currentText() == ''

        # handle bad call
        blank_view.setup_unit_selectors([])
        assert 'Set unit selectors to [], selected 0' in caplog.text

    def test_setup_axis_limits(self, blank_view, mocker, caplog):
        caplog.set_level(logging.DEBUG)
        # no current pane, nothing is done
        blank_view.setup_axis_limits()
        sigs = signals.Signals()
        assert blank_view.x_limit_min.text() == ''

        # set a pane, no limits available
        pane_ = pane.OneDimPane(sigs)
        blank_view.setup_axis_limits(pane_)
        assert blank_view.x_limit_min.text() == '0.000'
        assert blank_view.x_limit_max.text() == '1.000'
        assert blank_view.y_limit_min.text() == '0.000'
        assert blank_view.y_limit_max.text() == '1.000'

        # set some limits
        mocker.patch.object(pane_, 'get_axis_limits',
                            return_value={'x': [1, 2], 'y': [3, 4],
                                          'y_alt': [5, 6]})
        blank_view.setup_axis_limits(pane_)
        assert blank_view.x_limit_min.text() == '1.000'
        assert blank_view.x_limit_max.text() == '2.000'
        assert blank_view.y_limit_min.text() == '3.000'
        assert blank_view.y_limit_max.text() == '4.000'

        # setup for alt
        blank_view.setup_axis_limits(pane_, {'axis': 'alt'})
        assert blank_view.x_limit_min.text() == '1.000'
        assert blank_view.x_limit_max.text() == '2.000'
        assert blank_view.y_limit_min.text() == '5.000'
        assert blank_view.y_limit_max.text() == '6.000'

    def test_add_panes(self, blank_view, mocker):
        mocker.patch.object(figure.Figure, 'set_layout_style')
        add_mock = mocker.patch.object(figure.Figure, 'add_panes')
        kind = ['spectrum', 'image']

        blank_view.add_panes(2, kind=kind)

        assert add_mock.called_with({'n_dims': [1, 2]})

        with pytest.raises(eye_error.EyeError) as msg:
            blank_view.add_panes(2, kind=['spectrum', 'cube'])
        assert 'Valid pane kinds' in str(msg)

    def test_select_pane(self, blank_view, mocker):
        pane_mock = mocker.patch.object(view.View, 'set_current_pane')
        pane_id = 1
        item = QtWidgets.QTreeWidgetItem()
        item.setData(0, QtCore.Qt.UserRole, pane_id)

        blank_view.select_pane(item)

        assert pane_mock.called_with(pane_id)

    def test_enable_model(self, blank_view, mocker):
        pane_mock = mocker.patch.object(figure.Figure, 'set_enabled')
        pane_id = 1
        model_id = True
        state = True

        item = QtWidgets.QTreeWidgetItem()
        item.setCheckState(0, state)
        item.setData(0, QtCore.Qt.UserRole, (pane_id, model_id))

        blank_view.enable_model(item)
        assert pane_mock.call_count == 0

        model_id = 2
        item.setData(0, QtCore.Qt.UserRole, (pane_id, model_id))
        blank_view.enable_model(item)
        assert pane_mock.call_count == 1
        assert pane_mock.called_with([pane_id, model_id, state])

    @pytest.mark.parametrize('state,id_state,label',
                             [(True, False, 'Hide all'),
                              (False, True, 'Show all')])
    def test_enable_all_models(self, blank_view, mocker, qtbot,
                               state, id_state, label):
        enable_mock = mocker.patch.object(figure.Figure, 'set_all_enabled')
        pane_id = 0

        button = QtWidgets.QPushButton()
        button.setProperty('id', (pane_id, state))
        button.clicked.connect(blank_view.enable_all_models)

        qtbot.mouseClick(button, QtCore.Qt.LeftButton)

        assert enable_mock.called_with([pane_id, state])
        assert button.text() == label
        assert button.property('id') == (pane_id, id_state)

    def test_set_field(self, blank_view, qtbot):
        with qtbot.wait_signal(blank_view.signals.axis_field_changed):
            blank_view.set_field()

    def test_set_unit(self, blank_view, qtbot):
        with qtbot.wait_signal(blank_view.signals.axis_unit_changed):
            blank_view.set_unit()

    @pytest.mark.parametrize('results,fit_results,show_count,'
                             'add_count,fit_mode',
                             [(True, True, 0, 1, 'fit_gauss_linear'),
                              (True, True, 0, 1, 'fit_gauss_none'),
                              (True, True, 0, 1, 'fit_none_constant'),
                              (True, True, 0, 1, 'fit_moffat_linear'),
                              (True, False, 1, 1, 'fit_gauss_linear'),
                              (False, True, 0, 1, 'fit_gauss_linear')])
    def test_end_selection(self, blank_view, mocker, results, fit_results,
                           show_count, add_count, fit_mode):
        mocker.patch.object(figure.Figure, 'get_selection_results',
                            return_value=results)
        show_mock = mocker.patch.object(QtWidgets.QDialog, 'show')
        clear_mock = mocker.patch.object(view.View, 'clear_selection')
        blank_view.cid[fit_mode] = None
        if fit_results:
            fit_app = mocker.Mock(spec_set=fitting_results.FittingResults)
            fit_app.isVisible.return_value = False
            blank_view.fit_results = fit_app
            add_mock = mocker.patch.object(fit_app, 'add_results')
        else:
            add_mock = mocker.patch.object(fitting_results.FittingResults,
                                           'add_results')

        blank_view.end_selection()

        assert show_mock.call_count == show_count
        assert add_mock.call_count == add_count
        assert clear_mock.call_count == 1

    @pytest.mark.parametrize('x_check,y_check,x_out,y_out',
                             [(True, True, 'linear', 'linear'),
                              (False, False, 'log', 'log')])
    def test_pull_scale_from_gui(self, blank_view, x_check, y_check,
                                 x_out, y_out, qtbot):
        if x_check:
            blank_view.x_scale_linear_button.setChecked(True)
        else:
            blank_view.x_scale_log_button.setChecked(True)
        if y_check:
            blank_view.y_scale_linear_button.setChecked(True)
        else:
            blank_view.y_scale_log_button.setChecked(True)
        blank_view.y_scale_linear_button.setChecked(y_check)
        output = {'x': x_out, 'y': y_out}

        scale = blank_view._pull_scale_from_gui()

        assert scale == output

    def test_setup_axis_limits_bad_limits(self, blank_view, mocker):
        mocker.patch.object(pane.OneDimPane, 'get_axis_limits',
                            return_value={'x': None, 'y': None})
        sigs = signals.Signals()
        mocker.patch.object(figure.Figure, 'get_current_pane',
                            return_value=[pane.OneDimPane(sigs)])

        blank_view.setup_axis_limits()

        limits = [blank_view.x_limit_min, blank_view.x_limit_max,
                  blank_view.y_limit_min, blank_view.y_limit_max]
        assert all([limit.text() == '' for limit in limits])

    def test_toggle_overplot(self, blank_view, mocker, qtbot):
        flag_mock = mocker.patch.object(figure.Figure,
                                        'set_overplot_state')
        with qtbot.wait_signal(blank_view.signals.atrophy_bg_full):
            blank_view.toggle_overplot()
        assert flag_mock.call_count == 1

    def test_setup_overplot_flag(self, blank_view, blank_pane, mocker):
        blank_view.setup_overplot_flag(blank_pane)

        mocker.patch.object(figure.Figure, 'get_current_pane',
                            return_value=[blank_pane, blank_pane])
        mocker.patch.object(blank_pane, 'overplot_state',
                            side_effect=(False, True))
        blank_view.setup_overplot_flag()
        assert not blank_view.enable_overplot_checkbox.isChecked()

    def test_remove_model_id(self, blank_view, grism_hdul):
        model = high_model.Grism(grism_hdul)

        blank_view.model_collection[model.id] = model.filename
        blank_view.remove_model_id(model.id)
        assert model.id not in blank_view.model_collection

        blank_view.remove_model_id(model.id)

    def test_select_color_cycle(self, blank_view, mocker):
        fig_mock = mocker.patch.object(figure.Figure, 'set_color_cycle',
                                       return_value=None)
        fit_mock = mocker.Mock(spec_set=fitting_results.FittingResults)
        blank_view.fit_results = fit_mock
        text = 'tab10'
        blank_view.select_color_cycle(text)
        assert fig_mock.called_with(text)
        assert fit_mock.update_colors.called_with(None)

    def test_fit_visibility(self, blank_view, mocker, qtbot):
        blank_view.fit_results = fitting_results.FittingResults(blank_view)
        flag_mock = mocker.patch.object(figure.Figure,
                                        'toggle_fits_visibility')
        gather_mock = mocker.patch.object(blank_view.fit_results,
                                          'gather_models')
        with qtbot.wait_signal(blank_view.signals.atrophy):
            blank_view.toggle_fit_visibility()
        assert flag_mock.call_count == 1
        assert gather_mock.call_count == 1

    def test_fit_unit_change(self, blank_view, mocker, qtbot):
        blank_view.fit_results = fitting_results.FittingResults(blank_view)

        # smoke test: no panes, no fits, nothing should happen
        with qtbot.wait_signal(blank_view.signals.atrophy_bg_full):
            blank_view.axis_unit_changed()

    @pytest.mark.parametrize('ax_name,pane_val,axis_val,err_val',
                             [('All', 'all', 'all', False),
                              (' Primary', 'all', 'primary', False),
                              (' Overplot', 'all', 'alt', False),
                              (' Both', 'current', 'all', False),
                              (' Primary', 'current', 'primary', False),
                              (' Overplot', 'current', 'alt', False),
                              (' bad', '', '', True)])
    def test_selected_target_axis(self, blank_view, mocker,
                                  ax_name, pane_val, axis_val, err_val):
        mocker.patch.object(blank_view.axes_selector,
                            'currentText', return_value=f'{ax_name}')
        mocker.patch.object(blank_view, '_parse_filename_table_selection',
                            return_value=(None, pane_val))
        if not err_val:
            tgt = blank_view.selected_target_axis()
            assert len(tgt) == 2
            assert tgt['pane'] == pane_val
            assert tgt['axis'] == axis_val
        else:
            with pytest.raises(eye_error.EyeError) as err:
                blank_view.selected_target_axis()
            assert 'Unknown target axis' in str(err)

    def test_open_ref_data(self, mocker, blank_view):
        m1 = mocker.patch.object(reference_window.ReferenceWindow, 'show')
        m2 = mocker.patch.object(reference_window.ReferenceWindow, 'raise_')

        # first open
        blank_view.open_ref_data()
        m1.assert_called_once()
        m2.assert_called_once()
        rw = blank_view.reference_window
        assert isinstance(rw, reference_window.ReferenceWindow)

        # second open: show, raise called but instance not replaced
        blank_view.open_ref_data()
        assert blank_view.reference_window is rw
        assert m1.call_count == 2
        assert m2.call_count == 2

    def test_update_reference_lines(self, mocker, blank_view):
        m1 = mocker.patch.object(blank_view.figure, 'update_reference_lines')
        blank_view.update_reference_lines()
        m1.assert_called_once()

    def test_unload_reference_model(self, mocker, blank_view):
        m1 = mocker.patch.object(blank_view.reference_models, 'unload_data')
        m2 = mocker.patch.object(blank_view.figure, 'unload_reference_model')
        blank_view.unload_reference_model()
        m1.assert_called_once()
        m2.assert_called_once()

    def test_on_orders_changed(self, mocker, blank_view, qtbot):
        on_orders = [1, 2, 3]
        mocker.patch.object(view.View, 'decode_orders', return_value=on_orders)
        update_mock = mocker.patch.object(view.View,
                                          '_update_orders_from_gui')

        qtbot.mouseClick(blank_view.all_disabled_orders_button,
                         QtCore.Qt.LeftButton)
        assert update_mock.called_with((on_orders, True))

        qtbot.mouseClick(blank_view.all_disabled_orders_button,
                         QtCore.Qt.LeftButton)
        assert update_mock.called_with((on_orders, False))

    def test_all_enabled_orders(self, qtbot, mocker, blank_view):

        blank_view.on_orders_selector.setText('1-3')
        blank_view.off_orders_selector.setText('4-6')
        mocker.patch.object(view.View, 'decode_orders', return_value=None)
        update_mock = mocker.patch.object(view.View, '_update_orders_from_gui')

        blank_view.enable_all_orders()

        assert update_mock.called_once
        assert blank_view.on_orders_selector.text() == '*'
        assert blank_view.off_orders_selector.text() == '-'

    def test_all_disabled_orders(self, qtbot, mocker, blank_view):

        blank_view.on_orders_selector.setText('1-3')
        blank_view.off_orders_selector.setText('4-6')
        update_mock = mocker.patch.object(view.View, '_update_orders_from_gui')
        mocker.patch.object(view.View, 'decode_orders', return_value=None)

        blank_view.disable_all_orders()

        assert blank_view.on_orders_selector.text() == '-'
        assert blank_view.off_orders_selector.text() == '*'
        assert update_mock.called_once

    def test_off_orders_changed(self, mocker, blank_view, qtbot):
        on_orders = [1, 2, 3]
        mocker.patch.object(view.View, 'decode_orders', return_value=on_orders)
        update_mock = mocker.patch.object(view.View,
                                          '_update_orders_from_gui')

        blank_view.off_orders_changed()
        qtbot.wait(1000)

        assert update_mock.called_with((on_orders, False))

    def test_all_filenames_checking(self, mocker, blank_view):
        mock = mocker.patch.object(view.View, '_update_order_selector')

        blank_view.all_filenames_checking()

        assert mock.called_once

    def test_popout_cursor_position(self, blank_view, mocker):
        show_mock = mocker.patch.object(QtWidgets.QDialog, 'show')
        raise_mock = mocker.patch.object(QtWidgets.QDialog, 'raise_')

        assert not blank_view._cursor_popout
        assert blank_view.cursor_location_window is None

        blank_view.popout_cursor_position()
        assert isinstance(blank_view.cursor_location_window,
                          cursor_location.CursorLocation)
        assert show_mock.call_count == 1
        assert raise_mock.call_count == 0

        blank_view.popout_cursor_position()
        assert show_mock.call_count == 1
        assert raise_mock.call_count == 1

    def test_figure_clicked(self, blank_view, mocker, qtbot):
        event = mbb.MouseEvent('test', mbb.FigureCanvasBase(), 0, 0)
        event.inaxes = True

        mock = mocker.patch.object(view.View, 'set_current_pane')
        mocker.patch.object(figure.Figure, 'determine_selected_pane',
                            return_value=0)

        blank_view.figure_clicked(event)
        assert mock.called_once

    def test_filename_table_selection_changed(self, blank_view, mocker):
        mock = mocker.patch.object(view.View, '_update_order_selector')
        blank_view.filename_table_selection_changed()
        assert mock.called_once
        assert not blank_view.all_filenames_checkbox.isChecked()

    def test_all_filenames_selection_changed(self, blank_view, mocker):
        order_mock = mocker.patch.object(view.View, '_update_order_selector')
        mocker.patch.object(QtWidgets.QCheckBox, 'checkState',
                            return_value=True)
        file_mock = mocker.patch.object(view.View,
                                        '_update_filename_table_selection')

        blank_view.all_filenames_selection_changed()
        assert order_mock.called_once
        assert file_mock.called_with(True)

    def test_merge_dict(self):
        one = {'a': [1], 'b': [1, 4]}
        two = {'a': [1], 'c': [2, 1]}
        merged = view.View.merge_dicts(one, two)
        assert merged == {'a': [1, 1], 'b': [1, 4], 'c': [2, 1]}

    def test_add_filename(self, blank_view, caplog, grism_hdul):
        caplog.set_level(logging.DEBUG)
        model = high_model.Grism(grism_hdul)
        assert model.id not in blank_view.model_collection
        blank_view.add_filename(model.id, model.filename)
        assert model.id in blank_view.model_collection

        assert 'already in display model list' not in caplog.text
        blank_view.add_filename(model.id, model.filename)
        assert 'already in display model list' in caplog.text

    def test_filename_order_selector_changed(self, blank_view, mocker):
        pane_mock = mocker.patch.object(view.View, '_update_pane_selector')
        order_mock = mocker.patch.object(view.View, '_update_order_selector')

        blank_view.all_filenames_checkbox.setChecked(True)
        blank_view.filename_order_selector_changed()
        assert not blank_view.all_filenames_checkbox.isChecked()
        assert pane_mock.called_once()
        assert order_mock.called_once()

    @pytest.mark.parametrize('text,count', [(1, 1), ('one', 0)])
    def test_pane_order_selector_changed(self, blank_view, mocker, text,
                                         count):
        file_mock = mocker.patch.object(view.View, '_update_filename_table')
        order_mock = mocker.patch.object(view.View, '_update_order_selector')
        pane_mock = mocker.patch.object(view.View, 'set_current_pane')
        mocker.patch.object(QtWidgets.QComboBox, 'currentText',
                            return_value=text)
        blank_view.all_panes_checkbox.setChecked(True)

        blank_view.pane_order_selector_changed()

        assert not blank_view.all_panes_checkbox.isChecked()
        assert file_mock.call_count == count
        assert order_mock.call_count == count
        assert pane_mock.call_count == count

    @pytest.mark.parametrize('checked,pane_count,args',
                             [(True, 2, [0, 1, 2]),
                              (False, 2, [0]), (False, 0, [])])
    def test_all_panes_checking(self, blank_view, mocker, qtbot,
                                checked, pane_count, args):
        pane_mock = mocker.patch.object(view.View, 'set_current_pane')
        blank_view.all_panes_checkbox.setChecked(checked)
        mocker.patch.object(figure.Figure, 'pane_count',
                            return_value=pane_count)
        with qtbot.wait_signal(blank_view.signals.current_pane_changed):
            blank_view.all_panes_checking()
        assert pane_mock.called_with(args)

    def test_remove_file_from_pane(self, blank_view, mocker):
        mocker.patch.object(view.View, '_parse_filename_table_selection',
                            return_value=('x', 'y'))
        remove_mock = mocker.patch.object(figure.Figure,
                                          'remove_model_from_pane')

        blank_view.remove_file_from_pane()

        assert remove_mock.called_with('x', 'y')

    @pytest.mark.parametrize('text,checked,count', [('none', False, 0),
                                                    ('1', False, 1),
                                                    ('3', True, 1)])
    def test_update_orders_from_gui(self, blank_view, mocker, text, checked,
                                    count, qtbot):
        mocker.patch.object(QtWidgets.QComboBox, 'currentText',
                            return_value=text)
        mocker.patch.object(QtWidgets.QCheckBox, 'isChecked',
                            return_value=checked)
        mocker.patch.object(figure.Figure, 'pane_count', return_value=count)
        fig_mock = mocker.patch.object(figure.Figure, 'set_orders')

        orders = [0, 2]
        enable = True
        if count:
            with qtbot.wait_signal(blank_view.signals.atrophy):
                blank_view._update_orders_from_gui(orders, enable)
        else:
            blank_view._update_orders_from_gui(orders, enable)

        assert fig_mock.call_count == count

    def test_update_filename_table(self, mocker, qtbot, grism_hdul, blank_view,
                                   one_dim_pane, blank_figure):
        mocker.patch.object(QtWidgets.QCheckBox, 'isChecked',
                            return_value=True)
        model = high_model.Grism(grism_hdul)
        one_dim_pane.add_model(model)
        blank_figure.panes = [one_dim_pane]
        blank_view.figure = blank_figure

        blank_view._update_filename_table()
        mocker.patch.object(QtWidgets.QCheckBox, 'isChecked',
                            return_value=True)

        assert blank_view.filename_table.rowCount() == 1

        mocker.patch.object(QtWidgets.QCheckBox, 'isChecked',
                            return_value=False)
        mocker.patch.object(QtWidgets.QTableWidget, 'currentRow',
                            return_value=0)
        mocker.patch.object(QtWidgets.QComboBox, 'currentText',
                            return_value='1')
        blank_view._update_filename_table()

    @pytest.mark.parametrize('all_panes,all_files',
                             [(False, False),
                              (False, True),
                              (True, False),
                              (True, True)])
    def test_parse_filename_table_selection(self, blank_view, mocker,
                                            blank_figure,
                                            all_panes, all_files, one_dim_pane,
                                            grism_hdul, multi_ap_grism_hdul):

        mocker.patch.object(QtWidgets.QCheckBox, 'isChecked',
                            return_value=True)
        model = high_model.Grism(grism_hdul)
        one_dim_pane.add_model(model)
        blank_figure.panes = [one_dim_pane]
        blank_view.figure = blank_figure
        blank_view._update_filename_table()
        mocker.patch.object(QtWidgets.QCheckBox, 'isChecked',
                            side_effect=(all_panes, all_files))
        mocker.patch.object(QtWidgets.QComboBox, 'currentText',
                            return_value=0)
        mocker.patch.object(figure.Figure, 'pane_count', return_value=1)

        model_1 = high_model.Grism(grism_hdul)
        model_2 = high_model.Grism(multi_ap_grism_hdul)
        one_dim_pane.add_model(model_1)
        one_dim_pane.add_model(model_2)
        blank_view.figure.panes = [one_dim_pane]

        if all_panes or all_files:
            item = mocker.Mock(spec=QtWidgets.QTableWidgetItem)
            item.row.return_value = 0
            mocker.patch.object(QtWidgets.QTableWidget, 'selectedItems',
                                return_value=[item])
        results = blank_view._parse_filename_table_selection()

        assert isinstance(results, tuple)
        assert all([isinstance(result, list) for result in results])

    @pytest.mark.parametrize('multi_ord,multi_ap,kind',
                             [(False, False, 'order'), (True, False, 'order'),
                              (False, True, 'order'), (True, True, 'order')])
    def test_enabled_disabled_orders(self, blank_view, mocker, multi_ord,
                                     multi_ap, kind):
        mock = mocker.patch.object(figure.Figure, 'get_orders',
                                   side_effect=({0: [1, 2]},
                                                {0: [0, 1, 2, 3]}))
        blank_view.multi_order = multi_ord
        blank_view.multi_aper = multi_ap

        enabled, disabled = blank_view._enabled_disabled_orders(0, 'test')
        assert mock.called_with({'kind': kind, 'model_id': 'test',
                                 'target': 0})
        assert enabled == [1, 2]
        assert disabled == [0, 3]

        enabled, disabled = blank_view._enabled_disabled_orders(-1, 'test')
        assert len(enabled) == 0
        assert len(disabled) == 0

    @pytest.mark.parametrize('orders,result',
                             [([0, 1, 2, 3], '1-4'), ([0, 2, 3], '1,3-4'),
                              ([0, 1, 5, 6, 7, 9], '1-2,6-8,10')])
    def test_format_orders_pairs(self, orders, result):
        output = view.View.format_orders_pairs(orders)
        assert output == result

    @pytest.mark.parametrize('multi_aper,multi_order,text',
                             [(False, False, 'Orders'),
                              (False, True, 'Orders'),
                              (True, False, 'Apertures'),
                              (True, True, 'Orders')])
    def test_configure_order_selector_labels(self, multi_aper, multi_order,
                                             blank_view, text):
        blank_view.multi_order = multi_order
        blank_view.multi_aper = multi_aper
        blank_view.enabled_orders_label.setText('')
        blank_view.hidden_orders_label.setText('')

        blank_view._configure_order_selector_labels()

        assert text in blank_view.enabled_orders_label.text()
        assert text in blank_view.hidden_orders_label.text()

    @pytest.mark.parametrize('enabled,disabled,same,en_list,dis_list',
                             [([0, 1, 2], [3], True, '1-3', '4'),
                              ([], [], True, '-', '-'),
                              ([0, 1, 2], [3], False, '1-3', '-')])
    def test_populate_enabled_disabled_orders(self, blank_view, mocker, same,
                                              one_dim_pane, grism_hdul,
                                              enabled, disabled, en_list,
                                              dis_list, multi_ap_grism_hdul):
        model_1 = high_model.Grism(grism_hdul)
        model_2 = high_model.Grism(multi_ap_grism_hdul)
        ap_order_state = {model_1.id: {'order': 1, 'aper': 1},
                          model_2.id: {'order': 1, 'aper': 2}}

        if same:
            mocker.patch.object(blank_view, '_enabled_disabled_orders',
                                return_value=(enabled, disabled))
        else:
            mocker.patch.object(blank_view, '_enabled_disabled_orders',
                                side_effect=((enabled, disabled),
                                             (enabled, enabled)))
        blank_view.on_orders_selector.setText('')
        blank_view.off_orders_selector.setText('')
        blank_view._populate_enabled_disabled_orders([one_dim_pane], [],
                                                     ap_order_state)
        assert blank_view.on_orders_selector.text() == ''
        assert blank_view.off_orders_selector.text() == ''
        blank_view._populate_enabled_disabled_orders([one_dim_pane],
                                                     [model_1.id, model_2.id],
                                                     ap_order_state)
        assert blank_view.on_orders_selector.text() == en_list
        assert blank_view.off_orders_selector.text() == dis_list
        assert not blank_view.multi_order
        assert blank_view.multi_aper

    def test_update_filename_table_selection(self, blank_view, mocker,
                                             grism_hdul, one_dim_pane,
                                             blank_figure,
                                             multi_ap_grism_hdul):
        mocker.patch.object(QtWidgets.QCheckBox, 'isChecked',
                            return_value=True)
        model = high_model.Grism(grism_hdul)
        one_dim_pane.add_model(model)
        one_dim_pane.add_model(high_model.Grism(multi_ap_grism_hdul))
        blank_figure.panes = [one_dim_pane]
        blank_view.figure = blank_figure
        blank_view._update_filename_table()

        assert blank_view.filename_table.rowCount() == 2
        assert len(blank_view.filename_table.selectedItems()) == 0

        blank_view._update_filename_table_selection(rows=[1])
        selection = blank_view.filename_table.selectedItems()
        assert len(set([x.row() for x in selection])) == 1

        blank_view._update_filename_table_selection(all_=True)
        selection = blank_view.filename_table.selectedItems()
        assert len(set([x.row() for x in selection])) == 2

    def test_decode_orders(self, blank_view, mocker, grism_hdul, caplog):
        caplog.set_level(logging.DEBUG)
        model = high_model.Grism(grism_hdul)
        mocker.patch.object(view.View, '_parse_filename_table_selection',
                            return_value=([model.id], [0]))

        result = blank_view.decode_orders('-')
        assert isinstance(result, dict)
        assert len(result) == 0

        mocker.patch.object(view.View, '_parse_filename_table_selection',
                            return_value=([model.id], None))
        result = blank_view.decode_orders('*')
        assert len(result) == 0

        mocker.patch.object(view.View, '_parse_filename_table_selection',
                            return_value=([model.id], [0]))
        mocker.patch.object(figure.Figure, 'get_orders',
                            return_value={0: [1, 2, 3]})
        result = blank_view.decode_orders('*')
        assert result == {model.id: [1, 2, 3]}

        caplog.clear()
        result = blank_view.decode_orders('-1,a')
        assert 'Invalid order notation' in caplog.text
        assert result == {model.id: []}

        result = blank_view.decode_orders('1,3-5,8,9')
        assert result == {model.id: [0, 2, 3, 4, 7, 8]}


class TestApertureColors(object):

    def test_init(self, qtbot, qapp, mocker):
        mocker.patch.object(QtWidgets, 'QApplication',
                            return_value=qapp)
        colors = ['#241100', '#909900', '#241100']
        obj = view.ApertureColors(colors)
        assert isinstance(obj.layout(), QtWidgets.QHBoxLayout)
        assert obj.layout().count() == len(set(colors))
