# Licensed under a 3-clause BSD style license - see LICENSE.rst

import logging
import pytest
from sofia_redux.visualization.display import (view, figure, cursor_location,
                                               pane, fitting_results)
from sofia_redux.visualization.utils import eye_error
from sofia_redux.visualization import signals, log
from sofia_redux.visualization.models import high_model, mid_model

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

        mocker.patch.object(QtWidgets.QTreeWidget, 'hasFocus',
                            return_value=True)
        mock = mocker.patch.object(view.View, 'remove_pane')

        qtbot.keyClick(blank_view, QtCore.Qt.Key_Delete)

        assert 'Key pushed in view: Del' in caplog.text
        assert mock.call_count == 1

    def test_atrophy_controls(self, blank_view, mocker, qtbot, caplog):
        caplog.set_level(logging.DEBUG)
        clear_mock = mocker.patch.object(view.View, 'clear_selection')
        update_mock = mocker.patch.object(view.View, 'update_controls')

        blank_view.atrophy_controls()

        assert clear_mock.call_count == 1
        assert update_mock.call_count == 1
        assert 'Controls no longer in sync' in caplog.text

    def test_toggle_pane_highlight(self, blank_view, mocker, qtbot):
        flag_mock = mocker.patch.object(figure.Figure,
                                        'set_pane_highlight_flag')
        with qtbot.wait_signal(blank_view.signals.atrophy):
            blank_view.toggle_pane_highlight()

        assert flag_mock.call_count == 1

    def test_refresh_orders(self, blank_view, mocker):
        mock = mocker.patch.object(view.View, 'refresh_order_list')

        blank_view.refresh_orders()

        assert mock.call_count == 1

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

    def test_setup_initial_scales(self, blank_view, mocker):
        pane_mock = mocker.Mock(spec_set=pane.OneDimPane)
        pane_mock.get_axis_scale.return_value = {'x': 'loglin', 'y': 'loglin'}
        mocker.patch.object(figure.Figure, 'get_current_pane',
                            return_value=pane_mock)

        # invalid values are ignored
        blank_view.setup_initial_scales()

    def test_setup_property_selectors(self, blank_view, grism_hdul):
        # no current pane, nothing is done
        blank_view.setup_property_selectors()
        assert blank_view.x_property_selector.currentText() == ''

        # set a pane, no models
        pane_ = pane.OneDimPane()
        blank_view.setup_property_selectors(pane_)
        assert blank_view.x_property_selector.currentText() == '-'
        assert blank_view.y_property_selector.currentText() == '-'

        # set a model
        model_grism = high_model.Grism(grism_hdul)
        pane_.models = {1: model_grism}
        pane_.plot_kind = 'spectrum'
        blank_view.setup_property_selectors(pane_)
        assert blank_view.x_property_selector.currentText() == 'wavepos'
        assert blank_view.y_property_selector.currentText() == 'spectral_flux'

        # setup for alt
        pane_.fields['y_alt'] = 'transmission'
        blank_view.setup_property_selectors(pane_, {'axis': 'alt'})
        assert blank_view.x_property_selector.currentText() == 'wavepos'
        assert blank_view.y_property_selector.currentText() == 'transmission'

    def test_setup_unit_selectors(self, blank_view, grism_hdul):
        # no current pane, nothing is done
        blank_view.setup_unit_selectors()
        assert blank_view.x_unit_selector.currentText() == ''

        # set a pane, no models
        pane_ = pane.OneDimPane()
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

    def test_setup_axis_limits(self, blank_view, mocker):
        # no current pane, nothing is done
        blank_view.setup_axis_limits()
        assert blank_view.x_limit_min.text() == ''

        # set a pane, no limits available
        pane_ = pane.OneDimPane()
        blank_view.setup_axis_limits(pane_)
        assert blank_view.x_limit_min.text() == '0'
        assert blank_view.x_limit_max.text() == '1'
        assert blank_view.y_limit_min.text() == '0'
        assert blank_view.y_limit_max.text() == '1'

        # set some limits
        mocker.patch.object(pane_, 'get_axis_limits',
                            return_value={'x': [1, 2], 'y': [3, 4],
                                          'y_alt': [5, 6]})
        blank_view.setup_axis_limits(pane_)
        assert blank_view.x_limit_min.text() == '1'
        assert blank_view.x_limit_max.text() == '2'
        assert blank_view.y_limit_min.text() == '3'
        assert blank_view.y_limit_max.text() == '4'

        # setup for alt
        blank_view.setup_axis_limits(pane_, {'axis': 'alt'})
        assert blank_view.x_limit_min.text() == '1'
        assert blank_view.x_limit_max.text() == '2'
        assert blank_view.y_limit_min.text() == '5'
        assert blank_view.y_limit_max.text() == '6'

    def test_remove_filename_fail(self, blank_view, mocker):
        # mocker.patch.object(list, 'remove', side_effect=ValueError)
        list_mock = mocker.Mock(spec_set=list)
        list_mock.remove.side_effect = ValueError
        blank_view.model_collection = list_mock

        filename = 'test.fits'
        blank_view.model_collection.append(filename)

        blank_view.remove_filename(filename)

    def test_add_panes(self, blank_view, mocker):
        mocker.patch.object(figure.Figure, 'set_layout_style')
        add_mock = mocker.patch.object(figure.Figure, 'add_panes')
        kind = ['spectrum', 'image']

        blank_view.add_panes(2, kind=kind)

        assert add_mock.called_with({'n_dims': [1, 2]})

        with pytest.raises(eye_error.EyeError) as msg:
            blank_view.add_panes(2, kind=['spectrum', 'cube'])
        assert 'Valid pane kinds' in str(msg)

    def test_update_pane_tree(self, blank_view, mocker, grism_hdul,
                              multiorder_hdul_merged, qtbot):
        model_grism = high_model.Grism(grism_hdul)
        model_multi = high_model.MultiOrder(multiorder_hdul_merged)
        model_multi.enabled = False
        blank_view.add_panes(n_panes=2, kind=['spectrum', 'spectrum'])
        blank_view.display_model(model_grism)
        blank_view.display_model(model_multi)

        qtbot.wait(200)

        count = blank_view.pane_tree_display.invisibleRootItem().childCount()
        assert count == 0

        blank_view.update_pane_tree()
        count = blank_view.pane_tree_display.invisibleRootItem().childCount()
        assert count == 2

    def test_update_pane_tree_disabled(self, blank_view, grism_hdul):
        model = high_model.Grism(grism_hdul)
        blank_view.display_model(model)

        for p in blank_view.figure.panes:
            p.set_all_models_enabled(False)

        blank_view.update_pane_tree()

        root = blank_view.pane_tree_display.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            for j in range(item.childCount()):
                enabled_item = item.child(j).child(1)
                if enabled_item and enabled_item.text(0) == 'Enabled':
                    assert enabled_item.checkState(0) == 0

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

    def test_refresh_order_list_skipping(self, blank_view, grism_hdul):
        model = high_model.Grism(grism_hdul)

        blank_view.display_model(model)
        pane_ = blank_view.figure.panes[0]
        pane_.fields['y'] = 'flux'

        blank_view.refresh_order_list()
        assert blank_view.order_list_widget.count() == 0

    def test_refresh_order_list_unchecked(self, blank_view, mocker,
                                          multiorder_hdul_spec):
        details = {'name': 'order',
                   'fields': {'spectral_flux': False}}
        mocker.patch.object(mid_model.Order, 'describe', return_value=details)
        model = high_model.MultiOrder(multiorder_hdul_spec)

        blank_view.display_model(model)

        blank_view.refresh_order_list()
        assert blank_view.order_list_widget.count() == 10
        assert all([not blank_view.order_list_widget.item(i).checkState()
                    for i in range(blank_view.order_list_widget.count())])

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
        # add_mock = mocker.patch.object(fitting_results.FittingResults,
        #                                'add_results')
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
        mocker.patch.object(figure.Figure, 'get_current_pane',
                            return_value=pane.OneDimPane())

        blank_view.setup_axis_limits()

        limits = [blank_view.x_limit_min, blank_view.x_limit_max,
                  blank_view.y_limit_min, blank_view.y_limit_max]
        assert all([limit.text() == '' for limit in limits])

    def test_remove_pane_bad(self, blank_view, mocker):
        mocker.patch.object(QtWidgets.QTreeWidget, 'selectedItems',
                            return_value=None)
        fig_mock = mocker.patch.object(figure.Figure, 'remove_pane')

        blank_view.remove_pane()

        assert fig_mock.call_count == 0

    def test_toggle_overplot(self, blank_view, mocker, qtbot):
        flag_mock = mocker.patch.object(figure.Figure,
                                        'set_overplot_state')
        with qtbot.wait_signal(blank_view.signals.atrophy_bg_full):
            blank_view.toggle_overplot()
        assert flag_mock.call_count == 1

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

    @pytest.mark.parametrize('modifier,ax_name,pane_val,axis_val,err_val',
                             [('All', '', 'all', 'all', False),
                              ('All', ' Primary', 'all', 'primary', False),
                              ('All', ' Overplot', 'all', 'alt', False),
                              ('Current', ' Both', 'current', 'all', False),
                              ('Current', ' Primary',
                               'current', 'primary', False),
                              ('Current', ' Overplot',
                               'current', 'alt', False),
                              ('Bad', ' text', '', '', True),
                              ('Bad', ' Primary', '', '', True)])
    def test_selected_target_axis(self, blank_view, mocker, modifier,
                                  ax_name, pane_val, axis_val, err_val):
        mocker.patch.object(blank_view.axes_selector,
                            'currentText', return_value=f'{modifier}{ax_name}')
        if not err_val:
            tgt = blank_view.selected_target_axis()
            assert len(tgt) == 2
            assert tgt['pane'] == pane_val
            assert tgt['axis'] == axis_val
        else:
            with pytest.raises(eye_error.EyeError) as err:
                blank_view.selected_target_axis()
            assert 'Unknown target axis' in str(err)
