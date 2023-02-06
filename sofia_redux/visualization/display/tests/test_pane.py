#  Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy

import matplotlib.axes
import matplotlib.collections
import pytest
import logging
import astropy.units as u
import astropy.modeling as am
import numpy as np
import numpy.testing as npt
from scipy import optimize as sco
from matplotlib import figure as mpf
from matplotlib import axes as mpa
from matplotlib import lines as ml
from matplotlib import collections as mc
from matplotlib import backend_bases as mpb
from matplotlib import artist as mart
from matplotlib import patches as mpp
from matplotlib import pyplot as plt

from sofia_redux.visualization.display import pane, drawing
from sofia_redux.visualization.models import (high_model, reference_model,
                                              low_model)
from sofia_redux.visualization.utils import eye_error, model_fit
from sofia_redux.visualization import signals

PyQt5 = pytest.importorskip('PyQt5')


class TestPane(object):

    def test_eq(self):
        sigs = signals.Signals()
        pane1 = pane.OneDimPane(sigs)
        pane2 = pane.OneDimPane(sigs)
        assert pane1 != 'test'
        assert pane1 == pane1
        assert pane1 == pane2
        pane2.ax = 'test'
        assert pane1 != pane2

    def test_set_border_visibility(self, blank_pane):
        blank_pane.set_border_visibility(True)

        blank_pane.border = mart.Artist()
        assert blank_pane.border.get_visible()

        blank_pane.set_border_visibility(False)
        assert not blank_pane.border.get_visible()

    def test_remove_model(self, blank_onedim, mocker,
                          grism_hdul, multiorder_hdul_merged):
        multi_model = high_model.MultiOrder(multiorder_hdul_merged)
        blank_onedim.models[multi_model.id] = multi_model

        grism_filename = grism_hdul.filename()
        grism_model = high_model.Grism(grism_hdul)
        blank_onedim.models[grism_model.id] = grism_model

        assert len(blank_onedim.models) == 2
        blank_onedim.remove_model(filename=grism_filename)
        assert len(blank_onedim.models) == 1

        blank_onedim.models[grism_filename] = grism_model
        assert len(blank_onedim.models) == 2
        blank_onedim.remove_model(model=multi_model)
        assert len(blank_onedim.models) == 1

        blank_onedim.remove_model(filename=grism_filename)
        assert len(blank_onedim.models) == 0

    def test_possible_units(self, blank_onedim, mocker, grism_hdul):

        grism_filename = grism_hdul.filename()
        grism_model = high_model.Grism(grism_hdul)
        blank_onedim.models[grism_filename] = grism_model

        mocker.patch.object(high_model.Grism, 'retrieve',
                            return_value=None)

        output = blank_onedim.possible_units()
        assert all([len(v) == 0 for v in output.values()])

    def test_current_units_empty(self, blank_onedim, mocker, grism_hdul):
        output = blank_onedim.current_units()
        assert all([len(v) == 0 for v in output.values()])

        grism_filename = grism_hdul.filename()
        grism_model = high_model.Grism(grism_hdul)
        blank_onedim.models[grism_filename] = grism_model

        mocker.patch.object(high_model.Grism, 'retrieve',
                            return_value=None)

        output = blank_onedim.current_units()
        assert all([len(v) == 0 for v in output.values()])

    def test_current_units_full(self, blank_onedim, mocker, grism_hdul):

        grism_filename = grism_hdul.filename()
        grism_model = high_model.Grism(grism_hdul)
        blank_onedim.models[grism_filename] = grism_model
        blank_onedim.units = {'x': u.um, 'y': u.nm}

        output = blank_onedim.current_units()
        assert output['x'] == blank_onedim.units['x']
        assert output['y'] == blank_onedim.units['y']

    def test_set_orders(self, blank_onedim, caplog, mocker,
                        multiorder_hdul_spec,
                        multiorder_hdul_merged):
        caplog.set_level(logging.DEBUG)

        order_mock = mocker.patch.object(high_model.MultiOrder,
                                         'enable_orders')

        orders = dict()
        orders[multiorder_hdul_spec.filename()] = [1, 2, 3]
        orders[multiorder_hdul_merged.filename()] = [1]

        blank_onedim.models = {
            multiorder_hdul_spec.filename():
                high_model.MultiOrder(multiorder_hdul_spec),
            multiorder_hdul_merged.filename():
                high_model.MultiOrder(multiorder_hdul_merged)}

        blank_onedim.orders = {multiorder_hdul_spec.filename(): [1, 2, 3]}

        blank_onedim.data_changed = False
        blank_onedim.set_orders(orders)
        assert blank_onedim.data_changed
        assert 'Enabling ' in caplog.text
        assert order_mock.call_count == 1

    def test_set_model_enabled(self, blank_onedim, grism_hdul, caplog):
        caplog.set_level(logging.DEBUG)
        model = high_model.Grism(grism_hdul)
        model_id = 'one'
        blank_onedim.models[model_id] = model

        assert blank_onedim.models[model_id].enabled
        blank_onedim.set_model_enabled(model_id, False)
        assert not blank_onedim.models[model_id].enabled
        assert f'Model {model_id} enabled' in caplog.text

    def test_set_all_models_enabled(self, blank_onedim, mocker):
        mock = mocker.patch.object(pane.OneDimPane,
                                   'set_model_enabled')
        blank_onedim.models = dict.fromkeys(list('abcd'), None)

        blank_onedim.set_all_models_enabled(True)

        assert mock.call_count == 4

    def test_create_artists_from_current_models(self, blank_onedim, mocker):
        pass

    def test_plot_model(self, blank_onedim, mocker, grism_hdul, caplog):
        caplog.set_level(logging.DEBUG)
        mocker.patch.object(high_model.Grism, 'retrieve',
                            return_value=None)
        mocker.patch.object(pane.OneDimPane, 'apply_configuration')

        ax_mock = mocker.Mock(spec_set=mpa.Axes)
        blank_onedim.ax = ax_mock

        model = high_model.Grism(grism_hdul)
        blank_onedim.orders[model.id] = model.orders

        lines = blank_onedim._plot_model(model)

        assert 'Ending limits' in caplog.text
        assert len(lines) == 0

    def test_plot_model_skip(self, blank_onedim, grism_hdul):
        model = high_model.Grism(grism_hdul)
        ax = mpf.Figure().subplots(1, 1)
        blank_onedim.ax = ax
        blank_onedim.models[model.id] = model
        blank_onedim.orders[model.id] = [0]
        blank_onedim.colors[model.id] = 'blue'
        blank_onedim.fields['y'] = 'response_error'
        lines = blank_onedim._plot_model(model)
        assert all([isinstance(line, drawing.Drawing) for line in lines])
        assert all([line.kind != 'error_range' for line in lines])

    def test_plot_model_same_fields(self, blank_onedim, mocker, grism_hdul):

        marker = '^'
        ax = mpf.Figure().subplots(1, 1)
        blank_onedim.ax = ax

        model = high_model.Grism(grism_hdul)
        model.enabled = False
        blank_onedim.models[model.id] = model
        blank_onedim.orders[model.id] = [0]
        blank_onedim.markers[model.id] = marker
        blank_onedim.colors[model.id] = 'blue'
        blank_onedim.plot_type = 'scatter'

        # Same fields
        blank_onedim.fields['y'] = blank_onedim.fields['x']
        blank_onedim.units = {'x': u.um, 'y': u.nm}

        lines = blank_onedim._plot_model(model)
        line, cursor = tuple(lines)
        # Being tested for fields
        assert (line.get_artist()._label.split(',')[-1].strip()
                == blank_onedim.fields['x'])
        assert isinstance(line.get_artist(), ml.Line2D)
        assert line.get_artist().get_marker() == marker
        assert line.get_artist().get_linestyle() == 'None'
        assert isinstance(cursor.get_artist(), ml.Line2D)
        # assert isinstance(error.get_artist(), mc.PolyCollection)
        assert all([not ln.get_artist().get_visible()
                    for ln in lines])

    def test_plot_model_bad_shape(self, blank_onedim, mocker, grism_hdul,
                                  caplog):
        ax = mocker.Mock(spec_set=mpa.Axes)
        blank_onedim.ax = ax
        caplog.set_level(logging.DEBUG)
        model = high_model.Grism(grism_hdul)

        flux = model.orders[0].data['spectral_flux'].data.copy()
        model.orders[0].data['spectral_flux'].data = flux[:-1]
        blank_onedim.models[model.id] = model
        blank_onedim.orders[model.id] = [0]

        lines = blank_onedim._plot_model(model)

        assert len(lines) == 0
        assert 'Incompatible data shape' in caplog.text

    def test_plot_model_bad_field(self, blank_onedim, mocker, grism_hdul,
                                  caplog):
        ax = mocker.Mock(spec_set=mpa.Axes)
        ax_alt = mocker.Mock(spec_set=mpa.Axes)
        blank_onedim.ax = ax
        blank_onedim.ax_alt = ax_alt
        caplog.set_level(logging.DEBUG)
        model = high_model.Grism(grism_hdul)

        blank_onedim.models[model.id] = model
        blank_onedim.orders[model.id] = [0]
        mocker.patch.object(blank_onedim, '_plot_cursor')
        mocker.patch.object(blank_onedim, '_convert_low_model_units')
        mocker.patch.object(blank_onedim, '_plot_single_line',
                            return_value=ml.Artist())

        # okay fields
        blank_onedim.fields['x'] = 'wavepos'
        blank_onedim.fields['y'] = 'spectral_error'
        blank_onedim._plot_model(model)
        assert 'Failed' not in caplog.text

        # bad primary field
        blank_onedim.fields['x'] = 'test'
        blank_onedim._plot_model(model)
        assert 'Failed to retrieve raw data for primary' in caplog.text

        # bad alt field
        blank_onedim.show_overplot = True
        blank_onedim.fields['x'] = 'wavepos'
        blank_onedim.fields['y_alt'] = 'test'
        blank_onedim._plot_model(model)
        assert 'Failed to retrieve raw data for alt' in caplog.text

    def test_plot_alt(self, blank_onedim, mocker, grism_hdul):
        mocker.patch.object(pane.OneDimPane, 'apply_configuration')
        marker = '^'
        ax = mpf.Figure().subplots(1, 1)
        blank_onedim.ax = ax
        blank_onedim.ax_alt = ax

        model = high_model.Grism(grism_hdul)

        blank_onedim.models[model.id] = model
        blank_onedim.orders[model.id] = [0]
        blank_onedim.markers[model.id] = marker
        blank_onedim.colors[model.id] = 'blue'
        blank_onedim.limits['y_alt'] = [0, 1]
        blank_onedim.show_overplot = True

        lines = blank_onedim._plot_model(model)
        line, cursor, error = tuple(lines)
        assert line.get_artist().get_linestyle() == '-'
        assert line.get_artist().get_marker() == 'None'

    def test_plot_flux_error(self, blank_onedim, mocker, grism_hdul,
                             blank_figure):
        ax = mpf.Figure().subplots(1, 1)
        blank_onedim.ax = ax
        aa = np.ones(10)
        err = None
        label = 'test_fake_spectrum.txt'
        colors = ['#59d4db', '#2848ad']
        for color in colors:
            output = blank_onedim._plot_flux_error(aa, aa, err, label, color)
            assert isinstance(output, matplotlib.collections.PolyCollection)

    def test_plot_model_scatter(self, blank_onedim, mocker, grism_hdul,
                                caplog, blank_figure):
        marker = '^'
        ax = mpf.Figure().subplots(1, 1)
        blank_onedim.ax = ax
        caplog.set_level(logging.DEBUG)
        model = high_model.Grism(grism_hdul)
        model.enabled = False
        blank_onedim.models[model.id] = model
        blank_onedim.orders[model.id] = [0]
        blank_onedim.markers[model.id] = marker
        blank_onedim.colors[model.id] = 'blue'

        blank_onedim.plot_type = 'scatter'
        lines = blank_onedim._plot_model(model)
        line, cursor, error = tuple(lines)

        assert isinstance(line.get_artist(), ml.Line2D)
        assert line.get_artist().get_marker() == marker
        assert line.get_artist().get_linestyle() == 'None'
        assert isinstance(cursor.get_artist(), ml.Line2D)
        assert isinstance(error.get_artist(), mc.PolyCollection)
        assert all([not ln.get_artist().get_visible()
                    for ln in lines])

    @pytest.mark.parametrize(
        'axis,plot_type,show_marker,visible,style,marker,width,alpha',
        [('primary', 'line', False, True, '-', 'None', 1.5, 1),
         ('primary', 'line', True, True, '-', 'x', 1.5, 1),
         ('primary', 'line', False, False, '-', 'None', 1.5, 1),
         ('primary', 'scatter', False, True, 'None', 'x', 1.5, 1),
         ('alt', 'line', False, True, ':', 'None', 1, 0.5)])
    def test_plot_single_line(self, one_dim_pane, grism_hdul, axis,
                              plot_type, show_marker, visible, style,
                              marker, width, alpha):
        model = high_model.Grism(grism_hdul)
        one_dim_pane.add_model(model)
        one_dim_pane.plot_type = plot_type
        one_dim_pane.show_markers = show_marker
        one_dim_pane.set_overplot(True)
        x = np.linspace(2, 10, 100)
        y = np.log10(x)
        label = 'test'

        line = one_dim_pane._plot_single_line(x, y, label, model.id,
                                              visible, axis)
        assert isinstance(line, ml.Line2D)
        assert line.get_alpha() == alpha
        assert line.get_linestyle() == style
        assert line.get_marker() == marker
        assert line.get_linewidth() == width
        assert line.get_visible() is visible

    @pytest.mark.parametrize('name,unit,label',
                             [('flux', u.Jy, 'Flux [Jy]'),
                              ('flux', '', 'Flux'),
                              ('flux', u.electron / u.mol,
                               'Flux\n[electron / mol]')])
    def test_generate_axes_label(self, name, unit, label):
        output = pane.OneDimPane._generate_axes_label(name, unit)
        assert output == label

    def test_axis_limits(self, blank_onedim):
        limits = {'x': [0, 1], 'y': [1, 2]}
        blank_onedim.limits = limits

        for axis, limit in limits.items():
            output = blank_onedim.get_axis_limits(axis)
            assert output == limit

        output = blank_onedim.get_axis_limits()
        assert output == limits

    @pytest.mark.parametrize('axis,correct',
                             [('x', 'um'), ('y', 'Jy'), ('z', '-'),
                              ('w', None)])
    def test_get_unit_string(self, blank_onedim, mocker, axis, correct):
        mock_ax = mocker.Mock()
        mock_ax.get_xlabel.return_value = 'Wave [um]'
        mock_ax.get_ylabel.return_value = 'Flux [Jy]'
        mock_ax.get_zlabel.return_value = 'Response'
        blank_onedim.ax = mock_ax

        if correct is None:
            with pytest.raises(SyntaxError) as msg:
                blank_onedim.get_unit_string(axis)
            assert 'Invalid axis selection' in str(msg)
        else:
            unit = blank_onedim.get_unit_string(axis)
            assert unit == correct

    def test_get_orders(self, one_dim_pane, multiorder_multiap_hdul,
                        grism_hdul):
        model = high_model.MultiOrder(multiorder_multiap_hdul)
        grism_model = high_model.Grism(grism_hdul)
        model.orders[0].enabled = False

        one_dim_pane.add_model(model)
        one_dim_pane.add_model(grism_model)

        result = one_dim_pane.get_orders()
        assert isinstance(result, list)
        assert len(result) == 5

        result = one_dim_pane.get_orders(kind='aperture')
        assert len(result) == 3

        result = one_dim_pane.get_orders(kind='all')
        assert len(result) == 5

        result = one_dim_pane.get_orders(enabled_only=True)
        assert isinstance(result, list)
        assert len(result) == 5

        result = one_dim_pane.get_orders(kind='aperture', enabled_only=True)
        assert len(result) == 3

        result = one_dim_pane.get_orders(kind='all', enabled_only=True)
        assert len(result) == 5

        result = one_dim_pane.get_orders(filename=grism_model.filename)
        assert len(result) == 1

        result = one_dim_pane.get_orders(enabled_only=True, by_model=True)
        assert isinstance(result, dict)
        assert result[model.id] == [0, 1, 2, 3, 4]

    def test_get_unit_empty(self, blank_onedim):
        result = blank_onedim.get_unit()
        assert all([v == '' for v in result.values()])

    def test_set_limits(self, blank_onedim, mocker):
        limits = {'x': [3, 8], 'z': [0, 1]}
        ax = mocker.Mock(spec_set=mpa.Axes)
        blank_onedim.ax = ax

        starting_limits = blank_onedim.limits
        assert list(blank_onedim.limits.keys()) == ['x', 'y', 'y_alt']

        blank_onedim.set_limits(limits)

        assert list(blank_onedim.limits.keys()) == ['x', 'y', 'y_alt']
        assert blank_onedim.limits['x'] == limits['x']
        assert blank_onedim.limits['y'] == starting_limits['y']

    def test_set_scales(self, blank_onedim):
        scales = {'x': 'log', 'z': 'log'}

        starting_scales = blank_onedim.scale
        assert list(blank_onedim.scale.keys()) == ['x', 'y', 'y_alt']

        blank_onedim.set_scales(scales)
        assert list(blank_onedim.scale.keys()) == ['x', 'y', 'y_alt']
        assert blank_onedim.scale['x'] == scales['x']
        assert blank_onedim.scale['y'] == starting_scales['y']

    def test_get_xy_data(self, blank_onedim, mocker, caplog, grism_hdul):

        model = high_model.Grism(grism_hdul)
        #
        # blank_onedim.add_model(model)
        blank_onedim.fields['x'] = 'wavepos'
        blank_onedim.fields['y'] = 'wavepos'
        blank_onedim.units = {'x': u.um, 'y': u.nm}
        x, y = blank_onedim.get_xy_data(model, order=0, aperture=0)

        assert id(x) != id(model)
        assert id(y) != id(model)
        assert id(x) != id(y)

        spectrum_x = x.retrieve(order=0, aperture=0,
                                level='low', field='wavepos')
        spectrum_y = y.retrieve(order=0, aperture=0,
                                level='low', field='wavepos')
        assert spectrum_x.unit_key == 'um'
        assert spectrum_y.unit_key == 'nm'

    @pytest.mark.parametrize('target, current',
                             [({'x': u.nm, 'y': 'Jy'},
                               {'x': 'pixel', 'y': u.Jy}),
                              ({'x': 'pixel', 'y': u.mol},
                               {'x': 'pixel', 'y': u.Jy})])
    def test_set_units_pixels(self, qtbot, mocker, blank_onedim, grism_hdul,
                              target, current):
        mocker.patch.object(pane.OneDimPane, '_convert_low_model_units',
                            side_effect=ValueError)
        model = high_model.Grism(grism_hdul)
        blank_onedim.models[model.id] = model
        blank_onedim.orders[model.id] = ['0.0']
        blank_onedim.units = current
        units = target
        with qtbot.wait_signal(blank_onedim.signals.obtain_raw_model):
            blank_onedim.set_units(units, 'primary')

    def test_set_units_alt(self, one_dim_pane, grism_hdul):
        model = high_model.Grism(grism_hdul)
        one_dim_pane.add_model(model)

        one_dim_pane.set_overplot(True)
        one_dim_pane.set_fields({'y_alt': 'spectral_flux'})
        one_dim_pane.data_changed = False
        units = {'x': 'nm', 'y': 'W / m2'}
        one_dim_pane.set_units(units, axes='alt')
        assert one_dim_pane.data_changed

    def test_set_units_err_none(self, qtbot, mocker, blank_onedim, grism_hdul):
        mocker.patch.object(pane.OneDimPane, '_convert_low_model_units',
                            side_effect=ValueError)
        model = high_model.Grism(grism_hdul)

        blank_onedim.models[model.id] = model

        mocker.patch.object(model, 'retrieve',
                            return_value=None)

        # mock retrieve multiple times
        blank_onedim.orders[model.id] = ['0.0']
        blank_onedim.fields['y'] = 'flux'
        blank_onedim.fields['x'] = 'wavepos'
        blank_onedim.units = {'x': u.nm, 'y': u.Jy}
        units = {'x': u.m, 'y': u.mol}
        model = blank_onedim.models[model.id]
        blank_onedim.set_units(units, 'primary')
        error_spectrum = model.retrieve(order=0, level='low',
                                        field='spectral_error')

        assert error_spectrum is None

    def test_update_visibility(self, mocker, blank_onedim, grism_hdul):

        model = high_model.Grism(grism_hdul)

        spectrum = model.retrieve(order=0, aperture=0,
                                  level='low', field='spectral_flux')
        blank_onedim.models[model.id] = model

        mocker.patch.object(model, 'retrieve', return_value=spectrum)
        blank_onedim.orders[model.id] = ['0.0']

        updates = blank_onedim.update_visibility(error=None)
        assert updates[0]._kind == 'line'

    def test_set_units_fail(self, blank_onedim, mocker, caplog, grism_hdul):
        caplog.set_level(logging.DEBUG)
        mocker.patch.object(pane.OneDimPane, '_convert_low_model_units',
                            side_effect=ValueError)
        model = high_model.Grism(grism_hdul)

        blank_onedim.models[model.id] = model
        blank_onedim.orders[model.id] = ['0.0']
        blank_onedim.fields['y'] = 'response'
        blank_onedim.units = {'x': u.um, 'y': u.Jy}
        units = {'x': u.nm, 'y': u.mol}

        blank_onedim.data_changed = False

        updates = blank_onedim.set_units(units, 'primary')

        assert isinstance(updates, list)
        assert len(updates) == 0
        assert not blank_onedim.data_changed
        for unit in units.values():
            assert f'Cannot convert units to {unit}' in caplog.text

    @pytest.mark.parametrize('state,flag', [(True, True), (False, False)])
    def test_set_markers_scatter(self, blank_onedim, state, flag):
        blank_onedim.plot_type = 'scatter'

        updates = blank_onedim.set_markers(state)

        assert blank_onedim.show_markers is flag
        assert len(updates) == 0

    @pytest.mark.parametrize('xdata,isnan', [(-2, False), (5, True)])
    def test_data_at_cursor_invisible(self, blank_onedim, mocker,
                                      grism_hdul, blank_figure,
                                      xdata, isnan):

        mocker.patch.object(np, 'nanargmin', return_value=4)
        mocker.patch.object(np, 'isnan', return_value=np.array([isnan]))
        model = high_model.Grism(grism_hdul)

        blank_onedim.models[model.id] = model
        blank_onedim.orders[model.id] = ['0.0']
        blank_onedim.colors[model.id] = 'blue'

        # add an alt axis to test too
        blank_onedim.show_overplot = True
        blank_onedim.fields['y_alt'] = blank_onedim.fields['y']

        event = mpb.MouseEvent(x=2, y=3, canvas=blank_figure.widget.canvas,
                               name='motion_notify_event')
        event.xdata = xdata

        data = blank_onedim.data_at_cursor(event)
        assert data[model.id][0]['visible'] is False
        assert data[model.id][1]['visible'] is False

    @pytest.mark.parametrize('xdata,isnan', [(2000, False)])
    def test_data_at_cursor_same_fields(self, blank_onedim, mocker,
                                        grism_hdul, blank_figure, xdata,
                                        isnan):
        mocker.patch.object(np, 'nanargmin', return_value=4)
        mocker.patch.object(np, 'isnan', return_value=np.array([isnan]))
        model = high_model.Grism(grism_hdul)

        blank_onedim.models[model.id] = model
        blank_onedim.orders[model.id] = ['0.0']
        blank_onedim.colors[model.id] = 'blue'

        # add an alt axis to test too
        blank_onedim.show_overplot = False
        blank_onedim.fields['y'] = blank_onedim.fields['x']
        blank_onedim.units = {'x': u.nm, 'y': u.nm}

        event = mpb.MouseEvent(x=2, y=3, canvas=blank_figure.widget.canvas,
                               name='motion_notify_event')
        event.xdata = xdata

        data = blank_onedim.data_at_cursor(event)

        assert data[model.id][0]['visible'] is False
        assert data[model.id][0]['y_field'] == data[model.id][0]['x_field']

    def test_perform_fit_bad(self, blank_onedim):
        arts, params = blank_onedim.perform_fit('trapezoid', [0, 10])
        assert arts is None
        assert params is None
        arts, params = blank_onedim.perform_fit('fit_none_none', [0, 10])
        assert arts is None
        assert params is None

    @pytest.mark.parametrize('model_en,order_en,mock_ret,'
                             'mock_fit_1,mock_fit_2,xlow',
                             [(False, True, False, False, False, 3),
                              (True, False, False, False, False, 3),
                              (True, True, True, False, False, 3),
                              (True, True, False, False, False, 100),
                              (True, True, False, True, False, 3),
                              (True, True, False, False, True, 3)])
    def test_gauss_fit_failures(self, blank_onedim, mocker, caplog,
                                grism_hdul, model_en, order_en,
                                mock_ret, mock_fit_1, mock_fit_2, xlow):
        caplog.set_level(logging.DEBUG)
        model = high_model.Grism(grism_hdul)
        limits = [[xlow, 8], [30, 40]]

        model.enabled = model_en
        model.orders[0].data['spectral_flux'].enabled = order_en
        blank_onedim.models[model.id] = model
        blank_onedim.orders[model.id] = [0]
        blank_onedim.colors[model.id] = 'blue'
        ax = mpf.Figure().subplots(1, 1)
        blank_onedim.ax = ax

        if mock_ret:
            mocker.patch.object(high_model.Grism, 'retrieve',
                                return_value=None)
        if mock_fit_1:
            mocker.patch.object(sco, 'curve_fit',
                                side_effect=RuntimeError)
        if mock_fit_2:
            mocker.patch.object(sco, 'curve_fit',
                                side_effect=ValueError)

        arts, params = blank_onedim.perform_fit('fit_gauss_constant', limits)
        assert len(arts) == 0
        if mock_fit_1 or mock_fit_2 or xlow > 10:
            # empty/failed fit still returns parameters but not artists
            assert len(params) == 1
        else:
            assert len(params) == 0
        assert '; skipping' in caplog.text

    def test_contains_model(self, blank_onedim, grism_hdul):
        grism_model = high_model.Grism(grism_hdul)
        blank_onedim.models[grism_model.id] = grism_model
        blank_onedim.orders[grism_model.id] = [0]

        assert blank_onedim.contains_model(grism_model.id)
        assert blank_onedim.contains_model(grism_model.id, order=0)

        assert not blank_onedim.contains_model('test.fits')
        assert not blank_onedim.contains_model(grism_model.id, order=1)

    def test_get_field(self, blank_onedim):
        expected = {'x': 'wavepos', 'y': 'spectral_flux', 'y_alt': None}

        # no axis
        assert blank_onedim.get_field() == expected

        # x, y axes present
        assert blank_onedim.get_field(axis='x') == expected['x']
        assert blank_onedim.get_field(axis='y') == expected['y']

        # alt axis, default
        with pytest.raises(eye_error.EyeError) as err:
            blank_onedim.get_field(axis='alt')
        assert 'Unable to retrieve field' in str(err)

        # alt axis, specified
        blank_onedim.fields['y_alt'] = 'test'
        assert blank_onedim.get_field(axis='alt') == 'test'

        # bad axis
        with pytest.raises(eye_error.EyeError) as err:
            blank_onedim.get_field(axis='bad')
        assert 'Unable to retrieve field' in str(err)

    def test_get_unit(self, blank_onedim):
        expected = {'x': '', 'y': '', 'y_alt': ''}

        # no axis
        assert blank_onedim.get_unit() == expected

        # x, y axes present
        assert blank_onedim.get_unit(axis='x') == expected['x']
        assert blank_onedim.get_unit(axis='y') == expected['y']
        assert blank_onedim.get_unit(axis='y_alt') == expected['y_alt']

        # alt axis, specified
        blank_onedim.units['y_alt'] = 'test'
        assert blank_onedim.get_unit(axis='alt') == 'test'

        # bad axis
        with pytest.raises(eye_error.EyeError) as err:
            blank_onedim.get_unit(axis='bad')
        assert 'Unable to retrieve unit' in str(err)

        # no units present: reset to default
        blank_onedim.units = None
        assert blank_onedim.get_unit() == expected

    def test_get_scale(self, blank_onedim):
        expected = {'x': 'linear', 'y': 'linear', 'y_alt': None}

        # no axis
        assert blank_onedim.get_scale() == expected

        # x, y axes present
        assert blank_onedim.get_scale(axis='x') == expected['x']
        assert blank_onedim.get_scale(axis='y') == expected['y']

        # alt axis, default
        with pytest.raises(eye_error.EyeError) as err:
            blank_onedim.get_scale(axis='alt')
        assert 'Unable to retrieve scale' in str(err)

        # alt axis, specified
        blank_onedim.scale['y_alt'] = 'test'
        assert blank_onedim.get_scale(axis='alt') == 'test'

        # bad axis
        with pytest.raises(eye_error.EyeError) as err:
            blank_onedim.get_scale(axis='bad')
        assert 'Unable to retrieve scale' in str(err)

    def test_update_error_artists(self, blank_onedim, grism_hdul):
        grism_model = high_model.Grism(grism_hdul)
        blank_onedim.models[grism_model.id] = grism_model
        blank_onedim.orders[grism_model.id] = ['0.0']
        blank_onedim.markers[grism_model.id] = 'x'
        blank_onedim.colors[grism_model.id] = 'blue'
        ax = mpf.Figure().subplots(1, 1)
        blank_onedim.ax = ax

        updates = blank_onedim._update_error_artists()
        assert len(updates) == 1
        assert updates[0].updates['artist'].get_visible()

        # disable model: error artist should not be visible
        grism_model.enabled = False
        updates = blank_onedim._update_error_artists()
        assert len(updates) == 1
        assert not updates[0].updates['artist'].get_visible()

        # same for show_error = False
        grism_model.enabled = True
        blank_onedim.show_error = False
        updates = blank_onedim._update_error_artists()
        assert len(updates) == 1
        assert not updates[0].updates['artist'].get_visible()

    def test_overplot_no_op(self, blank_onedim):
        # todo: check if this is the desired behavior -
        #  should y_alt field be '' or None?
        blank_onedim.show_overplot = False
        blank_onedim.set_overplot(False)
        assert blank_onedim.fields['y_alt'] is None

        blank_onedim.show_overplot = True
        blank_onedim.set_overplot(False)
        assert blank_onedim.fields['y_alt'] == ''

    def test_reset_alt_axes_no_op(self, blank_onedim, caplog):
        caplog.set_level(logging.DEBUG)
        blank_onedim.reset_alt_axes(remove=True)
        assert 'Failed to remove' in caplog.text
        assert blank_onedim.ax_alt is None

    @pytest.mark.parametrize('xdata,ydata,altdata',
                             [(-2, 3, -0.1347),
                              (5, 4, -0.1347)])
    def test_xy_at_cursor(self, blank_onedim, grism_hdul, blank_figure,
                          xdata, ydata, altdata):
        fn = grism_hdul.filename()
        grism_model = high_model.Grism(grism_hdul)
        blank_onedim.models[fn] = grism_model
        blank_onedim.orders[fn] = [0]
        blank_onedim.markers[fn] = 'x'
        blank_onedim.colors[fn] = 'blue'
        ax = mpf.Figure().subplots(1, 1)
        blank_onedim.ax = ax

        event = mpb.MouseEvent(x=2, y=3, canvas=blank_figure.widget.canvas,
                               name='motion_notify_event')
        event.xdata = xdata
        event.ydata = ydata
        event.inaxes = ax

        x, y = blank_onedim.xy_at_cursor(event)
        assert x == event.xdata
        assert y == event.ydata

        # add an alternate axis: x/ydata is now for the alt,
        # xy_at_cursor should return the primary y, not the ydata
        blank_onedim.set_overplot(True)
        event.inaxes = blank_onedim.ax_alt
        x, y = blank_onedim.xy_at_cursor(event)
        assert x == event.xdata
        assert np.allclose(y, altdata, atol=1e-4)

    def test_update_reference_data(self, mocker, blank_onedim):
        # no reference models existing or provided
        assert blank_onedim.update_reference_data() is None

        # provide an empty reference model
        ref = reference_model.ReferenceData()
        updates = blank_onedim.update_reference_data(reference_models=ref)
        assert blank_onedim.reference is ref
        assert len(updates) == 0

        # bad x-field
        blank_onedim.fields['x'] = 'spectral_flux'
        updates = blank_onedim.update_reference_data(reference_models=ref)
        assert updates is None

        # plot true/false calls different functions
        blank_onedim.fields['x'] = 'wavepos'
        m1 = mocker.patch.object(blank_onedim, '_plot_reference_lines')
        m2 = mocker.patch.object(blank_onedim, '_current_reference_options')
        blank_onedim.update_reference_data(plot=True)
        blank_onedim.update_reference_data(plot=False)
        m1.assert_called_once()
        m2.assert_called_once()

    def test_unload_ref_model(self, mocker, blank_onedim):
        ref = reference_model.ReferenceData()
        m1 = mocker.patch.object(ref, 'unload_data')

        # no op if reference not loaded
        blank_onedim.unload_ref_model()

        # data unload called if is
        blank_onedim.reference = ref
        blank_onedim.unload_ref_model()
        m1.assert_called_once()

    def test_plot_reference_lines(self, blank_onedim, grism_hdul,
                                  line_list_csv):
        assert blank_onedim._plot_reference_lines() == list()

        # load some data
        grism_model = high_model.Grism(grism_hdul)
        blank_onedim.models[grism_model.id] = grism_model
        blank_onedim.orders[grism_model.id] = [0]
        blank_onedim.markers[grism_model.id] = 'x'
        blank_onedim.colors[grism_model.id] = 'blue'
        ax = mpf.Figure().subplots(1, 1)
        blank_onedim.ax = ax
        blank_onedim._plot_model(grism_model)

        # load a reference model
        ref = reference_model.ReferenceData()
        ref.add_line_list(line_list_csv)
        blank_onedim.reference = ref

        # all lines out of data range
        lines = blank_onedim._plot_reference_lines()
        assert len(lines) == 0

        # make a little test function to check for created artists
        def artist_count(lines):
            count_line = 0
            count_label = 0
            for ln in lines:
                assert isinstance(ln, drawing.Drawing)
                if ln.kind == 'ref_line' and ln.artist is not None:
                    count_line += 1
                elif ln.kind == 'ref_label' and ln.artist is not None:
                    count_label += 1
            return count_line, count_label

        # set limits to include 9 lines and labels
        blank_onedim.set_limits({'x': [0, 5]})
        lines = blank_onedim._plot_reference_lines()
        assert len(lines) == 18
        assert artist_count(lines) == (9, 9)

        # hide labels and update
        ref.set_visibility('ref_label', False)
        lines = blank_onedim._plot_reference_lines()
        assert len(lines) == 18
        assert artist_count(lines) == (9, 0)

        # hide lines and update: ignores both labels and lines
        ref.set_visibility('ref_label', True)
        ref.set_visibility('ref_line', False)
        lines = blank_onedim._plot_reference_lines()
        assert len(lines) == 18
        assert artist_count(lines) == (0, 0)

        # append a line with the same label as another line:
        # both should be plotted
        ref.line_list['H 7-5'] = [4.654, 4.781]
        ref.set_visibility('ref_line', True)
        lines = blank_onedim._plot_reference_lines()
        assert len(lines) == 20
        assert artist_count(lines) == (10, 10)

    def test_window_line_list(self, blank_onedim, grism_hdul, mocker):

        ref = reference_model.ReferenceData()
        blank_onedim.reference = ref
        mocker.patch.object(ref, 'convert_line_list_unit',
                            return_value=dict())
        fn = grism_hdul.filename()
        grism_model = high_model.Grism(grism_hdul)
        blank_onedim.models[fn] = grism_model
        blank_onedim.orders[fn] = [0]
        blank_onedim.units['x'] = 'pixel'

        name_limits = blank_onedim._window_line_list()
        assert name_limits == dict()

    def test_window_line_list_fail(self, blank_onedim, mocker):
        ref = reference_model.ReferenceData()
        blank_onedim.reference = ref
        mocker.patch.object(ref, 'convert_line_list_unit',
                            side_effect=KeyError)

        output = blank_onedim._window_line_list([2, 4])
        assert output == dict()

    def test_current_reference_options(self, blank_onedim, line_list_csv):
        ax = mpf.Figure().subplots(1, 1)
        blank_onedim.ax = ax

        # none without reference model
        options = blank_onedim._current_reference_options()
        assert len(options) == 0
        # load a reference model
        ref = reference_model.ReferenceData()
        ref.add_line_list(line_list_csv)
        blank_onedim.reference = ref

        # all lines out of range
        options = blank_onedim._current_reference_options()
        assert len(options) == 0

        # set limits to include 9 lines and labels
        blank_onedim.set_limits({'x': [0, 5]})
        options = blank_onedim._current_reference_options()
        assert len(options) == 18

        count_line = 0
        count_label = 0
        for opt in options:
            assert isinstance(opt, dict)
            if opt['model_id'] == 'ref_lines':
                count_line += 1
            elif opt['model_id'] == 'ref_labels':
                count_label += 1
        assert count_line == 9
        assert count_label == 9

    def test_set_units_reference(self, blank_onedim, grism_hdul):
        # load some data
        model = high_model.Grism(grism_hdul)
        blank_onedim.models[model.id] = model
        blank_onedim.orders[model.id] = ['0.0']
        blank_onedim.markers[model.id] = 'x'
        blank_onedim.colors[model.id] = 'blue'
        ax = mpf.Figure().subplots(1, 1)
        blank_onedim.ax = ax

        # units to change to/from
        units1 = {'x': u.um, 'y': u.Jy}
        units2 = {'x': u.nm, 'y': u.Jy}
        blank_onedim.units = units1

        # add a reference model with 3 lines in data range
        ref = reference_model.ReferenceData()
        ref.line_list = {'1': [5.1], '2': [6.3], '3': [7.4]}
        ref.line_unit = u.um
        ref.set_visibility('ref_line', True)
        blank_onedim.reference = ref

        # 2 plot updates (flux, error), 3 lines, 3 labels
        updates = blank_onedim.set_units(units2, 'primary')
        assert len(updates) == 8

        label = 1
        for ln in updates:
            if ln.kind == 'ref_line':
                # um value from line list should be converted to nm value
                # in vline artist
                assert np.allclose(ln.artist.get_data()[0],
                                   ref.line_list[str(label)][0] * 1000)
                label += 1

    def test_axes(self, one_dim_pane):
        assert not one_dim_pane.show_overplot
        axes = one_dim_pane.axes()
        assert isinstance(axes, list)
        assert len(axes) == 2
        assert isinstance(axes[0], mpa.Axes)
        assert axes[1] is None

        one_dim_pane.set_overplot(True)
        axes = one_dim_pane.axes()
        assert isinstance(axes[0], mpa.Axes)
        assert isinstance(axes[1], mpa.Axes)
        assert axes[0].bbox.bounds == axes[1].bbox.bounds

    def test_set_axis(self, blank_onedim):
        fig, ax = plt.subplots(1, 1)
        assert blank_onedim.ax is None
        assert blank_onedim.ax_alt is None
        assert blank_onedim.border is None

        blank_onedim.set_axis(ax, kind='alt')
        assert blank_onedim.ax is None
        assert blank_onedim.ax_alt == ax
        assert blank_onedim.border is None

        blank_onedim.set_axis(ax)
        assert blank_onedim.ax == ax
        assert blank_onedim.ax_alt == ax
        assert isinstance(blank_onedim.border, mpp.Rectangle)

    def test_overplot_state(self, one_dim_pane):
        assert not one_dim_pane.overplot_state()
        one_dim_pane.set_overplot(True)
        assert one_dim_pane.overplot_state()

    @pytest.mark.parametrize('method, nargs',
                             [('get_scale', 1), ('get_orders', 1),
                              ('set_limits', 1), ('set_scales', 1),
                              ('set_units', 2), ('set_fields', 1),
                              ('get_unit', 1), ('get_field', 1),
                              ('get_axis_scale', 0), ('get_unit_string', 1),
                              ('get_axis_limits', 0), ('current_units', 0),
                              ('possible_units', 0), ('set_orders', 1),
                              ('remove_model', 0), ('add_model', 1),
                              ('update_colors', 0), ('perform_zoom', 2),
                              ('set_plot_type', 1), ('set_markers', 1),
                              ('set_grid', 1), ('set_error', 1),
                              ('create_artists_from_current_models', 0)])
    def test_root_notimplemented(self, method, nargs):
        sigs = signals.Signals()
        p = pane.Pane(sigs)
        args = list(range(nargs))
        with pytest.raises(NotImplementedError):
            getattr(p, method)(*args)

    def test_ap_order_state(self, one_dim_pane, mocker):
        model_mock = mocker.Mock(spec=high_model.HighModel)
        n_ap = 2
        n_or = 10
        args = {'ap_order_count.return_value': (n_ap, n_or)}
        model_mock.configure_mock(**args)

        model_id = 1
        one_dim_pane.models[model_id] = model_mock

        apertures, orders = one_dim_pane.ap_order_state(model_id)
        assert isinstance(apertures, dict)
        assert isinstance(orders, dict)
        assert len(apertures) == len(orders)
        assert apertures[model_id] == n_ap
        assert orders[model_id] == n_or

        apertures, orders = one_dim_pane.ap_order_state(2)
        assert apertures[2] == 0
        assert orders[2] == 0

    def test_model_extensions(self, one_dim_pane, multiorder_hdul_merged):
        model = high_model.MultiOrder(multiorder_hdul_merged)
        one_dim_pane.add_model(model)

        output = one_dim_pane.model_extensions(model.id)
        assert output == model.extensions()

        output = one_dim_pane.model_extensions('id')
        assert isinstance(output, list)
        assert len(output) == 0

    def test_add_border_highlight(self, one_dim_pane):
        rect = one_dim_pane._add_border_highlight()

        assert isinstance(rect, mpp.Rectangle)
        assert not rect.get_visible()

    def test_get_border(self, one_dim_pane):
        output = one_dim_pane.get_border()
        assert output is None

        rect = one_dim_pane._add_border_highlight()
        one_dim_pane.border = rect

        output = one_dim_pane.get_border()
        assert output == rect

    @pytest.mark.parametrize('cycle,result', [('SPECTRAL', 'brewer_cycle'),
                                              ('tableau', 'tab10_cycle'),
                                              ('other', 'accessible_cycle')])
    def test_set_color_cycle_by_name(self, one_dim_pane, mocker, cycle,
                                     result):
        mock = mocker.patch.object(one_dim_pane, 'set_aperture_cycle')
        one_dim_pane.set_color_cycle_by_name(cycle)
        assert mock.called_once
        assert one_dim_pane.default_colors == getattr(one_dim_pane, result)

    @pytest.mark.parametrize('color, scheme, result',
                             [('#343434', 'yiq', 0.203921568),
                              (['#343434', '#454545'], 'yiq', 0.203921568),
                              ('#343434', 'hex', '#343434'),
                              ('#343434', 'rgb', (0.203922, 0.203922,
                                                  0.203922)),
                              ('#343434', 'other', None)])
    def test_grayscale(self, one_dim_pane, color, scheme, result):
        if result is None:
            with pytest.raises(eye_error.EyeError):
                one_dim_pane.grayscale(color, scheme)
        else:
            output = one_dim_pane.grayscale(color, scheme)
            if isinstance(result, float):
                npt.assert_allclose(float(output), result)
            elif isinstance(result, tuple):
                output = tuple([float(i) for i in output])
                npt.assert_allclose(output, result, atol=1e-4)
            else:
                assert output == result

    def test_set_aperture_cycle(self, one_dim_pane, mocker):
        mock = mocker.patch.object(one_dim_pane, 'analogous', return_value=5)

        one_dim_pane.set_aperture_cycle()

        assert mock.call_count == len(one_dim_pane.default_colors)
        assert all([v == 5 for v in one_dim_pane.aperture_cycle.values()])
        assert (list(one_dim_pane.aperture_cycle.keys())
                == one_dim_pane.default_colors)

    @pytest.mark.parametrize('color,result',
                             [('#343434', ['#343434', '#343434']),
                              ('#0000ff', ['#ff7f00', '#80ff00']),
                              ('#ffffff', ['#ffffff', '#ffffff']),
                              ('#123456', ['#561212', '#565612'])
                              ])
    def test_split_complementary(self, color, result):
        output = pane.Pane.split_complementary(color)
        assert output == result

    @pytest.mark.parametrize('color,degree,result',
                             [('#343434', 130, ['#343434', '#343434']),
                              ('#123456', 130, ['#3f5612', '#561229']),
                              ('#123456', 30, ['#125656', '#121256']),
                              ('#123456', 0, ['#123456', '#123456']),
                              ('#123456', 180, ['#563412', '#563412'])])
    def test_analogous(self, color, degree, result):
        output = pane.Pane.analogous(color, degree)
        assert output == result


class TestOneDimPane(object):

    def test_model_summaries(self, one_dim_pane, grism_hdul):
        model = high_model.Grism(grism_hdul)
        one_dim_pane.add_model(model)

        details = one_dim_pane.model_summaries()
        assert isinstance(details, dict)
        assert len(details) == 1
        assert len(details[model.id]) == 7
        assert details[model.id]['model_id'] == model.id

    def test_add_model(self, one_dim_pane, multiorder_multiap_hdul):
        mo_model = high_model.MultiOrder(multiorder_multiap_hdul)
        one_dim_pane.add_model(mo_model)
        assert mo_model.id in one_dim_pane.models.keys()

    def test_update_model(self, one_dim_pane, grism_hdul):
        model_1 = high_model.Grism(grism_hdul)
        model_2 = copy.deepcopy(model_1)
        assert model_1.id == model_2.id

        one_dim_pane.add_model(model_1)
        one_dim_pane.set_model_enabled(model_1.id, False)
        assert not one_dim_pane.models[model_1.id].enabled
        assert model_2.enabled

        one_dim_pane.update_model({model_1.id: model_2})
        assert not one_dim_pane.models[model_1.id].enabled

    def test_remove_model(self, one_dim_pane, grism_hdul, caplog):
        caplog.set_level(logging.DEBUG)
        model = high_model.Grism(grism_hdul)

        one_dim_pane.add_model(model)
        assert model.id in one_dim_pane.models
        one_dim_pane.remove_model(model_id=model.id)
        assert model.id not in one_dim_pane.models

        one_dim_pane.add_model(model)
        assert model.id in one_dim_pane.models
        one_dim_pane.remove_model(model=model)
        assert model.id not in one_dim_pane.models

        one_dim_pane.add_model(model)
        assert model.id in one_dim_pane.models
        one_dim_pane.remove_model(filename=model.filename)
        assert model.id not in one_dim_pane.models

        one_dim_pane.remove_model(filename=model.filename)
        assert 'Unable to remove' in caplog.text

    def test_create_artists_from_current_models(self, one_dim_pane, mocker,
                                                grism_hdul, qtbot):
        mocker.patch.object(pane.OneDimPane, '_plot_model',
                            return_value=[None])

        model = high_model.Grism(grism_hdul)
        one_dim_pane.add_model(model)

        result = one_dim_pane.create_artists_from_current_models()

        assert isinstance(result, list)
        assert len(result) == len(one_dim_pane.models)

        one_dim_pane.fields['x'] = 'wavepos'
        one_dim_pane.fields['y'] = 'wavepos'

        with qtbot.wait_signal(one_dim_pane.signals.obtain_raw_model):
            one_dim_pane.create_artists_from_current_models()

    def test_create_artists_from_current_models_bad(self, one_dim_pane,
                                                    mocker, grism_hdul,
                                                    caplog):
        caplog.set_level(logging.WARNING)
        model = high_model.Grism(grism_hdul)
        one_dim_pane.add_model(model)

        mocker.patch.object(pane.OneDimPane, '_plot_model', return_value=None)

        one_dim_pane.create_artists_from_current_models()

        assert 'not compatible' in caplog.text
        assert model.id not in one_dim_pane.models

    def test_update_colors(self, one_dim_pane, multiorder_multiap_hdul):
        model = high_model.MultiOrder(multiorder_multiap_hdul)
        one_dim_pane.add_model(model)

        updates = one_dim_pane.update_colors()

        assert len(updates) == 76
        assert all([isinstance(u, drawing.Drawing) for u in updates])

    def test_get_axis_scale(self, one_dim_pane):
        output = one_dim_pane.get_axis_scale()
        assert len(output) == 3
        assert output == {'x': 'linear', 'y': 'linear', 'y_alt': None}

    def test_get_orders(self, one_dim_pane, multiorder_hdul_spec, grism_hdul):
        model = high_model.MultiOrder(multiorder_hdul_spec)
        one_dim_pane.add_model(model)
        # one_dim_pane.add_model(high_model.Grism(grism_hdul))

        orders = one_dim_pane.get_orders(model_id=model.id)
        assert isinstance(orders, list)
        assert orders == list(range(10))

        orders = one_dim_pane.get_orders(filename=model.filename,
                                         enabled_only=True)
        assert orders == list(range(10))

        orders = one_dim_pane.get_orders(filename='bad')
        assert len(orders) == 0

        orders = one_dim_pane.get_orders(filename=model.filename,
                                         kind='aperture')
        assert orders == list(range(1))

        orders = one_dim_pane.get_orders(filename=model.filename,
                                         kind='aperture', enabled_only=False)
        assert orders == list(range(1))

    def test_set_legend(self, one_dim_pane):
        with pytest.raises(NotImplementedError):
            one_dim_pane.set_legend()

    def test_convert_low_model_units(self, one_dim_pane, grism_hdul, mocker):
        model = high_model.Grism(grism_hdul)
        one_dim_pane.add_model(model)

        with pytest.raises(eye_error.EyeError):
            one_dim_pane._convert_low_model_units(model, 2, 'x', '', 9)

        one_dim_pane.fields['y'] = 'transmission'
        mock = mocker.patch.object(low_model.Spectrum, 'convert')
        one_dim_pane._convert_low_model_units(model, 0, 'y', '', 0)
        assert mock.called_with('', None, None)

        mock.reset_mock()
        one_dim_pane.fields['y'] = 'spectral_flux'
        one_dim_pane._convert_low_model_units(model, 0, 'y', 'W / m2', 0)
        assert mock.call_count == 2

    def test_set_fields(self, one_dim_pane, grism_hdul):
        model = high_model.Grism(grism_hdul)
        one_dim_pane.add_model(model)

        fields = {'x': 'transmission', 'y': 'response', 'z': 'wavepos'}
        one_dim_pane.set_fields(fields)

        assert one_dim_pane.fields['x'] == 'transmission'
        assert one_dim_pane.fields['y'] == 'response'
        assert one_dim_pane.fields['y_alt'] is None
        assert 'z' not in one_dim_pane.fields.keys()

    def test_set_default_units_for_fields(self, one_dim_pane, grism_hdul):
        model = high_model.Grism(grism_hdul)
        one_dim_pane.add_model(model)

        one_dim_pane.units = {'x': 'nm', 'y': 'W / m2'}
        one_dim_pane.set_default_units_for_fields()
        assert one_dim_pane.units['x'] == 'um'
        assert one_dim_pane.units['y'] == 'Jy'

    def test_set_plot_type(self, one_dim_pane, grism_hdul):
        model = high_model.Grism(grism_hdul)
        one_dim_pane.add_model(model)

        assert one_dim_pane.plot_type == 'step'
        results = one_dim_pane.set_plot_type('scatter')
        assert one_dim_pane.plot_type == 'scatter'
        assert isinstance(results, list)
        assert isinstance(results[0], drawing.Drawing)
        assert all([result.axes == 'primary' for result in results])

    def test_set_markers(self, one_dim_pane, grism_hdul):
        model = high_model.Grism(grism_hdul)
        one_dim_pane.add_model(model)

        assert not one_dim_pane.show_markers
        assert not one_dim_pane.show_overplot
        results = one_dim_pane.set_markers(True)
        assert one_dim_pane.show_markers
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], drawing.Drawing)

        one_dim_pane.show_overplot = True
        results = one_dim_pane.set_markers(False)
        assert not one_dim_pane.show_markers
        assert len(results) == 1
        assert all([result.axes == 'primary' for result in results])

        one_dim_pane.plot_type = 'scatter'
        results = one_dim_pane.set_markers(True)
        assert one_dim_pane.show_markers
        assert len(results) == 0

    def test_get_marker(self, one_dim_pane, grism_hdul):
        model = high_model.Grism(grism_hdul)
        one_dim_pane.add_model(model)

        result = one_dim_pane.get_marker(model.id)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result == ['x']

    def test_get_color(self, one_dim_pane, grism_hdul):
        model = high_model.Grism(grism_hdul)
        one_dim_pane.add_model(model)

        result = one_dim_pane.get_color(model.id)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result == [['#2848ad']]

    def test_set_grid(self, one_dim_pane, grism_hdul):
        model = high_model.Grism(grism_hdul)
        one_dim_pane.add_model(model)

        assert not one_dim_pane.show_grid
        one_dim_pane.set_grid(True)
        assert one_dim_pane.show_grid

    def test_set_error(self, one_dim_pane):
        assert one_dim_pane.show_error
        one_dim_pane.set_error(False)
        assert not one_dim_pane.show_error

    def test_set_overplot(self, one_dim_pane):
        assert not one_dim_pane.show_overplot
        assert one_dim_pane.ax_alt is None

        one_dim_pane.set_overplot(True)
        assert one_dim_pane.show_overplot
        assert isinstance(one_dim_pane.ax_alt, mpa.Axes)
        assert one_dim_pane.fields['y_alt'] == 'transmission'
        assert one_dim_pane.scale['y_alt'] == 'linear'
        assert one_dim_pane.limits['y_alt'] == [0, 1]

        one_dim_pane.set_overplot(False)
        assert not one_dim_pane.show_overplot
        assert isinstance(one_dim_pane.ax_alt, mpa.Axes)
        assert one_dim_pane.fields['y_alt'] == ''
        assert one_dim_pane.scale['y_alt'] == ''
        assert one_dim_pane.limits['y_alt'] == []

    def test_reset_alt_axes(self, one_dim_pane, caplog):
        caplog.set_level(logging.DEBUG)
        one_dim_pane.reset_alt_axes(True)
        assert 'Failed to remove alt ax' in caplog.text

        one_dim_pane.set_overplot(True)
        assert isinstance(one_dim_pane.ax_alt, mpa.Axes)
        one_dim_pane.reset_alt_axes(True)
        assert 'Successfully removed alt ax' in caplog.text
        assert one_dim_pane.ax_alt is None

    def test_plot_crosshair(self, one_dim_pane):
        results = one_dim_pane.plot_crosshair()
        for result in results:
            assert isinstance(result, drawing.Drawing)
            assert result.kind == 'crosshair'
        assert results[0].mid_model == 'vertical'
        assert results[1].mid_model == 'horizontal'

    @pytest.mark.parametrize('kind,v_num,h_num',
                             [('vertical', 1, 0), ('horizontal', 0, 1),
                              ('cross', 1, 1), ('x', 1, 0), ('y', 0, 1),
                              ('b', 1, 1,), ('other', 0, 0)])
    def test_plot_guides(self, one_dim_pane, kind, v_num, h_num):
        points = [10, 15]
        guides = one_dim_pane.plot_guides(points, kind)
        assert all([isinstance(g, drawing.Drawing) for g in guides])
        assert len(guides) == v_num + h_num
        assert sum([g.mid_model == 'vertical' for g in guides]) == v_num
        assert sum([g.mid_model == 'horizontal' for g in guides]) == h_num
        assert all([g.kind == 'guide' for g in guides])

    @pytest.mark.parametrize('direction', ['x', 'y', 'b', 'a'])
    def test_perform_zoom(self, one_dim_pane, direction, caplog, mocker):
        caplog.set_level(logging.DEBUG)
        mock = mocker.patch.object(one_dim_pane, 'set_limits')
        one_dim_pane.set_overplot(True)
        points = [[1, 2], [3, 4]]

        limits = one_dim_pane.perform_zoom(points, direction)
        assert 'Changing axis limits' in caplog.text
        assert mock.called_with(limits)
        if direction in ['x', 'b']:
            assert 'Updating x limits' in caplog.text
        if direction in ['y', 'b']:
            assert 'Updating y limits' in caplog.text

    def test_calculate_fit(self, one_dim_pane, mocker):
        mock = mocker.patch.object(sco, 'curve_fit',
                                   side_effect=ValueError('bound'))
        fit = am.models.Gaussian1D(1.2, 0.9, 0.5)
        result = one_dim_pane.calculate_fit([], [], fit, None)
        assert result == fit

        mock.side_effect = ValueError('other')
        with pytest.raises(eye_error.EyeError):
            one_dim_pane.calculate_fit([], [], fit, None)

        mock.side_effect = RuntimeError
        with pytest.raises(eye_error.EyeError):
            one_dim_pane.calculate_fit([], [], fit, None)

        param = (1, 1, 1)
        x = np.arange(10)
        y = np.arange(10)
        bound = [2, 6]
        mock.reset_mock()
        mock = mocker.patch.object(sco, 'curve_fit',
                                   return_value=(param, None))
        result = one_dim_pane.calculate_fit(x, y, fit, bound)
        assert isinstance(result, am.Model)
        npt.assert_allclose(result.parameters, param)

    def test_plot_fit(self, one_dim_pane, grism_hdul):
        model = high_model.HighModel(grism_hdul)
        fit_obj = model_fit.ModelFit()
        fit_obj.model_id = model.id
        fit_obj.order = 0
        fit_obj.aperture = 0
        fit_obj.feature = 'gaussian'
        fit_obj.fit = am.models.Gaussian1D(1.2, 0.9, 0.5)

        one_dim_pane.colors = {model.id: '#340011'}

        x = np.arange(5)
        style = 'solid'
        line, vline = one_dim_pane.plot_fit(x, style, fit_obj=fit_obj)

        assert isinstance(line, ml.Line2D)
        assert isinstance(vline, ml.Line2D)


class TestTwoDimPane(object):

    def test_init(self):
        p = pane.TwoDimPane()
        assert isinstance(p, pane.TwoDimPane)
