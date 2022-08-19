#  Licensed under a 3-clause BSD style license - see LICENSE.rst
import matplotlib.collections
import pytest
import logging
import astropy.units as u
import numpy as np
from scipy import optimize as sco
from matplotlib import figure as mpf
from matplotlib import axes as mpa
from matplotlib import lines as ml
from matplotlib import collections as mc
from matplotlib import backend_bases as mpb
from matplotlib import artist as mart

from sofia_redux.visualization.display import pane, drawing
from sofia_redux.visualization.models import high_model, reference_model
from sofia_redux.visualization.utils import eye_error
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

    def test_update_model(self, multiorder_hdul_merged,
                          blank_onedim, grism_hdul):
        grism_filename = grism_hdul.filename()
        grism_model = high_model.Grism(grism_hdul)
        blank_onedim.models[grism_filename] = grism_model

        multi_model = dict()
        multi_model[grism_filename] = high_model.MultiOrder(
            multiorder_hdul_merged)

        blank_onedim.update_model(multi_model)
        assert id(blank_onedim.models[grism_filename]) != \
               id(high_model.MultiOrder(multiorder_hdul_merged))

    def test_remove_model(self, blank_onedim, mocker,
                          grism_hdul, multiorder_hdul_merged):
        multi_filename = multiorder_hdul_merged.filename()
        multi_model = high_model.MultiOrder(multiorder_hdul_merged)
        blank_onedim.models[multi_filename] = multi_model

        grism_filename = grism_hdul.filename()
        grism_model = high_model.Grism(grism_hdul)
        blank_onedim.models[grism_filename] = grism_model

        assert len(blank_onedim.models) == 2
        blank_onedim.remove_model(filename=grism_filename)
        assert len(blank_onedim.models) == 1

        blank_onedim.models[grism_filename] = grism_model
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

        assert not blank_onedim.data_changed
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

    def test_plot_model_same_fields(self, blank_onedim, mocker, grism_hdul):

        marker = '^'
        ax = mpf.Figure().subplots(1, 1)
        blank_onedim.ax = ax

        model = high_model.Grism(grism_hdul)
        model.enabled = False
        blank_onedim.models[model.filename] = model
        blank_onedim.orders[model.filename] = [0]
        blank_onedim.markers[model.filename] = marker
        blank_onedim.colors[model.filename] = 'blue'

        # Same fields
        blank_onedim.fields['y'] = blank_onedim.fields['x']
        blank_onedim.units = {'x': u.um, 'y': u.nm}

        blank_onedim.plot_type = 'scatter'
        lines = blank_onedim._plot_model(model)
        line, cursor = tuple(lines)
        # Being tested for fields
        assert line.get_artist()._label.split(',')[-1].strip() == \
               blank_onedim.fields['x']
        assert isinstance(line.get_artist(), ml.Line2D)
        assert line.get_artist().get_marker() == marker
        assert line.get_artist().get_linestyle() == 'None'
        assert isinstance(cursor.get_artist(), mc.PathCollection)
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
        blank_onedim.models[model.filename] = model
        blank_onedim.orders[model.filename] = [0]

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

        blank_onedim.models[model.filename] = model
        blank_onedim.orders[model.filename] = [0]
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

        blank_onedim.models[model.filename] = model
        blank_onedim.orders[model.filename] = [0]
        blank_onedim.markers[model.filename] = marker
        blank_onedim.colors[model.filename] = 'blue'
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
        blank_onedim.models[model.filename] = model
        blank_onedim.orders[model.filename] = [0]
        blank_onedim.markers[model.filename] = marker
        blank_onedim.colors[model.filename] = 'blue'

        blank_onedim.plot_type = 'scatter'
        lines = blank_onedim._plot_model(model)
        line, cursor, error = tuple(lines)

        assert isinstance(line.get_artist(), ml.Line2D)
        assert line.get_artist().get_marker() == marker
        assert line.get_artist().get_linestyle() == 'None'
        assert isinstance(cursor.get_artist(), mc.PathCollection)
        assert isinstance(error.get_artist(), mc.PolyCollection)
        assert all([not ln.get_artist().get_visible()
                    for ln in lines])

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

    def test_get_orders(self, blank_onedim, grism_hdul):
        model = high_model.Grism(grism_hdul)

        blank_onedim.models[model.filename] = model

        result = blank_onedim.get_orders(enabled_only=True, by_model=True)

        assert result[model.filename] == [0]

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
        x, y = blank_onedim.get_xy_data(model, order=0)

        assert id(x) != id(model)
        assert id(y) != id(model)
        assert id(x) != id(y)

        spectrum_x = x.retrieve(order=0,
                                level='low', field='wavepos')
        spectrum_y = y.retrieve(order=0,
                                level='low', field='wavepos')
        assert spectrum_x.unit_key == 'um'
        assert spectrum_y.unit_key == 'nm'

    @pytest.mark.parametrize('target, current',
                             [({'x': u.nm, 'y': 'Jy'}, {'x': 'pixel',
                                                        'y': u.Jy}),
                              ({'x': 'pixel', 'y': u.mol}, {'x': 'pixel',
                                                            'y': u.Jy})])
    def test_set_units_pixels(self, qtbot, mocker, blank_onedim, grism_hdul,
                              target, current):
        mocker.patch.object(pane.OneDimPane, '_convert_low_model_units',
                            side_effect=ValueError)
        model = high_model.Grism(grism_hdul)
        blank_onedim.models[model.filename] = model
        blank_onedim.orders[model.filename] = [0]
        blank_onedim.units = current
        units = target
        with qtbot.wait_signal(blank_onedim.signals.obtain_raw_model):
            blank_onedim.set_units(units, 'primary')

    def test_set_units_err_none(self, qtbot, mocker, blank_onedim, grism_hdul):
        mocker.patch.object(pane.OneDimPane, '_convert_low_model_units',
                            side_effect=ValueError)
        model = high_model.Grism(grism_hdul)

        spectrum = model.retrieve(order=0,
                                  level='low', field='flux')
        wave_spectrum = model.retrieve(order=0,
                                       level='low', field='wavepos')
        blank_onedim.models[model.filename] = model

        mocker.patch.object(model, 'retrieve',
                            side_effect=[spectrum, wave_spectrum, None])

        # mock retrieve multiple times
        blank_onedim.orders[model.filename] = [0]
        blank_onedim.fields['y'] = 'flux'
        blank_onedim.fields['x'] = 'wavepos'
        blank_onedim.units = {'x': u.nm, 'y': u.Jy}
        units = {'x': u.m, 'y': u.mol}
        model = blank_onedim.models[model.filename]
        blank_onedim.set_units(units, 'primary')
        error_spectrum = model.retrieve(order=0, level='low',
                                        field='spectral_error')

        assert error_spectrum is None

    def test_update_visibility(self, mocker, blank_onedim,
                               grism_hdul):

        model = high_model.Grism(grism_hdul)

        spectrum = model.retrieve(order=0,
                                  level='low', field='flux')
        # wave_spectrum = model.retrieve(order=0,
        #                                level='low', field='wavepos')
        blank_onedim.models[model.filename] = model

        mocker.patch.object(model, 'retrieve',
                            side_effect=spectrum)
        blank_onedim.orders[model.filename] = [0]

        updates = blank_onedim.update_visibility(error=None)
        assert updates[0]._kind == 'line'

    def test_set_units_fail(self, blank_onedim, mocker, caplog, grism_hdul):
        caplog.set_level(logging.DEBUG)
        mocker.patch.object(pane.OneDimPane, '_convert_low_model_units',
                            side_effect=ValueError)
        model = high_model.Grism(grism_hdul)

        blank_onedim.models[model.filename] = model
        blank_onedim.orders[model.filename] = [0]
        blank_onedim.fields['y'] = 'response'
        blank_onedim.units = {'x': u.um, 'y': u.Jy}
        units = {'x': u.nm, 'y': u.mol}

        assert not blank_onedim.data_changed

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

        mocker.patch.object(np, 'nanargmin',
                            return_value=4)
        mocker.patch.object(np, 'isnan', return_value=np.array([isnan]))
        model = high_model.Grism(grism_hdul)

        blank_onedim.models[model.filename] = model
        blank_onedim.orders[model.filename] = [0]
        blank_onedim.colors[model.filename] = 'blue'

        # add an alt axis to test too
        blank_onedim.show_overplot = True
        blank_onedim.fields['y_alt'] = blank_onedim.fields['y']

        event = mpb.MouseEvent(x=2, y=3, canvas=blank_figure.widget.canvas,
                               name='motion_notify_event')
        event.xdata = xdata

        data = blank_onedim.data_at_cursor(event)
        assert data[model.filename][0]['visible'] is False
        assert data[model.filename][1]['visible'] is False

    @pytest.mark.parametrize('xdata,isnan', [(2000, False)])
    def test_data_at_cursor_same_fields(self, blank_onedim, mocker,
                                        grism_hdul, blank_figure, xdata,
                                        isnan):
        mocker.patch.object(np, 'nanargmin', return_value=4)
        mocker.patch.object(np, 'isnan', return_value=np.array([isnan]))
        model = high_model.Grism(grism_hdul)

        blank_onedim.models[model.filename] = model
        blank_onedim.orders[model.filename] = [0]
        blank_onedim.colors[model.filename] = 'blue'

        # add an alt axis to test too
        blank_onedim.show_overplot = False
        blank_onedim.fields['y'] = blank_onedim.fields['x']
        blank_onedim.units = {'x': u.nm, 'y': u.nm}

        event = mpb.MouseEvent(x=2, y=3, canvas=blank_figure.widget.canvas,
                               name='motion_notify_event')
        event.xdata = xdata

        data = blank_onedim.data_at_cursor(event)

        assert data[model.filename][0]['visible'] is False
        assert data[model.filename][0]['y_field'] == data[model.filename][
            0]['x_field']

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
        blank_onedim.models[model.filename] = model
        blank_onedim.orders[model.filename] = [0]
        blank_onedim.colors[model.filename] = 'blue'
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
            # empty/failed fit still returns parameters but not gallery
            assert len(params) == 1
        else:
            assert len(params) == 0
        assert '; skipping' in caplog.text

    def test_contains_model(self, blank_onedim, grism_hdul):
        grism_filename = grism_hdul.filename()
        grism_model = high_model.Grism(grism_hdul)
        blank_onedim.models[grism_filename] = grism_model
        blank_onedim.orders[grism_filename] = [0]

        assert blank_onedim.contains_model(grism_filename)
        assert blank_onedim.contains_model(grism_filename, order=0)

        assert not blank_onedim.contains_model('test.fits')
        assert not blank_onedim.contains_model(grism_filename, order=1)

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
        fn = grism_hdul.filename()
        grism_model = high_model.Grism(grism_hdul)
        blank_onedim.models[fn] = grism_model
        blank_onedim.orders[fn] = [0]
        blank_onedim.markers[fn] = 'x'
        blank_onedim.colors[fn] = 'blue'
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

        # plot true/false calls different functions
        m1 = mocker.patch.object(blank_onedim, '_plot_reference')
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
        fn = grism_hdul.filename()
        grism_model = high_model.Grism(grism_hdul)
        blank_onedim.models[fn] = grism_model
        blank_onedim.orders[fn] = [0]
        blank_onedim.markers[fn] = 'x'
        blank_onedim.colors[fn] = 'blue'
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
        # mocker.patch.object(pane.OneDimPane, 'get_axis_limits()',
        #                      side_effect=ValueError)
        fn = grism_hdul.filename()
        grism_model = high_model.Grism(grism_hdul)
        blank_onedim.models[fn] = grism_model
        blank_onedim.orders[fn] = [0]
        blank_onedim.units['x'] = 'pixel'
        # blank_onedim.markers[fn] = 'x'
        # blank_onedim.colors[fn] = 'blue'
        # ax = mpf.Figure().subplots(1, 1)
        # blank_onedim.ax = ax

        name_limits = blank_onedim._window_line_list()
        assert name_limits == dict()

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
        blank_onedim.models[model.filename] = model
        blank_onedim.orders[model.filename] = [0]
        blank_onedim.markers[model.filename] = 'x'
        blank_onedim.colors[model.filename] = 'blue'
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
