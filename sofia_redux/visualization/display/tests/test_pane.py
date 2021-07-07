#  Licensed under a 3-clause BSD style license - see LICENSE.rst


import pytest
import logging
import astropy.units as u
import numpy as np
from scipy import optimize as sco
from matplotlib import figure as mpf
from matplotlib import axes as mpa
from matplotlib import lines as ml
from matplotlib import backend_bases as mpb
from matplotlib import artist as mart

from sofia_redux.visualization.display import pane
from sofia_redux.visualization.models import high_model
from sofia_redux.visualization.utils import eye_error

PyQt5 = pytest.importorskip('PyQt5')


class TestPane(object):

    def test_init(self):
        pass

    def test_set_border_visibility(self, blank_pane):
        blank_pane.set_border_visibility(True)

        blank_pane.border = mart.Artist()
        assert blank_pane.border.get_visible()

        blank_pane.set_border_visibility(False)
        assert not blank_pane.border.get_visible()

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
                            return_value='line')

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

        assert all([isinstance(ln['line']['artist'], ml.Line2D)
                    for ln in lines.values()])
        assert all([ln['line']['artist'].get_marker() == marker
                    for ln in lines.values()])
        assert all([ln['line']['artist'].get_linestyle() == 'None'
                    for ln in lines.values()])
        assert all([not ln['line']['artist'].get_visible()
                    for ln in lines.values()])

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

        updates, err_update = blank_onedim.set_units(units, 'primary')

        assert not blank_onedim.data_changed
        assert len(err_update) == 0
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
            # empty/failed fit still returns parameters but not artists
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
        assert updates[0]['new_artist'].get_visible()

        # disable model: error artist should not be visible
        grism_model.enabled = False
        updates = blank_onedim._update_error_artists()
        assert len(updates) == 1
        assert not updates[0]['new_artist'].get_visible()

        # same for show_error = False
        grism_model.enabled = True
        blank_onedim.show_error = False
        updates = blank_onedim._update_error_artists()
        assert len(updates) == 1
        assert not updates[0]['new_artist'].get_visible()

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
