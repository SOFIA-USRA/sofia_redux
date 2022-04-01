import logging

import pytest
import numpy as np
import numpy.testing as npt
import matplotlib.lines as ml
import matplotlib.collections as mc
from matplotlib import colors

from sofia_redux.visualization.display import drawing


class TestDrawing(object):

    def test_eq(self):
        d1 = drawing.Drawing()
        d2 = drawing.Drawing()
        assert d1 != 'test'
        assert d1 == d1
        assert d1 == d2
        d2._data_id = 'test'
        assert d1 != d2

    def test_type_error(self):
        type_ = 'str'
        field = 'test'
        with pytest.raises(TypeError) as msg:
            drawing.Drawing._type_error(type_, field)
        for s in ['Improper type', type_, field]:
            assert s in str(msg)

    def test_value_error(self):
        type_ = 'str'
        field = 'test'
        with pytest.raises(ValueError) as msg:
            drawing.Drawing._value_error(type_, field)
        for s in ['Invalid value', type_, field]:
            assert s in str(msg)

    @pytest.mark.parametrize('kind,col,vis,mark',
                             [('line', True, True, True),
                              ('cursor', True, False, False),
                              ('fit', True, True, False),
                              ('ref', True, True, False),
                              ('border', True, True, False),
                              ('error_range', True, True, False),
                              ('patch', True, True, False),
                              ('default', True, True, True)])
    def test_update_options(self, line, kind, col, vis, mark):
        settings = {'color': 'blue', 'visible': True, 'marker': 's'}
        updates = {'color': 'red', 'visible': False, 'marker': 'x'}
        initial = drawing.Drawing(artist=line, **settings)
        other = drawing.Drawing(updates=updates)

        result = initial.update_options(other, kind=kind)

        assert result is True
        if col:
            assert initial.color == updates['color']
        else:
            assert initial.color == settings['color']
        if vis:
            assert initial.visible == updates['visible']
        else:
            assert initial.visible == settings['visible']
        if mark:
            assert initial.marker == updates['marker']
        else:
            assert initial.marker == settings['marker']

    def test_update_options_error(self):
        draw = drawing.Drawing()
        other = 'test'
        with pytest.raises(RuntimeError) as err:
            draw.update_options(other)
        assert 'Can only update with another Drawing' in str(err)

        # no updates to make
        other = drawing.Drawing()
        updated = draw.update_options(other, 'test')
        assert not updated

    def test_clear_updates(self, line_draw):
        line_draw.updates = {'marker': 'x'}
        line_draw.clear_updates()
        assert isinstance(line_draw.updates, dict)
        assert len(line_draw.updates) == 0

    def test_populate_properties(self, line_draw):
        args = {'color': 'red', 'marker': 's', 'alpha': 0.1,
                'linestyle': '--', 'visible': False}
        props = line_draw.artist.properties()
        for field, value in args.items():
            assert props[field] != value

        line_draw.populate_properties(args)

        props = line_draw.artist.properties()
        for field, value in args.items():
            assert props[field] == value

    def test_matches(self, one_dim_pane):
        props = {'high_model': 'one', 'mid_model': 'a',
                 'kind': 'line', 'data_id': 0, 'pane': one_dim_pane,
                 'axes': 'primary'}

        x = np.arange(1, 10, 1)
        y = x + 2
        line1 = one_dim_pane.ax.plot(x, y)[0]
        line2 = one_dim_pane.ax.plot(x, y)[0]

        draw1 = drawing.Drawing(color='blue', artist=line1,
                                fields={'x': 'wavepos', 'y': 'flux'},
                                **props)
        draw2 = drawing.Drawing(color='red', artist=line2,
                                fields={'x': 'wavepos', 'y': 'flux'},
                                **props)
        assert draw1.matches(draw2)
        assert draw1.matches(draw2, strict=True)
        draw2.set_fields(['wavepos', 'transmission'])
        assert draw1.matches(draw2)
        assert not draw1.matches(draw2, strict=True)

    @pytest.mark.parametrize('own,other,result', [('test', 'test', True),
                                                  ('test', 'TEST', True),
                                                  ('true', True, True),
                                                  ('all test', 'test', True),
                                                  ('test', 'all test', False),
                                                  ('test', 'wrong', False)])
    def test_match_high_model(self, own, other, result):
        draw = drawing.Drawing(high_model=own)
        output = draw.match_high_model(other)
        assert output is result

    @pytest.mark.parametrize('own,other,result', [('test', 'test', True),
                                                  ('test', 'TEST', True),
                                                  ('true', True, True),
                                                  ('all test', 'test', True),
                                                  ('test', 'all test', False),
                                                  ('test', 'wrong', False)])
    def test_match_mid_model(self, own, other, result):
        draw = drawing.Drawing(mid_model=own)
        output = draw.match_mid_model(other)
        assert output is result

    @pytest.mark.parametrize('own,other,result', [('test', 'test', True),
                                                  ('test', 'TEST', True),
                                                  ('true', True, True),
                                                  ('all test', 'test', False),
                                                  ('test', 'all test', False),
                                                  ('test', 'wrong', False)])
    def test_match_data_id(self, own, other, result):
        draw = drawing.Drawing(data_id=own)
        output = draw.match_data_id(other)
        assert output is result

    @pytest.mark.parametrize('own,other,result',
                             [('line', 'line', True),
                              ('line', 'ref_ling', False),
                              ('fit_line', 'fit_center', True),
                              ('error_range', 'errors', True),
                              ('fit', 'error', False)])
    def test_match_kind(self, own, other, result):
        draw = drawing.Drawing(kind=own)
        output = draw.match_kind(other)
        assert output is result

    @pytest.mark.parametrize('other,result',
                             [({'x': 'wavepos', 'y': 'flux'}, True),
                              ({'x': 'wavepos', 'y': 'error'}, False),
                              ({'x': 'wavepos', 'y_alt': 'transmission'},
                               True),
                              ({'x': 'wavepos', 'z': 'flux'}, False),
                              (['wavepos', 'flux'], True),
                              (['flux', 'wavepos'], False),
                              (['wavepos', 'flux', 'transmission'], True),
                              (['wavepos', 'flux', 'error'], False)])
    def test_match_fields(self, other, result):
        own = {'x': 'wavepos', 'y': 'flux', 'y_alt': 'transmission'}
        draw = drawing.Drawing(fields=own)
        output = draw.match_fields(other)
        assert output is result

    def test_match_axes(self, fig):
        ax = fig.add_subplot()
        draw = drawing.Drawing(axes=ax)
        assert draw.match_axes(ax)

        other = fig.add_subplot()
        assert not draw.match_axes(other)

    def test_match_text(self, one_dim_pane, patch):
        text = one_dim_pane.ax.text(1, 1, 'test')
        draw = drawing.Drawing()
        draw.artist = text
        assert draw.match_text(text)

        # mismatched text
        t2 = one_dim_pane.ax.text(1, 1, 'test')
        t2.set_text('other')
        assert not draw.match_text(t2)

        # mismatched artist
        assert not draw.match_text(patch)

        # inappropriate artist
        draw.artist = patch
        assert not draw.match_text(patch)

    def test_apply_updates(self, mocker):
        mock = mocker.patch.object(drawing.Drawing, 'set_data')
        draw = drawing.Drawing()
        updates = {'color': 'red'}
        draw.apply_updates(updates)

        assert mock.call_count == 1
        assert mock.called_with(updates)

    def test_set_data_lines(self, mocker, caplog):
        caplog.set_level(logging.DEBUG)
        line_mock = mocker.patch.object(drawing.Drawing, '_set_line_data')
        scatter_mock = mocker.patch.object(drawing.Drawing,
                                           '_set_scatter_data')

        artist = ml.Line2D([1], [2])
        draw = drawing.Drawing(artist=artist)

        update = {'x_data': [3], 'y_data': [5]}
        update_draw = drawing.Drawing(updates=update)
        draw.set_data(update=update_draw)
        assert line_mock.call_count == 1
        assert scatter_mock.call_count == 0
        assert line_mock.called_with({'data': update['x_data'], 'artist': None,
                                      'axis': None})

        update = {'z_data': [3], 'y_data': [5]}
        line_mock.reset_mock()
        scatter_mock.reset_mock()

        update_draw = drawing.Drawing(updates=update)
        draw.set_data(update=update_draw)
        assert line_mock.call_count == 1
        assert scatter_mock.call_count == 0
        assert line_mock.called_with({'data': update['y_data'], 'artist': None,
                                      'axis': None})

        update = {'artist': ml.Line2D([3], [4])}
        line_mock.reset_mock()
        scatter_mock.reset_mock()

        update_draw = drawing.Drawing(updates=update)
        draw.set_data(update=update_draw)
        assert line_mock.call_count == 1
        assert scatter_mock.call_count == 0
        assert line_mock.called_with({'data': None, 'artist': update['artist'],
                                      'axis': None})

    def test_set_data_scatter(self, mocker, caplog, one_dim_pane):
        x = np.arange(1, 10, 1)
        y = x + 2
        scatter = one_dim_pane.ax.scatter(x, y)

        caplog.set_level(logging.DEBUG)
        line_mock = mocker.patch.object(drawing.Drawing, '_set_line_data')
        scatter_mock = mocker.patch.object(drawing.Drawing,
                                           '_set_scatter_data')

        draw = drawing.Drawing(artist=scatter)

        update = {'x_data': [3], 'y_data': [5]}
        update_draw = drawing.Drawing(updates=update)
        draw.set_data(update=update_draw)
        assert line_mock.call_count == 0
        assert scatter_mock.call_count == 1
        assert line_mock.called_with({'data': update['x_data'], 'artist': None,
                                      'axis': None})

        update = {'z_data': [3], 'y_data': [5]}
        line_mock.reset_mock()
        scatter_mock.reset_mock()

        update_draw = drawing.Drawing(updates=update)
        draw.set_data(update=update_draw)
        assert line_mock.call_count == 0
        assert scatter_mock.call_count == 1
        assert line_mock.called_with({'data': update['y_data'], 'artist': None,
                                      'axis': None})

        scatter2 = one_dim_pane.ax.scatter(x, y)
        update = {'artist': scatter2}
        line_mock.reset_mock()
        scatter_mock.reset_mock()

        update_draw = drawing.Drawing(updates=update)
        draw.set_data(update=update_draw)
        assert line_mock.call_count == 0
        assert scatter_mock.call_count == 1
        assert line_mock.called_with({'data': None, 'artist': update['artist'],
                                      'axis': None})

    def test_set_data_patch(self, caplog, fill, mocker):
        line_mock = mocker.patch.object(drawing.Drawing, '_set_line_data')
        scatter_mock = mocker.patch.object(drawing.Drawing,
                                           '_set_scatter_data')
        caplog.set_level(logging.DEBUG)

        draw = drawing.Drawing(artist=fill)
        draw.set_data()

        assert line_mock.call_count == 0
        assert scatter_mock.call_count == 0
        assert 'Unable to process' in caplog.text

    def test_set_line_data(self, patch):
        x, y = [2], [3]
        draw = drawing.Drawing(artist=ml.Line2D(x, y))
        update = [10]
        draw._set_line_data(data=update, axis='x')
        assert draw.get_artist().get_xdata() == update
        assert draw.get_artist().get_ydata() == y

        update = [[30], [20]]
        new_artist = ml.Line2D(update[0], update[1])
        draw._set_line_data(artist=new_artist)
        assert draw.get_artist().get_xdata() == new_artist.get_xdata()
        assert draw.get_artist().get_ydata() == new_artist.get_ydata()

        # no op if attribute error
        draw._set_line_data(artist=patch)
        assert draw.get_artist().get_xdata() == new_artist.get_xdata()
        assert draw.get_artist().get_ydata() == new_artist.get_ydata()

    def test_set_scatter_data(self, fig, ax):
        points = np.array([[1, 2], [3, 4], [5, 6]])
        factor = 2
        artist = ax.scatter(points[:, 0], points[:, 1])

        draw = drawing.Drawing(artist=artist)
        update = {'x': points[:, 0] * factor,
                  'y': points[:, 1] * factor}

        for axis, data in update.items():
            draw._set_scatter_data(data=data, axis=axis)
        new_data = draw.get_artist().get_offsets()
        assert np.all(new_data / points == factor)

        new_artist = ax.scatter(points[:, 0], points[:, 1])
        draw._set_scatter_data(artist=new_artist)
        new_data_2 = draw.get_artist().get_offsets()
        assert np.all(new_data_2 / points == 1)

        # bad axis value sets both x and y, like all
        draw._set_scatter_data(data=update, axis='bad')
        assert np.all(new_data_2 / points == 1)

    @pytest.mark.parametrize('marker,correct', [('x', 'x'), (None, 'o')])
    def test_convert_to_scatter(self, marker, correct, line, fig):
        ax = fig.add_subplot()
        correct_scatter = ax.scatter([0], [1], marker=correct)

        scatter = drawing.Drawing.convert_to_scatter(line, marker)
        assert isinstance(scatter, mc.PathCollection)
        assert scatter.get_label() == line.get_label()
        assert scatter.axes == line.axes
        assert all([colors.to_hex(fc) == line.get_color()
                    for fc in scatter.get_facecolor()])
        npt.assert_array_equal(scatter.get_offsets().T,
                               line.get_data())
        npt.assert_array_equal(scatter.get_paths()[0].vertices,
                               correct_scatter.get_paths()[0].vertices)

    @pytest.mark.parametrize('style,drawstyle', [('line', 'default'),
                                                 ('step', 'steps-mid')])
    def test_convert_to_line(self, style, drawstyle, fig):
        face = 'black'
        edge = 'red'
        ax = fig.add_subplot()

        scatter = ax.scatter([1, 2, 3], [1, 2, 3], facecolor=face,
                             edgecolors=edge)

        draw = drawing.Drawing(artist=scatter)
        draw.convert_to_line(drawstyle=style, marker='x')

        assert isinstance(draw.get_artist(), ml.Line2D)
        assert draw.get_artist().get_drawstyle() == drawstyle
        npt.assert_array_equal(draw.get_artist().get_color(),
                               colors.to_rgba(face))

    def test_in_pane(self, line, one_dim_pane):
        draw = drawing.Drawing(artist=line, pane=one_dim_pane)
        result = draw.in_pane(one_dim_pane)
        assert result is True

        result = draw.in_pane(one_dim_pane, alt=True)
        assert result is True

    def test_in_pane_alt(self, line_alt, one_dim_pane):
        draw = drawing.Drawing(artist=line_alt, pane=one_dim_pane)
        result = draw.in_pane(one_dim_pane)
        assert result is False

        result = draw.in_pane(one_dim_pane, alt=True)
        assert result is True

    @pytest.mark.parametrize('input,output', [('pri', 'primary'),
                                              ('p', 'primary'),
                                              ('primary', 'primary'),
                                              ('sec', 'alt'),
                                              ('s', 'alt'),
                                              ('secondary', 'alt'),
                                              ('alt', 'alt'),
                                              ('alternate', 'alt')])
    def test_axes_setting(self, input, output):
        draw = drawing.Drawing()
        assert draw.axes == 'primary'
        draw.axes = input
        assert draw.axes == output
        draw.set_axes(input)
        assert draw.get_axes() == output

    def test_axes_setting_error(self):
        draw = drawing.Drawing()
        assert draw.axes == 'primary'
        with pytest.raises(ValueError) as msg:
            draw.axes = 'main'
        assert draw.axes == 'primary'
        assert 'Invalid value' in str(msg)

        with pytest.raises(TypeError) as msg:
            draw.axes = 0
        assert draw.axes == 'primary'
        assert 'Improper type' in str(msg)

    def test_artist_setting(self, line, patch):
        draw = drawing.Drawing()

        # artist is okay
        draw.artist = line
        assert draw.artist is line

        # other types raise error
        with pytest.raises(TypeError) as msg:
            draw.artist = 'test'
        assert draw.artist is line
        assert 'Improper type' in str(msg)

        # getter/setter
        draw.set_artist(patch)
        assert draw.get_artist() is patch

    def test_updates_setting(self):
        draw = drawing.Drawing()

        # direct property get/set
        update = {'test': 1}
        draw.updates = update
        assert draw.updates is update

        # getter/setter
        draw.set_updates(update)
        assert draw.get_updates() is update

        # other types raise errors
        with pytest.raises(TypeError) as msg:
            draw.updates = 'test'
        assert draw.updates is update
        assert 'Improper type' in str(msg)

    def test_fields_setting(self):
        draw = drawing.Drawing()
        assert draw.fields == dict()

        # dict input: used directly
        update = {'test': 1}
        draw.fields = update
        assert draw.fields == update

        with pytest.raises(TypeError) as msg:
            draw.fields = 'test'
        assert draw.fields == update
        assert 'Improper type' in str(msg)

        # list input: assumed x, y, alt
        # added to initial dict
        draw.fields = ['1', '2', '3']
        assert draw.fields == {'x': '1', 'y': '2', 'y_alt': '3', 'test': 1}

    @pytest.mark.parametrize('value', ['high_model', 'mid_model',
                                       'kind', 'data_id', 'state', 'pane',
                                       'label'])
    def test_simple_setting(self, value):
        draw = drawing.Drawing()

        # direct property setting
        setattr(draw, value, 'test')
        assert getattr(draw, value) == 'test'

        # get/set function setting
        set_func = getattr(draw, f'set_{value}')
        get_func = getattr(draw, f'get_{value}')
        set_func('test')
        assert get_func() == 'test'

    @pytest.mark.parametrize('value', ['update'])
    def test_bool_setting(self, value):
        draw = drawing.Drawing()
        setattr(draw, value, True)
        assert getattr(draw, value) is True

        set_func = getattr(draw, f'set_{value}')
        get_func = getattr(draw, f'get_{value}')
        set_func(False)
        assert get_func() is False

    @pytest.mark.parametrize('key,value,error',
                             [('color', 'blue', 'test'),
                              ('marker', '.', None),
                              ('visible', True, None)])
    def test_needs_artist_setting(self, line, key, value, error):
        draw = drawing.Drawing()
        # no artist
        assert getattr(draw, key) is None
        setattr(draw, key, value)
        assert getattr(draw, key) is None

        # with line artist
        draw.artist = line
        setattr(draw, key, value)
        assert getattr(draw, key) == value

        # with bad value
        if error is not None:
            with pytest.raises(Exception):
                setattr(draw, key, error)

        # getter/setter
        set_func = getattr(draw, f'set_{key}')
        get_func = getattr(draw, f'get_{key}')
        set_func(value)
        assert get_func() == value

    def test_set_animated(self, line):
        draw = drawing.Drawing()
        assert draw.get_animated() is None

        draw = drawing.Drawing(artist=line)
        assert not draw.get_animated()

        draw.set_animated(True)
        assert draw.get_animated()

        draw.set_animated(False)
        assert not draw.get_animated()

    def test_set_visible(self, line):
        draw = drawing.Drawing()
        assert draw.visible is None

        draw = drawing.Drawing(artist=line)
        assert draw.visible

        draw.visible = False
        assert not draw.visible

    def test_set_marker(self, line, fill):
        draw = drawing.Drawing()
        assert draw.marker is None

        draw = drawing.Drawing(artist=fill)
        assert draw.marker is None
        draw.marker = 'x'

        draw = drawing.Drawing(artist=line)
        assert draw.marker == 'None'
        draw.marker = 'x'
        assert draw.marker == 'x'

    def test_remove(self, line):
        draw = drawing.Drawing(artist=line)
        assert isinstance(draw.artist, ml.Line2D)
        assert line.axes is not None

        draw.remove()

        assert draw.artist is None
        assert line.axes is None
