#  Licensed under a 3-clause BSD style license - see LICENSE.rst

import copy
import logging
import pytest
import numpy as np
import numpy.testing as npt

from matplotlib import artist as ma
from matplotlib import lines as ml
from matplotlib import collections as mc
from matplotlib import colors

from sofia_redux.visualization.display import artists, pane

PyQt5 = pytest.importorskip('PyQt5')


class TestArtists(object):

    def test_init(self):
        keys = ['line', 'cursor', 'error_range', 'crosshair',
                'guide', 'patch', 'fit', 'line_alt', 'cursor_alt']
        obj = artists.Artists()

        assert all([k in obj.arts.keys() for k in keys])
        assert all([isinstance(v, list) for v in obj.arts.values()])

    def test_add_patches(self):
        blank = ma.Artist()
        assert blank.get_visible()
        patches = {'one': {'kind': 'test',
                           'artist': blank,
                           'visible': False}}

        obj = artists.Artists()
        obj.add_patches(patches)

        assert obj.arts['patch'] == [{'model_id': 'test_one',
                                      'artist': blank, 'state': 'new'}]
        assert not obj.arts['patch'][0]['artist'].get_visible()

    def test_set_pane_highlight_flag(self):
        blank_1 = ma.Artist()
        blank_2 = ma.Artist()
        blank_3 = ma.Artist()
        obj = artists.Artists()

        obj.arts['patch'] = [{'model_id': 'border_pane_1',
                              'artist': blank_1},
                             {'model_id': 'border_pane_2',
                              'artist': blank_2},
                             {'model_id': 'test',
                              'artist': blank_3}]

        obj.set_pane_highlight_flag(pane_number=1, state=False)
        assert not obj.arts['patch'][0]['artist'].get_visible()
        assert not obj.arts['patch'][1]['artist'].get_visible()
        assert obj.arts['patch'][2]['artist'].get_visible()

    def test_add_crosshairs(self):
        blank = ma.Artist()
        assert blank.get_visible()
        crosshairs = {'one': {'kind': 'test',
                              'artist': blank,
                              'direction': 'v',
                              'visible': False}}

        obj = artists.Artists()
        obj.add_crosshairs(crosshairs)

        assert obj.arts['crosshair'] == [{'model_id': 'test_one',
                                          'direction': 'v', 'artist': blank,
                                          'state': 'new'}]
        assert not obj.arts['crosshair'][0]['artist'].get_visible()

    @pytest.mark.parametrize('direction,change_x,change_y',
                             [('v', True, False), ('h', False, True),
                              ('vh', True, True), ('', True, True)])
    def test_update_crosshair(self, direction, change_x, change_y):
        old_point = {'x': [6], 'y': [3]}
        new_point = {'x': [5], 'y': [2]}
        cross_one = {'1': {'kind': 'crosshair_pane',
                           'artist': ml.Line2D(old_point['x'], old_point['y']),
                           'direction': 'v',
                           'visible': False}}
        cross_two = {'1': {'kind': 'crosshair_pane',
                           'artist': ml.Line2D(old_point['x'], old_point['y']),
                           'direction': 'h',
                           'visible': False}}
        cross_three = {'2': {'kind': 'crosshair_pane',
                             'artist': ml.Line2D(old_point['x'],
                                                 old_point['y']),
                             'direction': 'h',
                             'visible': False}}

        obj = artists.Artists()
        obj.add_crosshairs(cross_one)
        obj.add_crosshairs(cross_two)
        obj.add_crosshairs(cross_three)

        obj.update_crosshair(pane_number=1, direction=direction,
                             data_point=(new_point['x'], new_point['y']))

        if change_x:
            target_x = new_point['x']
        else:
            target_x = old_point['x']
        if change_y:
            target_y = new_point['y']
        else:
            target_y = old_point['y']

        if 'v' in direction:
            assert obj.arts['crosshair'][0]['artist'].get_xdata() == target_x
        if 'h' in direction:
            assert obj.arts['crosshair'][1]['artist'].get_ydata() == target_y

    def test_hide_crosshair(self):
        old_point = {'x': [6], 'y': [3]}
        cross_one = {'1': {'kind': 'crosshair_pane',
                           'artist': ml.Line2D(old_point['x'], old_point['y']),
                           'direction': 'v',
                           'visible': True}}
        cross_two = {'1': {'kind': 'crosshair_pane',
                           'artist': ml.Line2D(old_point['x'], old_point['y']),
                           'direction': 'h',
                           'visible': True}}

        obj = artists.Artists()
        obj.add_crosshairs(cross_one)
        obj.add_crosshairs(cross_two)

        for crosshair in obj.arts['crosshair']:
            assert crosshair['artist'].get_visible()

        obj.hide_crosshair()

        for crosshair in obj.arts['crosshair']:
            assert not crosshair['artist'].get_visible()

    def test_update_line_data(self, mocker, capsys):
        line = ml.Line2D([2], [2])
        scatter = mc.PathCollection([2, 34])
        patch = mc.PatchCollection(list())
        lines = [{'model_id': 'line', 'order': 1, 'artist': line,
                  'fields': ['wavelength', 'flux']},
                 {'model_id': 'scatter', 'order': 1, 'artist': scatter,
                  'fields': ['wavelength', 'flux']},
                 {'model_id': 'scatter', 'order': 1, 'artist': patch,
                  'fields': ['wavelength', 'flux']}]
        updates = [{'model_id': 'line', 'order': 1, 'artist': ma.Artist(),
                   'field': 'flux'},
                   {'model_id': 'scatter', 'order': 1, 'artist': ma.Artist(),
                    'field': 'flux'},
                   {'model_id': 'patch', 'order': 1, 'artist': list(),
                    'field': 'flux'}]

        # returns same result for line and line_alt
        mocker.patch.object(artists.Artists, 'artists_in_pane',
                            return_value=lines)
        line_mock = mocker.patch.object(artists.Artists, '_set_line_data')
        path_mock = mocker.patch.object(artists.Artists, '_set_scatter_data')

        obj = artists.Artists()
        obj.update_line_data(pane=1, updates=updates, axes='both')
        captured = capsys.readouterr()

        assert line_mock.call_count == 2
        assert path_mock.call_count == 2
        assert 'Unable to process' in captured.out

    def test_set_line_data(self):
        x, y = [2], [3]
        line = {'artist': ml.Line2D(x, y)}

        update = {'new_x_data': [10], 'new_y_data': [20]}
        artists.Artists._set_line_data(line, update)
        assert line['artist'].get_xdata() == update['new_x_data']
        assert line['artist'].get_ydata() == y

        update = {'new_z_data': [10], 'new_y_data': [20]}
        line = {'artist': ml.Line2D(x, y)}
        artists.Artists._set_line_data(line, update)
        assert line['artist'].get_xdata() == x
        assert line['artist'].get_ydata() == update['new_y_data']

    def test_set_scatter_data(self, fig):
        points = np.array([[1, 2], [3, 4], [5, 6]])
        factor = 2
        line = {'artist': mc.PathCollection(points[:, 0], points[:, 1])}
        update = {'new_x_data': points[:, 0] * factor,
                  'new_y_data': points[:, 1] * factor}

        artists.Artists._set_scatter_data(line, update)
        new_data = line['artist'].get_offsets()
        assert np.all(new_data / points == factor)

        update = {'new_w_data': points[:, 0] * factor,
                  'new_z_data': points[:, 1] * factor}
        artists.Artists._set_scatter_data(line, update)
        new_data_2 = line['artist'].get_offsets()
        assert np.all(new_data_2 / new_data == 1)

    def test_line_fields(self, mocker):
        points = np.array([[1, 2], [3, 4], [5, 6]])
        factor = 2
        model_id = 'mid'
        order = 1
        line = {'artist': ml.Line2D(points[:, 0], points[:, 1],
                                    label='orig points'),
                'model_id': model_id, 'order': order,
                'state': 'stale'}
        mocker.patch.object(artists.Artists, 'artists_in_pane',
                            return_value=[line])

        update = {'new_x_data': points[:, 0] * factor,
                  'model_id': model_id, 'order': order,
                  'new_field': 'new'}
        obj = artists.Artists()
        obj.update_line_fields(1, [update])

        assert line['state'] == 'new'
        assert np.all(line['artist'].get_xdata() / points[:, 0] == factor)
        assert np.all(line['artist'].get_ydata() / points[:, 1] == 1)
        assert 'new' in line['artist'].get_label()

        update = {'new_y_data': points[:, 1] * factor,
                  'model_id': model_id, 'order': order,
                  'new_field': 'new'}
        obj.update_line_fields(1, [update])
        assert np.all(line['artist'].get_xdata() / points[:, 0] == factor)
        assert np.all(line['artist'].get_ydata() / points[:, 1] == factor)

    @pytest.mark.parametrize('new_type,drawstyle,linestyle',
                             [('step', 'steps-mid', '-'),
                              ('line', 'default', '-'),
                              ('scatter', 'default', 'None')])
    def test_update_line_type_line(self, mocker, new_type, drawstyle,
                                   linestyle):
        points = np.array([[1, 2], [3, 4], [5, 6]])
        model_id = 'mid'
        order = 1
        line = {'artist': ml.Line2D(points[:, 0], points[:, 1],
                                    label='orig points'),
                'model_id': model_id, 'order': order,
                'state': 'stale'}
        mocker.patch.object(artists.Artists, 'artists_in_pane',
                            return_value=[line])

        update = {'model_id': model_id, 'order': order,
                  'new_type': new_type,
                  'new_field': 'new'}

        obj = artists.Artists()
        obj.update_line_type(1, [update])

        assert line['artist'].get_drawstyle() == drawstyle
        assert line['artist'].get_linestyle() == linestyle

    def test_update_line_type_scatter(self, mocker, fig):
        points = np.array([[1, 2], [3, 4], [5, 6]])
        model_id = 'mid'
        order = 1
        ax = fig.add_subplot()
        line = {'artist': ax.scatter(points[:, 0], points[:, 1]),
                'model_id': model_id, 'order': order,
                'state': 'stale'}
        mocker.patch.object(artists.Artists, 'artists_in_pane',
                            return_value=[line])
        convert_mock = mocker.patch.object(artists.Artists, 'convert_to_line',
                                           return_value=ma.Artist())
        replace_mock = mocker.patch.object(artists.Artists, '_replace_artist')

        update = {'model_id': model_id, 'order': order,
                  'new_type': 'line', 'new_field': 'new'}
        assert line['artist'].figure is not None
        obj = artists.Artists()
        obj.update_line_type(1, [update])

        assert convert_mock.called_once()
        assert replace_mock.called_with({'kind': 'line'})
        assert line['artist'].figure is None

    @pytest.mark.parametrize('marker,correct', [('x', 'x'), (None, 'o')])
    def test_convert_to_scatter(self, marker, correct, line, fig):
        ax = fig.add_subplot()
        correct_scatter = ax.scatter([0], [1], marker=correct)

        scatter = artists.Artists.convert_to_scatter(line, marker)
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

        art = artists.Artists.convert_to_line(scatter, drawstyle=style,
                                              marker='x')

        assert isinstance(art, ml.Line2D)
        assert art.get_drawstyle() == drawstyle
        npt.assert_array_equal(art.get_color(), colors.to_rgba(face))

    @pytest.mark.parametrize('kinds,counts',
                             [('line', [2, 2, 2, 0]),
                              ('line_alt', [2, 2, 2, 0]),
                              ('error', [0, 0, 1, 1]),
                              ('error_range', [0, 0, 1, 1]),
                              ('border', [0, 0, 1, 0]),
                              ('cursor', [0, 0, 2, 0]),
                              ('cursor_alt', [0, 0, 2, 0]),
                              ('fit', [0, 1, 1, 0]),
                              ('all', [2, 2, 6, 1]),
                              (None, [2, 2, 6, 1])],)
    def test_update_artist_options(self, mocker, kinds, counts):
        marker_mock = mocker.patch.object(artists.Artists,
                                          '_update_marker_style')
        vis_mock = mocker.patch.object(artists.Artists, '_update_visibility')
        color_mock = mocker.patch.object(artists.Artists, '_update_color')
        err_vis_mock = mocker.patch.object(artists.Artists,
                                           '_update_error_visibility')
        model_id = 'border_mid'
        order = 1
        line = {'artist': ml.Line2D(list(), list()),
                'model_id': model_id, 'order': order,
                'state': 'stale'}
        mocker.patch.object(artists.Artists, 'artists_in_pane',
                            return_value=[line])

        update = {'model_id': model_id, 'order': order,
                  'new_type': 'line', 'new_field': 'new'}

        if kinds == 'fit':
            line['data_id'] = 'fit_id'
            update['data_id'] = 'fit'

        obj = artists.Artists()
        obj.update_artist_options(0, kinds, [update])

        assert marker_mock.call_count == counts[0]
        assert vis_mock.call_count == counts[1]
        assert color_mock.call_count == counts[2]
        assert err_vis_mock.call_count == counts[3]

        # test for data_id mismatch: counts should stay the same
        if kinds == 'fit':
            del update['data_id']
        else:
            update['data_id'] = 'fit'
        obj.update_artist_options(0, kinds, [update])
        assert marker_mock.call_count == counts[0]
        assert vis_mock.call_count == counts[1]
        assert color_mock.call_count == counts[2]
        assert err_vis_mock.call_count == counts[3]

    def test_update_fit_artist_options(self, mocker):
        vis_mock = mocker.patch.object(artists.Artists, '_update_visibility')
        color_mock = mocker.patch.object(artists.Artists, '_update_color')

        model_id = 'm_id'
        data_id = 'fit_id'
        order = 1
        line = {'artist': ml.Line2D(list(), list()),
                'model_id': model_id, 'order': order,
                'data_id': data_id, 'state': 'stale'}

        mocker.patch.object(artists.Artists, 'artists_in_pane',
                            return_value=[line])
        obj = artists.Artists()

        # update without data id
        update = {'model_id': model_id, 'order': order,
                  'new_color': 'black', 'new_visibility': False}
        obj._update_fit_artist_options(0, [update])

        # fit line is not updated
        assert vis_mock.call_count == 0
        assert color_mock.call_count == 0

        # update with matching data id
        update = {'model_id': model_id, 'order': order,
                  'new_color': 'black', 'new_visibility': False,
                  'data_id': 'fit'}
        obj._update_fit_artist_options(0, [update])

        # fit line is updated
        assert vis_mock.call_count == 1
        assert color_mock.call_count == 1

    @pytest.mark.parametrize('start,new,result',
                             [('x', 'o', 'o'), ('x', '~', 'x')])
    def test_update_marker_style(self, start, new, result):
        line = {'artist': ml.Line2D(list(), list(), marker=start),
                'marker': start}
        option = {'new_marker': new}

        artists.Artists._update_marker_style(line, option)

        assert line['artist'].get_marker() == result

    def test_update_color(self):
        line = {'artist': ml.Line2D(list(), list(), color='blue')}

        option = {'new_color': 'red'}
        artists.Artists._update_color(line, option)
        assert line['artist'].get_color() == 'red'

        option = {'new_marker': 'green'}
        artists.Artists._update_color(line, option)
        assert line['artist'].get_color() == 'red'

    def test_update_visibility(self):
        line = {'artist': ml.Line2D(list(), list())}
        assert line['artist'].get_visible()

        option = {'new_visibility': False}
        artists.Artists._update_visibility(line, option)
        assert not line['artist'].get_visible()

        option = {'new_marker': 'green'}
        artists.Artists._update_visibility(line, option)
        assert not line['artist'].get_visible()

    def test_update_error_visibility(self):
        line = {'artist': ml.Line2D(list(), list())}
        assert line['artist'].get_visible()

        option = {'new_error_visibility': False}
        artists.Artists._update_error_visibility(line, option)
        assert not line['artist'].get_visible()

        option = {'new_marker': 'green'}
        artists.Artists._update_error_visibility(line, option)
        assert not line['artist'].get_visible()

    def test_replace_artist(self):
        obj = artists.Artists()
        old_artist = ml.Line2D(list(), list(), label='old')
        old = {'artist': old_artist,
               'model_id': 'one', 'order': 1}
        obj.arts = {'line': [old]}

        new_artist = ml.Line2D(list(), list(), label='new')
        obj._replace_artist(kind='line', model='one',
                            order=1, new_artist=new_artist)

        assert obj.arts['line'][0]['artist'] == new_artist

    def test_update_error_ranges(self, mocker, fig):
        line = {'artist': ml.Line2D(list(), list(), label='old'),
                'model_id': 'one', 'order': 1}
        mocker.patch.object(artists.Artists, 'artists_in_pane',
                            return_value=[line])

        updates = {'new_artist': mc.PathCollection(list(),
                                                   list(), label='new'),
                   'model_id': 'one', 'order': 1}

        obj = artists.Artists()
        obj.update_error_ranges(1, [updates])

        assert isinstance(line['artist'], mc.PathCollection)
        assert line['artist'].get_label() == 'old'

    def test_reset_artists_all(self, caplog, mocker):
        caplog.set_level(logging.DEBUG)
        clear_mock = mocker.patch.object(artists.Artists,
                                         '_clear_artists')
        obj = artists.Artists()
        obj.arts = {'line': list(), 'crosshair': list()}
        obj.reset_artists(selection='all')
        assert clear_mock.call_count == 2
        assert 'Resetting all artists' in caplog.text

    def test_reset_artists_alt(self, caplog, mocker):
        caplog.set_level(logging.DEBUG)
        clear_mock = mocker.patch.object(artists.Artists,
                                         '_clear_artists')
        obj = artists.Artists()
        obj.arts = {'line_alt': list(), 'cursor_alt': list(),
                    'line': list()}
        obj.reset_artists(selection='alt')
        assert clear_mock.call_count == 2
        assert 'Resetting alt artists' in caplog.text

    @pytest.mark.parametrize('kind', ['lines', 'cursor', 'collections',
                                      'crosshair', 'patch', 'fit',
                                      'line_alt', 'cursor_alt'])
    def test_reset_artists_selections(self, caplog, mocker, kind):
        caplog.set_level(logging.DEBUG)
        clear_mock = mocker.patch.object(artists.Artists,
                                         '_clear_artists')
        obj = artists.Artists()
        obj.arts = {'line': list(), 'crosshair': list()}
        obj.reset_artists(selection=kind)

        assert f'Resetting {kind}' in caplog.text
        assert clear_mock.called_with({'selection': kind})

    def test_reset_artists_guide(self, caplog, mocker):
        caplog.set_level(logging.DEBUG)
        clear_mock = mocker.patch.object(artists.Artists,
                                         '_clear_guides')
        obj = artists.Artists()
        obj.reset_artists(selection='v_guide')

        assert 'Resetting v_guide' in caplog.text
        assert clear_mock.called_with({'selection': 'v'})

    def test_reset_artists_fail(self, caplog):
        caplog.set_level(logging.DEBUG)
        selection = 'fail'

        obj = artists.Artists()
        obj.reset_artists(selection=selection)

        assert f'Resetting {selection} artists' in caplog.text
        assert 'Invalid' in caplog.text

    @pytest.mark.parametrize('flag,panes,n_remain,count',
                             [('v', None, 1, 1),
                              ('h', None, 1, 1),
                              ('h', 1, 1, 1),
                              ('a', [1, 2], 0, 4),
                              ('b', None, 2, 0)])
    def test_clear_guides(self, flag, panes, n_remain, count, mocker, guide):
        arts = [{'artist': copy.copy(guide),
                 'model_id': 'v_guide'},
                {'artist': copy.copy(guide),
                 'model_id': 'h_guide'}]
        mocker.patch.object(artists.Artists, 'artists_in_pane',
                            return_value=arts)
        rev_mock = mocker.patch.object(ma.Artist, 'remove')

        obj = artists.Artists()
        obj.arts = {'guide': arts}
        obj._clear_guides(flag, panes)

        assert len(obj.arts['guide']) == n_remain
        assert rev_mock.call_count == count

    @pytest.mark.parametrize('panes,clear,n_remain,count',
                             [(None, True, 0, 2),
                              (None, False, 2, 0),
                              ([1, 2], True, 0, 2)])
    def test_clear_artists(self, mocker, line, panes, clear, n_remain, count):
        arts = [{'artist': copy.copy(line),
                 'model_id': 'one'},
                {'artist': copy.copy(line),
                 'model_id': 'two'}]
        if clear:
            return_value = arts
        else:
            return_value = list()
        mocker.patch.object(artists.Artists, 'artists_in_pane',
                            return_value=return_value)

        rev_mock = mocker.patch.object(ma.Artist, 'remove')
        kind = 'line'

        obj = artists.Artists()
        obj.arts = {kind: arts}
        obj._clear_artists(kind, panes)

        assert len(obj.arts[kind]) == n_remain
        assert rev_mock.call_count == count

        # mock an error in the remove call: should have same effect
        rev_mock = mocker.patch.object(ma.Artist, 'remove',
                                       side_effect=ValueError)
        obj = artists.Artists()
        obj.arts = {kind: arts}
        obj._clear_artists(kind, panes)

        assert len(obj.arts[kind]) == n_remain
        assert rev_mock.call_count == count

    def test_add_artists(self, mocker, line):
        patch = mocker.patch.object(artists.Artists, 'add_artist',
                                    return_value=True)
        arts = {'model_1': {1: {'line': copy.copy(line)}},
                'model_2': {1: {'line': copy.copy(line)}}}

        obj = artists.Artists()
        count = obj.add_artists(arts)

        assert count == 2
        assert patch.called_with({'name': 'model_1'})
        assert patch.called_with({'name': 'model_2'})

    def test_add_artist(self, line, scatter, guide, fit):
        name = 'test'
        arts = {1: {'line': {'artist': copy.copy(line),
                             'data_id': 1},
                    'errors': {'artist': copy.copy(scatter)}},
                2: {'line': {'artist': copy.copy(line)},
                    'guide': {'artist': copy.copy(guide)},
                    'fit': {'artist': copy.copy(fit)}}}

        obj = artists.Artists()
        obj.arts = {'line': list(),
                    'errors': list(),
                    'guide': list(),
                    'crosshairs': list(),
                    'fit': list()}
        result = obj.add_artist(artist=arts, name=name)

        assert result
        correct = {'line': 2, 'errors': 1, 'guide': 1,
                   'fit': 1, 'crosshairs': 0}
        for field, value in correct.items():
            assert len(obj.arts[field]) == value

    @pytest.mark.parametrize('kind,count',
                             [('line', 2), ('cursor', 1),
                              ('crosshair', 0), ('fit', 1),
                              ('line_alt', 1), ('bad', 6)])
    def test_artists_in_pane(self, kind, count,
                             line, line_alt, scatter, guide, fit):

        arts = {'line': [{'artist': line,
                          'model_id': 'model_1'},
                         {'artist': line,
                          'model_id': 'model_2'}],
                'line_alt': [{'artist': line_alt,
                              'model_id': 'model_3'}],
                'cursor': [{'artist': scatter,
                            'model_id': 'model_1'}],
                'error_range': list(),
                'crosshair': list(),
                'guide': [{'artist': guide,
                           'model_id': 'model_1'}],
                'patch': list(),
                'fit': [{'artist': fit,
                         'model_id': 'model_1'}]
                }

        obj = artists.Artists()
        obj.arts = arts
        pane_ = pane.OneDimPane(arts['line'][0]['artist'].axes)
        pane_.show_overplot = True
        pane_.ax_alt = line_alt.axes

        result = obj.artists_in_pane(pane_, kind)
        assert len(result) == count

        result = obj.artists_in_pane(None, kind)
        assert len(result) == count

    def test_gather_artists(self, line, scatter, guide, fit):
        arts = {'line': [{'artist': line,
                          'model_id': 'model_1'},
                         {'artist': line,
                          'model_id': 'model_2'}],
                'cursor': [{'artist': scatter,
                            'model_id': 'model_1'}],
                'error_range': list(),
                'crosshair': list(),
                'guide': [{'artist': guide,
                           'model_id': 'model_1'}],
                'patch': list(),
                'fit': [{'artist': fit,
                         'model_id': 'model_1'}]
                }

        obj = artists.Artists()
        obj.arts = arts

        gathered = obj.gather_artists('line', preserve=True)
        assert all([isinstance(g['artist'], ml.Line2D) for g in gathered])
        assert all([g['state'] == 'new' for g in gathered])

        gathered = obj.gather_artists('line', preserve=False)
        assert all([g['state'] == 'fresh' for g in gathered])

    def test_print_artists(self, line, capsys, scatter):
        arts = {'line': [{'artist': line,
                          'model_id': 'model_1'},
                         {'artist': line,
                          'model_id': 'model_2'}],
                'cursor': [{'artist': scatter,
                            'model_id': 'model_1'}]}
        obj = artists.Artists()
        obj.arts = arts

        obj.print_artists()
        captured = capsys.readouterr()

        assert captured.out.count('Line2D') == len(arts['line'])
        assert captured.out.count('PathCollection') == len(arts['cursor'])

    @pytest.mark.parametrize('state,mode,correct',
                             [('new', 'new', True),
                              ('any', 'all', True),
                              ('any', 'new', False),
                              ('fresh', 'new', False),
                              ('fresh', 'viable', True),
                              ('new', 'viable', True),
                              ('stale', 'new', False),
                              ('stale', 'viable', False),
                              ('stale', 'all', True)])
    def test_artist_fits_mode(self, state, mode, correct):

        art = {'state': state}
        result = artists.Artists._artist_fits_mode(art, mode)
        assert result is correct

    def test_flatten(self):
        mixed = [(1, 2), (3, 4, 5), 6, (7, 8)]
        result = artists.Artists._flatten(mixed)
        assert result == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_age_artists(self):
        arts = {'line': [{'artist': {'state': 'new'}},
                         {'artist': {'state': 'stale'}},
                         {'artist': {'state': 'fresh'}}]}
        obj = artists.Artists()
        obj.arts = arts

        obj.age_artists()

        assert all([a['artist']['state'] == 'stale'
                    for a in obj.arts['line']])

    def test_update_marker(self, scatter):
        orig = copy.copy(scatter)
        model_id = 'one'
        order = 1
        arts = {'cursor': [{'artist': scatter, 'state': 'stale',
                            'model_id': model_id, 'order': order,
                            'fields': ['wavepos', 'flux']}],
                'cursor_alt': list()}

        x, y = 2, 4
        data = {model_id: [{'order': order, 'visible': False,
                            'bin_x': x, 'bin_y': y,
                            'x_field': 'wavepos', 'y_field': 'flux'}],
                'other_mode': None}

        obj = artists.Artists()
        obj.arts = arts
        obj.update_marker(data)
        assert obj.arts['cursor'][0]['state'] == 'new'
        assert not obj.arts['cursor'][0]['artist'].get_visible()
        npt.assert_array_equal(orig.get_offsets(),
                               obj.arts['cursor'][0]['artist'].get_offsets())

        data = {model_id: [{'order': order, 'visible': True,
                            'bin_x': x, 'bin_y': y,
                            'x_field': 'wavepos', 'y_field': 'flux'}]}
        obj.update_marker(data)
        assert obj.arts['cursor'][0]['artist'].get_visible()
        npt.assert_array_equal([[x, y]],
                               obj.arts['cursor'][0]['artist'].get_offsets())

    def test_hide_cursor_markers(self, scatter):
        cursor_1 = copy.copy(scatter)
        cursor_2 = copy.copy(scatter)
        cursor_3 = copy.copy(scatter)

        arts = {'cursor': [{'artist': cursor_1},
                           {'artist': cursor_2}],
                'cursor_alt': [{'artist': cursor_3}]}

        obj = artists.Artists()
        obj.arts = arts

        assert all([c['artist'].get_visible()
                    for c in obj.arts['cursor']])
        assert all([c['artist'].get_visible()
                    for c in obj.arts['cursor_alt']])

        obj.hide_cursor_markers()

        assert all([not c['artist'].get_visible()
                    for c in obj.arts['cursor']])
        assert all([not c['artist'].get_visible()
                    for c in obj.arts['cursor_alt']])
