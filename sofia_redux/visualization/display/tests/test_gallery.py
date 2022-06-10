#  Licensed under a 3-clause BSD style license - see LICENSE.rst

import copy
import logging
import pytest
import numpy as np
import numpy.testing as npt

from matplotlib import artist as ma
from matplotlib import lines as ml
from matplotlib import collections as mc

from sofia_redux.visualization.display import gallery, pane, drawing
from sofia_redux.visualization import signals

PyQt5 = pytest.importorskip('PyQt5')


class TestGallery(object):

    def test_init(self):
        keys = ['line', 'cursor', 'error_range', 'crosshair',
                'guide', 'patch', 'fit', 'line_alt', 'cursor_alt']
        obj = gallery.Gallery()

        assert all([k in obj.arts.keys() for k in keys])
        assert all([isinstance(v, list) for v in obj.arts.values()])

    def test_str(self):
        obj = gallery.Gallery()
        obj_str = str(obj)
        assert obj_str.startswith('Gallery at')
        lines = obj_str.split('\n')
        assert len(lines) == len(obj.arts) + 2

    def test_add_patches(self):
        blank = ma.Artist()
        assert blank.get_visible()
        patches = {'one': {'kind': 'test',
                           'artist': blank,
                           'visible': False}}

        obj = gallery.Gallery()
        obj.add_patches(patches)

        assert isinstance(obj.arts['patch'][0], drawing.Drawing)
        assert obj.arts['patch'][0].match_high_model('test_one')
        assert not obj.arts['patch'][0].get_artist().get_visible()

    def test_set_pane_highlight_flag(self):
        blank_1 = ma.Artist()
        blank_2 = ma.Artist()
        blank_3 = ma.Artist()
        obj = gallery.Gallery()

        obj.arts['patch'] = [drawing.Drawing(high_model='border_pane_1',
                                             artist=blank_1),
                             drawing.Drawing(high_model='border_pane_2',
                                             artist=blank_2),
                             drawing.Drawing(high_model='test',
                                             artist=blank_3)]

        obj.set_pane_highlight_flag(pane_number=1, state=False)
        assert not obj.arts['patch'][0].get_artist().get_visible()
        assert not obj.arts['patch'][1].get_artist().get_visible()
        assert obj.arts['patch'][2].get_artist().get_visible()

    def test_add_crosshairs(self, caplog):
        blank = ma.Artist()
        assert blank.get_visible()
        options = {'high_model': 'crosshair_pane_1', 'mid_model': 'vertical',
                   'kind': 'crosshair', 'artist': blank, 'visible': False}
        crosshairs = drawing.Drawing(**options)

        obj = gallery.Gallery()
        obj.add_crosshairs([crosshairs])

        assert crosshairs in obj.arts['crosshair']
        assert not obj.arts['crosshair'][0].get_artist().get_visible()

        # adding again has no effect
        caplog.set_level(logging.DEBUG)
        obj.add_crosshairs([crosshairs])
        assert 'Crosshair already present' in caplog.text
        assert len(obj.arts['crosshair']) == 1

    @pytest.mark.parametrize('direction,change_x,change_y',
                             [('v', True, False), ('h', False, True),
                              ('vh', True, True), ('', True, True)])
    def test_update_crosshair(self, direction, change_x, change_y):
        old_point = {'x': [6], 'y': [3]}
        new_point = {'x': [5], 'y': [2]}
        common_options = {'kind': 'crosshair', 'visible': False}
        panes = [1, 1, 2]
        directions = ['vertical', 'horizontal', 'horizontal']
        crosshairs = list()
        for p, d in zip(panes, directions):
            cross = drawing.Drawing(high_model=f'crosshair_pane_{p}',
                                    mid_model=d,
                                    artist=ml.Line2D(old_point['x'],
                                                     old_point['y']),
                                    **common_options)
            crosshairs.append(cross)

        obj = gallery.Gallery()
        obj.add_crosshairs(crosshairs)
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
            xdata = obj.arts['crosshair'][0].get_artist().get_xdata()
            assert xdata == target_x
        if 'h' in direction:
            ydata = obj.arts['crosshair'][1].get_artist().get_ydata()
            assert ydata == target_y

    def test_hide_crosshair(self):
        old_point = {'x': [6], 'y': [3]}
        common_options = {'kind': 'crosshair', 'visible': True,
                          'high_model': 'crosshair_pane_1'}
        directions = ['vertical', 'horizontal']
        crosshairs = list()
        for d in directions:
            cross = drawing.Drawing(artist=ml.Line2D(old_point['x'],
                                                     old_point['y']),
                                    mid_model=d, **common_options)
            crosshairs.append(cross)

        obj = gallery.Gallery()
        obj.add_crosshairs(crosshairs)

        for crosshair in obj.arts['crosshair']:
            assert crosshair.get_visible()

        obj.hide_crosshair()

        for crosshair in obj.arts['crosshair']:
            assert not crosshair.get_visible()

    def test_update_line_data(self, mocker, capsys):
        line = ml.Line2D([2], [2])
        scatter = mc.PathCollection([2, 34])
        patch = mc.PatchCollection(list())
        common_options = {'mid_model': 1, 'fields': ['wavelength', 'flux']}
        highs = ['line', 'scatter', 'scatter']
        kinds = ['line', 'scatter', 'scatter']
        arts = [line, scatter, patch]
        lines = list()
        for high, kind, artist in zip(highs, kinds, arts):
            draw = drawing.Drawing(high_model=high, kind=kind, artist=artist,
                                   **common_options)
            lines.append(draw)
        updates = list()
        highs = ['line', 'scatter', 'patch']
        arts = [ma.Artist(), ma.Artist(), list()]
        for high, kind, artist in zip(highs, kinds, arts):
            draw = drawing.Drawing(high_model=high, kind=kind,
                                   updates={'artist': artist},
                                   **common_options)
            updates.append(draw)

        # returns same result for line and line_alt
        mocker.patch.object(gallery.Gallery, 'artists_in_pane',
                            return_value=lines)
        line_mock = mocker.patch.object(drawing.Drawing, '_set_line_data')
        path_mock = mocker.patch.object(drawing.Drawing, '_set_scatter_data')

        obj = gallery.Gallery()
        obj.update_line_data(pane_=1, updates=updates, axes='both')
        captured = capsys.readouterr()

        assert line_mock.call_count == 2
        assert path_mock.call_count == 2
        assert 'Unable to process' in captured.out

    def test_line_fields(self, mocker):
        points = np.array([[1, 2], [3, 4], [5, 6]])
        factor = 2
        model_id = 'mid'
        order = 1
        art = ml.Line2D(points[:, 0], points[:, 1], label='orig points')
        line_options = {'high_model': model_id, 'mid_model': order,
                        'state': 'stale'}
        line = drawing.Drawing(artist=art, **line_options)
        mocker.patch.object(gallery.Gallery, 'artists_in_pane',
                            return_value=[line])

        update = {'new_x_data': points[:, 0] * factor, 'new_field': 'new'}
        update_line = drawing.Drawing(updates=update, **line_options)
        obj = gallery.Gallery()
        obj.update_line_fields(1, [update_line])

        assert line.get_state() == 'new'
        assert np.all(line.get_artist().get_xdata() / points[:, 0] == factor)
        assert np.all(line.get_artist().get_ydata() / points[:, 1] == 1)
        assert 'new' in line.get_artist().get_label()

        update = {'new_y_data': points[:, 1] * factor, 'new_field': 'new'}
        update_line = drawing.Drawing(updates=update, **line_options)
        obj.update_line_fields(1, [update_line])
        assert np.all(line.get_artist().get_xdata() / points[:, 0] == factor)
        assert np.all(line.get_artist().get_ydata() / points[:, 1] == factor)

    @pytest.mark.parametrize('new_type,drawstyle,linestyle',
                             [('step', 'steps-mid', '-'),
                              ('line', 'default', '-'),
                              ('scatter', 'default', 'None')])
    def test_update_line_type_line(self, mocker, new_type, drawstyle,
                                   linestyle):
        points = np.array([[1, 2], [3, 4], [5, 6]])
        model_id = 'mid'
        order = 1
        line_options = {'artist': ml.Line2D(points[:, 0], points[:, 1],
                                            label='orig points'),
                        'high_model': model_id, 'mid_model': order,
                        'state': 'stale'}
        line = drawing.Drawing(**line_options)
        mocker.patch.object(gallery.Gallery, 'artists_in_pane',
                            return_value=[line])

        update_options = {'type': new_type, 'field': 'new'}
        update = drawing.Drawing(high_model=model_id, mid_model=order,
                                 updates=update_options)

        obj = gallery.Gallery()
        obj.update_line_type(1, [update])

        assert line.get_artist().get_drawstyle() == drawstyle
        assert line.get_artist().get_linestyle() == linestyle

    def test_update_line_type_scatter(self, mocker, fig):
        points = np.array([[1, 2], [3, 4], [5, 6]])
        model_id = 'mid'
        order = 1
        ax = fig.add_subplot()

        line_options = {'artist': ax.scatter(points[:, 0], points[:, 1]),
                        'high_model': model_id, 'mid_model': order,
                        'kind': 'line', 'state': 'stale'}
        line = drawing.Drawing(**line_options)
        mocker.patch.object(gallery.Gallery, 'artists_in_pane',
                            return_value=[line])
        convert_mock = mocker.patch.object(drawing.Drawing, 'convert_to_line')

        update_options = {'model_id': model_id, 'order': order, 'kind': 'line',
                          'updates': {'type': 'line', 'field': 'new'}}
        update = drawing.Drawing(**update_options)

        assert line.get_artist().figure is not None
        obj = gallery.Gallery()
        obj.update_line_type(1, [update])

        assert convert_mock.called_once()
        assert convert_mock.called_with({'marker': None})
        assert line.get_artist().figure is not None

    @pytest.mark.parametrize('kinds,call_counts',
                             [('line', [1, 0, 0, 0, 0, 0]),
                              ('line_alt', [1, 0, 0, 0, 0, 0]),
                              ('error', [0, 1, 0, 0, 0, 0]),
                              ('error_range', [0, 1, 0, 0, 0, 0]),
                              ('border', [0, 0, 1, 0, 0, 0]),
                              ('cursor', [0, 0, 0, 1, 0, 0]),
                              ('cursor_alt', [0, 0, 0, 1, 0, 0]),
                              ('fit', [0, 0, 0, 0, 1, 0]),
                              ('ref_line', [0, 0, 0, 0, 0, 1]),
                              ('ref_label', [0, 0, 0, 0, 0, 1]),
                              ('all', [1, 1, 1, 1, 1, 1]),
                              (None, [1, 1, 1, 1, 1, 1])])
    def test_update_artist_options(self, mocker, kinds, call_counts):
        names = ['line', 'error', 'border', 'cursor', 'fit', 'reference']
        mocks = list()
        for name in names:
            mocks.append(mocker.patch.object(gallery.Gallery,
                                             f'_update_{name}_artist_options'))
        model_id = 'border_mid'
        order = 1
        line = drawing.Drawing(artist=ml.Line2D(list(), list()),
                               high_model=model_id, mid_model=order,
                               state='stale')
        mocker.patch.object(gallery.Gallery, 'artists_in_pane',
                            return_value=[line])

        update_options = {'high_model': model_id, 'mid_model': order,
                          'updates': {'type': 'line', 'field': 'new'}}
        update = drawing.Drawing(**update_options)

        if kinds == 'fit':
            line.set_data_id('fit_id')
            update.set_data_id('fit')

        obj = gallery.Gallery()
        obj.update_artist_options(0, kinds, [update])

        assert [m.call_count for m in mocks] == call_counts
        for m in mocks:
            m.reset_mock()

        # test for data_id mismatch: counts should stay the same
        if kinds == 'fit':
            update.set_data_id('')
        else:
            update.set_data_id('fit')
        obj.update_artist_options(0, kinds, update)
        assert [m.call_count for m in mocks] == call_counts
        for m in mocks:
            m.reset_mock()

        # test for None options: calls should be same
        obj.update_artist_options(0, kinds, None)
        assert [m.call_count for m in mocks] == call_counts

    def test_update_fit_artist_options(self, mocker):
        mock = mocker.patch.object(drawing.Drawing, 'update_options',
                                   return_value=True)

        model_id = 'm_id'
        data_id = 'fit_id'
        order = 1
        line = drawing.Drawing(artist=ml.Line2D(list(), list()),
                               high_model=model_id, mid_model=order,
                               state='stale', data_id=data_id)

        mocker.patch.object(gallery.Gallery, 'artists_in_pane',
                            return_value=[line])
        obj = gallery.Gallery()

        # update without data id
        new_color = 'red'
        update_options = {'high_model': model_id, 'mid_model': order,
                          'updates': {'color': new_color,
                                      'visibility': False}}
        update = drawing.Drawing(**update_options)

        obj._update_fit_artist_options(0, [update])

        # fit line is not updated
        assert mock.call_count == 0

        # update with matching data id
        update = drawing.Drawing(data_id=data_id, **update_options)
        obj._update_fit_artist_options(0, [update])

        # fit line is updated
        assert mock.call_count == 1

    def test_update_reference_artist_options(self, mocker):
        mock = mocker.patch.object(drawing.Drawing, 'update_options',
                                   return_value=True)

        model_id = 'm_id'
        data_id = 'ref_id'
        order = 1
        line = drawing.Drawing(artist=ml.Line2D(list(), list()),
                               high_model=model_id, mid_model=order,
                               state='stale')

        mocker.patch.object(gallery.Gallery, 'artists_in_pane',
                            return_value=[line])
        obj = gallery.Gallery()

        # update without data id
        update_options = {'high_model': model_id, 'mid_model': order,
                          'updates': {'visibility': False}}
        update = drawing.Drawing(**update_options)

        obj._update_reference_artist_options(0, [update])

        # line is updated
        assert mock.call_count == 2
        mock.reset_mock()

        # update with data id
        update = drawing.Drawing(data_id=data_id, **update_options)
        obj._update_reference_artist_options(0, [update])

        # line is not updated
        assert mock.call_count == 0

    def test_replace_artist(self):
        obj = gallery.Gallery()
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
                'high_model': 'one', 'mid_model': 1}
        line = drawing.Drawing(**line)
        mocker.patch.object(gallery.Gallery, 'artists_in_pane',
                            return_value=[line])

        updates = {'updates': {'artist': mc.PathCollection(list(), list(),
                                                           label='new')},
                   'high_model': 'one', 'mid_model': 1}
        updates = drawing.Drawing(**updates)

        obj = gallery.Gallery()
        obj.update_error_ranges(1, [updates])

        assert isinstance(line.get_artist(), mc.PathCollection)
        assert line.get_artist().get_label() == 'old'

    def test_reset_artists_all(self, caplog, mocker):
        caplog.set_level(logging.DEBUG)
        clear_mock = mocker.patch.object(gallery.Gallery,
                                         '_clear_artists')
        obj = gallery.Gallery()
        obj.arts = {'line': list(), 'crosshair': list()}
        obj.reset_artists(selection='all')
        assert clear_mock.call_count == 2
        assert 'Resetting all gallery' in caplog.text

    def test_reset_artists_alt(self, caplog, mocker):
        caplog.set_level(logging.DEBUG)
        clear_mock = mocker.patch.object(gallery.Gallery,
                                         '_clear_artists')
        obj = gallery.Gallery()
        obj.arts = {'line_alt': list(), 'cursor_alt': list(),
                    'line': list()}
        obj.reset_artists(selection='alt')
        assert clear_mock.call_count == 2
        assert 'Resetting alt gallery' in caplog.text

    @pytest.mark.parametrize('kind', ['lines', 'cursor', 'collections',
                                      'crosshair', 'patch', 'fit',
                                      'line_alt', 'cursor_alt', 'ref_line',
                                      'ref_label'])
    def test_reset_artists_selections(self, caplog, mocker, kind):
        caplog.set_level(logging.DEBUG)
        clear_mock = mocker.patch.object(gallery.Gallery,
                                         '_clear_artists')
        obj = gallery.Gallery()
        obj.arts = {'line': list(), 'crosshair': list()}
        obj.reset_artists(selection=kind)

        assert f'Resetting {kind}' in caplog.text
        assert clear_mock.called_with({'selection': kind})

    def test_reset_artists_guide(self, caplog, mocker):
        caplog.set_level(logging.DEBUG)
        clear_mock = mocker.patch.object(gallery.Gallery,
                                         '_clear_guides')
        obj = gallery.Gallery()
        obj.reset_artists(selection='v_guide')

        assert 'Resetting v_guide' in caplog.text
        assert clear_mock.called_with({'selection': 'v'})

    def test_reset_artists_fail(self, caplog):
        caplog.set_level(logging.DEBUG)
        selection = 'fail'

        obj = gallery.Gallery()
        obj.reset_artists(selection=selection)

        assert f'Resetting {selection} gallery' in caplog.text
        assert 'Invalid' in caplog.text

    @pytest.mark.parametrize('flag,panes,n_remain,count',
                             [('v', None, 1, 1),
                              ('h', None, 1, 1),
                              ('h', 1, 1, 1),
                              ('a', [1, 2], 0, 4),
                              ('b', None, 2, 0)])
    def test_clear_guides(self, flag, panes, n_remain, count, mocker, guide):
        draws = [drawing.Drawing(high_model='guide', mid_model='vertical',
                                 kind='guide', artist=copy.copy(guide)),
                 drawing.Drawing(high_model='guide', mid_model='horizontal',
                                 kind='guide', artist=copy.copy(guide))]
        mocker.patch.object(gallery.Gallery, 'artists_in_pane',
                            return_value=draws)
        rev_mock = mocker.patch.object(ma.Artist, 'remove')

        obj = gallery.Gallery()
        obj.arts = {'guide': draws}
        obj._clear_guides(flag, panes)

        assert len(obj.arts['guide']) == n_remain
        assert rev_mock.call_count == count

    @pytest.mark.parametrize('panes,clear,n_remain,count',
                             [(None, True, 0, 2),
                              (None, False, 2, 0),
                              ([1, 2], True, 0, 2)])
    def test_clear_artists(self, mocker, line, panes, clear, n_remain, count):
        arts = [drawing.Drawing(artist=copy.copy(line), high_model='one'),
                drawing.Drawing(artist=copy.copy(line), high_model='two')]
        if clear:
            return_value = arts
        else:
            return_value = list()
        mocker.patch.object(gallery.Gallery, 'artists_in_pane',
                            return_value=return_value)

        rev_mock = mocker.patch.object(ma.Artist, 'remove')
        kind = 'line'

        obj = gallery.Gallery()
        obj.arts = {kind: arts}
        obj._clear_artists(kind, panes)

        assert len(obj.arts[kind]) == n_remain
        assert rev_mock.call_count == count

        # mock an error in the remove call: should have same effect
        rev_mock = mocker.patch.object(ma.Artist, 'remove',
                                       side_effect=ValueError)
        obj = gallery.Gallery()
        obj.arts = {kind: arts}
        obj._clear_artists(kind, panes)

        assert len(obj.arts[kind]) == n_remain
        assert rev_mock.call_count == count

    def test_add_drawings(self, mocker):
        patch = mocker.patch.object(gallery.Gallery, 'add_drawing',
                                    return_value=True)
        draw_1 = drawing.Drawing()
        draw_2 = drawing.Drawing()
        draws = [draw_1, draw_2]

        obj = gallery.Gallery()
        count = obj.add_drawings(draws)

        assert count == 2
        assert patch.called_with(draw_1)
        assert patch.called_with(draw_2)

    def test_add_drawing_empty(self):
        draw = drawing.Drawing(kind='line')

        obj = gallery.Gallery()
        assert len(obj.arts['line']) == 0

        result = obj.add_drawing(draw)

        assert result is False
        assert len(obj.arts['line']) == 0

    def test_add_drawing(self):
        art = ml.Artist()
        draw = drawing.Drawing(kind='line', artist=art)

        obj = gallery.Gallery()
        assert len(obj.arts['line']) == 0

        result = obj.add_drawing(draw)

        assert result is True
        assert len(obj.arts['line']) == 1
        assert draw in obj.arts['line']

    def test_add_drawing_new_type(self):
        art = ml.Artist()
        kind = 'unknown'
        draw = drawing.Drawing(kind=kind, artist=art)

        obj = gallery.Gallery()
        assert kind not in obj.arts.keys()

        result = obj.add_drawing(draw)

        assert result is True
        assert len(obj.arts[kind]) == 1
        assert draw in obj.arts[kind]

    def test_add_drawing_bad(self):
        obj = gallery.Gallery()
        with pytest.raises(TypeError) as err:
            obj.add_drawing('test')
        assert 'not of a valid type' in str(err)

    @pytest.mark.parametrize('kind,count',
                             [('line', 2), ('cursor', 1),
                              ('crosshair', 0), ('fit', 1),
                              ('line_alt', 1), ('bad', 6)])
    def test_artists_in_pane(self, kind, count,
                             line, line_alt, scatter, guide, fit):
        arts = {'line': [drawing.Drawing(artist=line, kind='line',
                                         high_model='model_1'),
                         drawing.Drawing(artist=line, kind='line',
                                         kindhigh_model='model_2')],
                'line_alt': [drawing.Drawing(artist=line_alt, kind='line',
                                             high_model='model_3')],
                'cursor': [drawing.Drawing(artist=scatter, kind='cursor',
                                           high_model='model_1')],
                'error_range': list(),
                'crosshair': list(),
                'guide': [drawing.Drawing(artist=guide, kind='guide',
                                          high_model='model_1')],
                'patch': list(),
                'fit': [drawing.Drawing(artist=fit, kind='fit',
                                        high_model='model_1')]
                }

        obj = gallery.Gallery()
        obj.arts = arts
        sigs = signals.Signals()
        pane_ = pane.OneDimPane(sigs, arts['line'][0].get_artist().axes)
        pane_.show_overplot = True
        pane_.ax_alt = line_alt.axes

        result = obj.artists_in_pane(pane_, kind)
        assert len(result) == count

        result = obj.artists_in_pane(None, kind)
        assert len(result) == count

    def test_gather_artists(self, line, scatter, guide, fit):

        arts = {'line': [drawing.Drawing(artist=line, kind='line',
                                         high_model='model_1'),
                         drawing.Drawing(artist=line, kind='line',
                                         kindhigh_model='model_2')],
                'line_alt': list(),
                'cursor': [drawing.Drawing(artist=scatter, kind='cursor',
                                           high_model='model_1')],
                'error_range': list(),
                'crosshair': list(),
                'guide': [drawing.Drawing(artist=guide, kind='guide',
                                          high_model='model_1')],
                'patch': list(),
                'fit': [drawing.Drawing(artist=fit, kind='fit',
                                        high_model='model_1')]
                }

        obj = gallery.Gallery()
        obj.arts = arts

        gathered = obj.gather_artists('line', preserve=True)
        assert all([isinstance(g.get_artist(), ml.Line2D) for g in gathered])
        assert all([g.get_state() == 'new' for g in gathered])

        gathered = obj.gather_artists('line', preserve=False)
        assert all([g.get_state() == 'fresh' for g in gathered])

    def test_print_artists(self, line, capsys, scatter):
        arts = {'line': [drawing.Drawing(artist=line, high_model='model_1',
                                         kind='line'),
                         drawing.Drawing(artist=line, high_model='model_2',
                                         kind='line')],
                'cursor': [drawing.Drawing(artist=scatter, kind='cursor',
                                           high_model='model_1')]}
        obj = gallery.Gallery()
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
        art = drawing.Drawing()
        art.set_state(state)
        result = gallery.Gallery._drawing_fits_mode(art, mode)
        assert result is correct

    def test_flatten(self):
        mixed = [(1, 2), (3, 4, 5), 6, (7, 8)]
        result = gallery.Gallery._flatten(mixed)
        assert result == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_age_artists(self):
        lines = [drawing.Drawing(kind='line'), drawing.Drawing(kind='line'),
                 drawing.Drawing(kind='line')]
        states = ['new', 'stale', 'fresh']
        for line, state in zip(lines, states):
            line.set_state(state)
        arts = {'line': lines}
        obj = gallery.Gallery()
        obj.arts = arts

        obj.age_artists()

        assert all([draw.get_state() == 'stale' for draw in obj.arts['line']])

    @pytest.mark.parametrize('alt', [True, False])
    def test_update_marker(self, scatter, alt):
        orig = copy.copy(scatter)
        model_id = 'one'
        order = 1
        if alt:
            draw = {'artist': scatter, 'state': 'stale',
                    'high_model': model_id,
                    'mid_model': order, 'axes': 'alt',
                    'fields': {'x': 'wavepos', 'y_alt': 'flux'}}
            arts = {'cursor': list(),
                    'cursor_alt': [drawing.Drawing(**draw)]}
            field = 'cursor_alt'
        else:
            draw = {'artist': scatter, 'state': 'stale',
                    'high_model': model_id,
                    'mid_model': order, 'axes': 'primary',
                    'fields': {'x': 'wavepos', 'y': 'flux'}}
            arts = {'cursor': [drawing.Drawing(**draw)],
                    'cursor_alt': list()}
            field = 'cursor'

        x, y = 2, 4
        data = {model_id: [{'order': order, 'visible': False,
                            'bin_x': x, 'bin_y': y,
                            'x_field': 'wavepos', 'y_field': 'flux',
                            'alt': alt}],
                'other_mode': None}

        obj = gallery.Gallery()
        obj.arts = arts
        obj.update_marker(data)
        assert obj.arts[field][0].state == 'new'
        assert not obj.arts[field][0].visible
        npt.assert_array_equal(
            orig.get_offsets(),
            obj.arts[field][0].get_artist().get_offsets())

        data = {model_id: [{'order': order, 'visible': True,
                            'bin_x': x, 'bin_y': y,
                            'x_field': 'wavepos', 'y_field': 'flux',
                            'alt': alt}]}
        obj.update_marker(data)
        assert obj.arts[field][0].visible
        npt.assert_array_equal(
            [[x, y]], obj.arts[field][0].get_artist().get_offsets())

    def test_hide_cursor_markers(self, scatter):
        cursor_1 = copy.copy(scatter)
        cursor_2 = copy.copy(scatter)
        cursor_3 = copy.copy(scatter)

        arts = {'cursor': [drawing.Drawing(artist=cursor_1),
                           drawing.Drawing(artist=cursor_2)],
                'cursor_alt': [drawing.Drawing(artist=cursor_3)]}

        obj = gallery.Gallery()
        obj.arts = arts

        assert all([c.visible for c in obj.arts['cursor']])
        assert all([c.visible for c in obj.arts['cursor_alt']])

        obj.hide_cursor_markers()

        assert all([not c.visible for c in obj.arts['cursor']])
        assert all([not c.visible for c in obj.arts['cursor_alt']])

    def test_update_reference_data(self, mocker):
        obj = gallery.Gallery()

        # check that clear then add are called
        m1 = mocker.patch.object(obj, '_clear_artists')
        m2 = mocker.patch.object(obj, 'add_drawings')

        obj.update_reference_data('pane', 'updates')
        assert m1.call_count == 1
        assert m1.called_with(kind=['ref_line', 'ref_label'], panes=['pane'])
        assert m2.call_count == 1
        assert m2.called_with('updates')

    @pytest.mark.parametrize('visible,pane_id,overlaps,'
                             'update_count,new_count,new_text',
                             [(True, False, True, 5, 1,
                               'test 0,test 1,test 2,test 3,test 4,...'),
                              (False, False, True, 0, 6, 'test 0'),
                              (True, True, True, 0, 6, 'test 0'),
                              (True, False, False, 0, 6, 'test 0'),
                              ])
    def test_catch_label_overlaps(self, blank_blitter, one_dim_pane,
                                  visible, pane_id, overlaps, update_count,
                                  new_count, new_text, mocker):
        renderer = blank_blitter._canvas.get_renderer()
        obj = gallery.Gallery()

        # make a bunch of labels on top of each other
        model_id = 'm_id'
        order = 1

        for i in range(6):
            if overlaps:
                art = one_dim_pane.ax.text(1, 1, f'test {i}', visible=visible)
            else:
                art = one_dim_pane.ax.text(10 * i + 1, 10 * i + 1,
                                           f'test {i}', visible=visible)
            if pane_id:
                pane_label = i + 1
            else:
                pane_label = 1
            label = drawing.Drawing(artist=art, pane=pane_label,
                                    high_model=model_id, mid_model=order,
                                    data_id=i, kind='ref_label')
            obj.add_drawing(label)
        assert len(obj.arts['ref_label']) == 6

        # if visible and in same pane, will be reduced to single label
        # if not, stays the same
        updated = obj.catch_label_overlaps(renderer)
        assert len(updated) == update_count
        assert len(obj.arts['ref_label']) == new_count
        assert obj.arts['ref_label'][0].artist.get_text() == new_text
