#  Licensed under a 3-clause BSD style license - see LICENSE.rst

import logging
import pytest
import matplotlib.backends.backend_qt5agg as mb
from matplotlib import figure
import matplotlib.lines as ml

from sofia_redux.visualization.display import blitting, artists

PyQt5 = pytest.importorskip('PyQt5')


class TestBlitManager(object):

    def test_init(self):

        fig = figure.Figure()
        can = mb.FigureCanvasAgg(fig)
        art = artists.Artists()
        obj = blitting.BlitManager(can, art)

        assert obj.canvas == can
        assert obj._artists == art
        assert obj._background is None
        assert isinstance(obj.draw_cid, int)

    def test_reset_background(self, blank_blitter):
        blank_blitter._background = list()

        blank_blitter.reset_background()

        assert blank_blitter._background is None

    def test_safe_draw(self, blank_blitter, mocker):
        draw_mock = mocker.patch.object(mb.FigureCanvasQTAgg,
                                        'draw')
        blank_blitter.safe_draw()
        assert draw_mock.called_once()

    def test_update_background(self, blank_blitter, mocker):
        new_bg = 'background'
        attrs = {'copy_from_bbox.return_value': new_bg}
        canvas = mocker.MagicMock(spec=mb.FigureCanvasQTAgg,
                                  fig=figure.Figure(), **attrs)
        draw_mock = mocker.patch.object(blitting.BlitManager,
                                        'safe_draw')
        blank_blitter.canvas = canvas

        blank_blitter.update_background()

        assert draw_mock.called_once()
        assert blank_blitter._background == new_bg

    def test_update_animated(self, blank_blitter, mocker):
        mock = mocker.patch.object(blitting.BlitManager, 'blit')

        blank_blitter.update_animated()

        assert mock.called_once()

    def test_update_all(self, blank_blitter, mocker):
        background = mocker.patch.object(blitting.BlitManager,
                                         'update_background')
        draw = mocker.patch.object(blitting.BlitManager, '_draw_animated')

        blank_blitter.update_all()

        for mock in [background, draw]:
            assert mock.called_once()

    def test_blit(self, blank_blitter, mocker):
        attrs = {'restore_region.return_value': None}
        canvas = mocker.MagicMock(spec=mb.FigureCanvasQTAgg,
                                  fig=figure.Figure(), **attrs)
        blank_blitter.canvas = canvas
        draw = mocker.patch.object(blitting.BlitManager, '_draw_animated')

        blank_blitter.blit()

        assert canvas.restore_region.called_once()
        assert canvas.flush_events.called_once()
        assert draw.called_once()

    def test_draw_animated(self, blank_blitter, mocker, caplog):
        arts = [ml.Line2D([], []), ml.Line2D([], [])]
        caplog.set_level(logging.DEBUG)
        attrs = {'gather_artists.return_value': arts}
        art_mocks = mocker.MagicMock(**attrs)
        canvas = mocker.MagicMock(spec=mb.FigureCanvasQTAgg,
                                  fig=figure.Figure())
        draw = mocker.patch.object(figure.Figure, 'draw_artist')

        blank_blitter.canvas = canvas
        blank_blitter._artists = art_mocks

        blank_blitter._draw_animated()

        assert f'Drawing {len(arts)} artists' in caplog.text
        assert draw.call_count == len(arts)
