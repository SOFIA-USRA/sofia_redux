#  Licensed under a 3-clause BSD style license - see LICENSE.rst

from matplotlib.backends.backend_agg import FigureCanvasAgg

from sofia_redux.visualization import log
from sofia_redux.visualization.display.gallery import Gallery


__all__ = ['BlitManager']


class BlitManager(object):
    """
    Manage drawing for background and animated gallery.

    Parameters
    ----------
    canvas : FigureCanvasAgg
        The canvas to work with, this only works for sub-classes of the Agg
        canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
        `~FigureCanvasAgg.restore_region` methods.
    artists : sofia_redux.visualization.display.artist.Gallery
        Gallery instance that tracks all gallery to draw.

    Attributes
    ----------
    canvas : matplotlib.canvas.Canvas
        The canvas to draw on.
    draw_cid : int
        The matplotlib event CID for the draw_event signal.
    """
    def __init__(self, canvas: FigureCanvasAgg, gallery: Gallery,
                 signals) -> None:
        self._canvas = canvas
        self._gallery = gallery
        self._background = None
        self._signals = signals
        self.draw_cid = self._canvas.mpl_connect('draw_event',
                                                 self.update_all)

    def reset_background(self) -> None:
        """Reset the canvas background."""
        self._background = None

    def safe_draw(self) -> None:
        """
        Draw the canvas.

        Disconnect the draw event signal before drawing, and
        reconnect it afterward.
        """
        self._canvas.mpl_disconnect(self.draw_cid)
        self._canvas.draw()
        self.draw_cid = self._canvas.mpl_connect('draw_event',
                                                 self.update_all)

    def update_background(self) -> None:
        """Update the canvas background."""
        self.safe_draw()
        self._background = self._canvas.copy_from_bbox(
            self._canvas.fig.bbox)

    def update_animated(self) -> None:
        """Update all animated gallery."""
        self.blit()

    def update_all(self, event=None) -> None:
        """Update the background and gallery."""
        self.update_background()
        self._draw_animated()
        self._catch_overlaps()

    def blit(self) -> None:
        """
        Blit the canvas.

        Restore the background without updating it, then update
        the animated gallery on top of the restored background.
        """
        self._canvas.restore_region(self._background)
        self._draw_animated()
        self._canvas.blit(self._canvas.fig.bbox)
        self._canvas.flush_events()

    def _draw_animated(self) -> None:
        """Draw all of the animated gallery."""
        artists = self._gallery.gather_artists(mode='viable')
        log.debug(f'Drawing {len(artists)} artists')
        for artist in artists:
            self._canvas.figure.draw_artist(artist)

    def _catch_overlaps(self):
        renderer = self._canvas.get_renderer()
        updated = self._gallery.catch_label_overlaps(renderer)
        if updated:
            self._signals.atrophy_bg_partial.emit()
