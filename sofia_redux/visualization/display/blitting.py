#  Licensed under a 3-clause BSD style license - see LICENSE.rst

from matplotlib.backends.backend_agg import FigureCanvasAgg

from sofia_redux.visualization import log
from sofia_redux.visualization.display.artists import Artists


__all__ = ['BlitManager']


class BlitManager:
    """
    Manage drawing for background and animated artists.

    Parameters
    ----------
    canvas : FigureCanvasAgg
        The canvas to work with, this only works for sub-classes of the Agg
        canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
        `~FigureCanvasAgg.restore_region` methods.
    artists : sofia_redux.visualization.display.artist.Artists
        Artists instance that tracks all artists to draw.

    Attributes
    ----------
    canvas : matplotlib.canvas.Canvas
        The canvas to draw on.
    draw_cid : int
        The matplotlib event CID for the draw_event signal.
    """
    def __init__(self, canvas: FigureCanvasAgg, artists: Artists) -> None:
        self.canvas = canvas
        self._artists = artists
        self._background = None
        self.draw_cid = self.canvas.mpl_connect('draw_event',
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
        self.canvas.mpl_disconnect(self.draw_cid)
        self.canvas.draw()
        self.draw_cid = self.canvas.mpl_connect('draw_event',
                                                self.update_all)

    def update_background(self) -> None:
        """Update the canvas background."""
        self.safe_draw()
        self._background = self.canvas.copy_from_bbox(
            self.canvas.fig.bbox)

    def update_animated(self) -> None:
        """Update all animated artists."""
        self.blit()

    def update_all(self, event=None) -> None:
        """Update the background and artists."""
        self.update_background()
        self._draw_animated()

    def blit(self) -> None:
        """
        Blit the canvas.

        Restore the background without updating it, then update
        the animated artists on top of the restored background.
        """
        self.canvas.restore_region(self._background)
        self._draw_animated()
        self.canvas.blit(self.canvas.fig.bbox)
        self.canvas.flush_events()

    def _draw_animated(self) -> None:
        """Draw all of the animated artists."""
        artists = self._artists.gather_artists(mode='viable')
        log.debug(f'Drawing {len(artists)} artists')
        for artist in artists:
            self.canvas.fig.draw_artist(artist)
