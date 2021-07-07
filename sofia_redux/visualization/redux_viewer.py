# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.visualization import eye, log

try:
    from sofia_redux.pipeline.viewer import Viewer
except ImportError:
    Viewer = object
    HAS_PIPELINE = False
else:
    HAS_PIPELINE = True

__all__ = ['EyeViewer']


class EyeViewer(Viewer):
    """
    Redux Viewer interface to the Eye of SOFIA.

    Currently, only FORCAST and EXES spectra are supported for
    display in the Eye viewer.

    Attributes
    ----------
    eye : `eye.Eye`
        Spectral viewer.
    first_display : bool
        Flag that indicates whether the Eye has been initialized
        for display.

    See Also
    --------
    sofia_redux.visualization.controller : standalone Eye application
    """
    def __init__(self):
        if not HAS_PIPELINE:
            raise ImportError(
                'Unable to import Viewer from '
                'sofia_redux.pipeline.viewer.')
        super().__init__()
        self.name = 'EyeViewer'
        self.embedded = False
        self.display_data = list()
        self.parent = None
        self.eye = None
        self.first_display = True

    def start(self, parent=None) -> None:
        """
        Start up the viewer.

        Parameters
        ----------
        parent : QtWidgets.QWidget, optional
            Widget to act as parent to the viewer.
        """
        self.parent = parent

        # for debugging: enable debug level logs
        DEBUG = False
        if DEBUG:  # pragma: no cover
            class Args:
                log_level = 'DEBUG'
                filenames = None
            self.eye = eye.Eye(Args)
        else:
            self.eye = eye.Eye()
        self.eye.set_parent(parent)
        self.first_display = True

    def close(self) -> None:
        """Close the viewer."""
        if self.eye is not None:
            log.debug('Closing the Eye')
            self.eye.close()

    def reset(self) -> None:
        """Reset the viewer."""
        self.close()

    def display(self) -> None:
        """
        Display data.

        Data items to display (filenames or astropy.io.fits.HDUList)
        should be set in the `display_data` attribute by the `update`
        method.

        The display_data should contain only displayable spectra, for
        viewing in the Eye interface.
        """
        if not self.eye:
            self.first_display = True
            return
        if not self.display_data:
            self.close()
            return

        log.debug(f'Updating with: {type(self.display_data)}, '
                  f'{len(self.display_data)}')
        self.eye.reset()
        self.eye.add_panes(kind='spectrum', n_panes=1, layout='rows')
        try:
            self.eye.load(self.display_data)
        except (TypeError, KeyError):
            log.warning('Invalid data; not displaying')
            self.display_data = None
            self.eye.reset()
            return

        self.eye.assign_data(mode='first')
        self.eye.open_eye()

        if self.first_display:
            # set preferred control defaults - must be done after eye is open
            self.eye.toggle_file_panel()
            self.eye.toggle_pane_panel()
            self.eye.toggle_axis_panel()
            self.eye.toggle_controls()
            self.eye.toggle_cursor()

        self.first_display = False

        # trigger refresh
        self.eye.generate()
