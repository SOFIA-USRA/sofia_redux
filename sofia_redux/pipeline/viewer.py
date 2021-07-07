# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Base class for Redux viewers."""

from astropy import log


class Viewer(object):
    """
    Parent class for Redux data viewers.

    This class establishes the API for data viewers.  The display
    method, at a minimum, should be overridden by child classes.
    The display function for this class only logs a message at the
    DEBUG level.

    Attributes
    ----------
    name : str
        Name of the viewer.  This value is used to identify the viewer
        in the reduction classes.
    display_data : list of object
        Data to display.  May contain any object the viewer understands
        how to display.
    parent : widget
        May contain a parent widget to embed the viewer in.
    embedded : bool
        If True, the viewer is build as an embedded widget child of
        the `parent` widget.  This flag may be used by the GUI interface
        (see `sofia_redux.pipeline.gui.main`, for example) to determine
        how to instantiate the viewer.
    """
    def __init__(self):
        """Instantiate the viewer."""
        self.name = "Viewer"
        self.display_data = []
        self.parent = None
        self.embedded = False

    def start(self, parent=None):
        """
        Start the viewer.

        Parameters
        ----------
        parent : widget, optional
            Parent widget for the viewer.
        """
        self.parent = parent

    def display(self):
        """
        Display the data.

        This function only logs the input data.  Override it in
        a subclass to implement custom displays.  See
        `sofia_redux.pipeline.qad.qad_viewer` for an example.
        """
        for datum in self.display_data:
            log.debug("Viewing data: {}".format(datum))

    def update(self, data):
        """
        Update the viewer with new data.

        Data is passed to the `display` method.

        Parameters
        ----------
        data : `list` of object or object
            Data to display.
        """
        if type(data) is not list:
            data = [data]
        self.display_data = data
        self.display()

    def reset(self):
        """Reset the viewer."""
        self.display_data = []
        self.display()

    def close(self):
        """Close the viewer."""
        # no action necessary for default viewer
        return
