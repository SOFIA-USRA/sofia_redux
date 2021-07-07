# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Choose Redux reduction objects based on input data."""

from sofia_redux.pipeline.reduction import Reduction


class Chooser(object):
    """
    Choose Redux reduction objects.

    This class provides the API for Redux choosers. It
    returns a `Reduction` object for all input data.
    Any more complex behavior should be implemented in
    a subclass of this class.

    Attributes
    ----------
    supported: dict
        Keys are the instruments supported by this chooser;
        values are lists of supported modes for the instrument.
    """

    def __init__(self):
        """Initialize the chooser."""
        self.supported = {}

    def choose_reduction(self, data=None, config=None):
        """
        Return a `Reduction` object.

        Parameters
        ----------
        data : `list` of str, optional
            Input data file names.
        config : str, dict, or ConfigObj, optional
            Configuration file or object.  May be any type
            accepted by the `configobj.ConfigObj` constructor.
        """
        return Reduction()
