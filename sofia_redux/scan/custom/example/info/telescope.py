# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.info.telescope import TelescopeInfo

__all__ = ['ExampleTelescopeInfo']


class ExampleTelescopeInfo(TelescopeInfo):

    def __init__(self):
        super().__init__()
        self.telescope = "Example Telescope"

    @staticmethod
    def get_telescope_name():
        """
        Return the telescope name.

        Returns
        -------
        name : str
        """
        return "EGTEL"
