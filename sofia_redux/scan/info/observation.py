# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.info.base import InfoBase

__all__ = ['ObservationInfo']


class ObservationInfo(InfoBase):

    def __init__(self):
        super().__init__()
        self.source_name = None
        self.project = None

    @property
    def log_id(self):
        """
        Return the string log ID for the info.

        The log ID is used to extract certain information from table data.

        Returns
        -------
        str
        """
        return 'obs'

    def set_source(self, source_name):
        """
        Set the source name and update configuration options if necessary.

        Parameters
        ----------
        source_name : str
            The source object name.

        Returns
        -------
        None
        """
        self.source_name = source_name
        self.configuration.set_object(self.source_name, validate=True)
