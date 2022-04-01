# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.info.base import InfoBase

__all__ = ['OriginationInfo']


class OriginationInfo(InfoBase):

    def __init__(self):
        """
        Initialize the origination information.

        Origination information consists of data such as the organization to
        which the data belongs, the observer responsible for the data,
        who or what created the data, any operators, the original filename and
        description.
        """
        super().__init__()
        self.organization = None
        self.observer = None
        self.creator = None
        self.operator = None
        self.filename = None
        self.descriptor = None

    @property
    def log_id(self):
        """
        Return the string log ID for the info.

        The log ID is used to extract certain information from table data.

        Returns
        -------
        str
        """
        return 'orig'
