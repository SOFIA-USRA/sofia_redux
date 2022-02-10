# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.info.base import InfoBase
from sofia_redux.scan.utilities.utils import insert_info_in_header

__all__ = ['SofiaMissionInfo']


class SofiaMissionInfo(InfoBase):

    def __init__(self):
        super().__init__()
        self.obs_plan_id = ''
        self.base = ''
        self.mission_id = ''
        self.flight_leg = -1

    @property
    def log_id(self):
        """
        Return the string log ID for the info.

        The log ID is used to extract certain information from table data.

        Returns
        -------
        str
        """
        return 'missn'

    def apply_configuration(self):
        options = self.options
        if options is None:
            return
        self.obs_plan_id = options.get_string("PLANID")
        self.base = options.get_string("DEPLOY")
        self.mission_id = options.get_string("MISSN-ID")
        self.flight_leg = options.get_int("FLIGHTLG")

    def edit_header(self, header):
        """
        Edit an image header with available information.

        Parameters
        ----------
        header : astropy.fits.Header
            The FITS header to apply.

        Returns
        -------
        None
        """
        info = [
            ('COMMENT', "<------ SOFIA Mission Data ------>"),
            ('DEPLOY', self.base, 'Aircraft base of operation.'),
            ('MISSN-ID', self.mission_id,
             'unique Mission ID in Mission Plan from MCCS.'),
            ('FLIGHTLG', self.flight_leg, 'Flight leg identifier.')
        ]

        if self.obs_plan_id is not None:
            info.append(('PLANID', self.obs_plan_id,
                         'Observing plan containing all AORs.'))

        insert_info_in_header(header, info, delete_special=True)

    def get_table_entry(self, name):
        """
        Return a parameter value for the given name.

        Parameters
        ----------
        name : str
            The name of the parameter to retrieve.

        Returns
        -------
        value
        """
        if name == 'leg':
            return self.flight_leg
        elif name == 'id':
            return self.mission_id
        elif name == 'plan':
            return self.obs_plan_id
        else:
            return super().get_table_entry(name)
