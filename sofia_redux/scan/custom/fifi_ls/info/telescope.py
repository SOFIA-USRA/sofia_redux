# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.custom.sofia.info.telescope import SofiaTelescopeInfo


arcsec = units.Unit('arcsec')


class FifiLsTelescopeInfo(SofiaTelescopeInfo):

    def __init__(self):
        """
        Initialize the FIFI-LS telescope information.

        Contains information on the FIFI-LS specific telescope.  The telescope
        RA and DEC are offset by the map.
        """
        super().__init__()
        self.delta_map = None

    def apply_configuration(self):
        """
        Update telescope information with the FITS header configuration data.

        Returns
        -------
        None
        """
        super().apply_configuration()
        if self.configuration is None or self.options is None:
            return

        if (self.requested_equatorial is None or
                self.requested_equatorial.is_nan()):
            raise ValueError("No valid OBSRA/OBDEC in header.")

        map_lambda = self.options.get_float('DLAM_MAP', default=0) * arcsec
        map_beta = self.options.get_float('DBET_MAP', default=0) * arcsec
        self.delta_map = Coordinate2D([map_lambda, map_beta])
        self.boresight_equatorial = self.requested_equatorial.copy()
        self.boresight_equatorial.subtract_offset(self.delta_map)
