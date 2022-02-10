# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy import log, units
from astropy.io import fits

from sofia_redux.scan.custom.sofia.scan.scan import SofiaScan

__all__ = ['HawcPlusScan']


class HawcPlusScan(SofiaScan):

    def __init__(self, channels, reduction=None):
        self.prior_pipeline_step = None
        self.use_between_scans = False
        super().__init__(channels, reduction=reduction)

    @property
    def transit_tolerance(self):
        """
        Return the chopper transit tolerance.

        Returns
        -------
        astropy.units.Quantity
        """
        return self.info.chopping.transit_tolerance

    @property
    def focus_t_offset(self):
        """
        Return the focus T offset.

        Returns
        -------
        astropy.units.Quantity
        """
        return self.info.telescope.focus_t_offset

    @property
    def info(self):
        """
        Return the information object for the scan.

        The information object contains the reduction configuration and various
        parameters pertaining the this scan.

        Returns
        -------
        HawcPlusInfo
        """
        return super().info

    def copy(self):
        """
        Return a copy of the HawcPlusScan.

        Returns
        -------
        HawcPlusScan
        """
        return super().copy()

    @property
    def astrometry(self):
        """
        Return the scan astrometry information.

        Returns
        -------
        info : HawcPlusAstrometryInfo
        """
        return super().astrometry

    def get_integration_instance(self):
        """
        Return an integration instance of the correct type for the scan.

        Returns
        -------
        integration : HawcPlusIntegration
        """
        return super().get_integration_instance()

    def get_first_integration(self):
        """
        Return the first integration of the scan.

        Returns
        -------
        integration : HawcPlusIntegration
            Will be `None` if no integrations exist.
        """
        return super().get_first_integration()

    def get_last_integration(self):
        """
        Return the last integration of the scan.

        Returns
        -------
        integration : HawcPlusIntegration
            Will be `None` if no integrations exist.
        """
        return super().get_last_integration()

    def get_first_frame(self):
        """
        Return the first frame of the first integration.

        Returns
        -------
        HawcPlusFrames
        """
        return super().get_first_frame()

    def get_last_frame(self):
        """
        Return the last frame of the last integration.

        Returns
        -------
        HawcPlusFrames
        """
        return super().get_last_frame()

    def __getitem__(self, index):
        """
        Return an integration(s) at the correct index.

        Parameters
        ----------
        index : int or slice

        Returns
        -------
        integration : HawcPlusIntegration or list (HawcPlusIntegration)
        """
        return super().__getitem__(index)

    def edit_scan_header(self, header):
        """
        Edit scan FITS header information.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The header to edit.

        Returns
        -------
        None
        """
        super().edit_scan_header(header)
        if self.prior_pipeline_step is not None:
            header['PROCLEVL'] = (
                self.prior_pipeline_step,
                'Last processing step on input scan.')

    def add_integrations_from_hdul(self, hdul):
        """
        Add integrations to the scan integrations from an open HDUL.

        Any "timestream" HDUs from the HDUList are read by the integration.

        Parameters
        ----------
        hdul : astropy.io.fits.hdu.hdulist.HDUList

        Returns
        -------
        None
        """
        if self.integrations is None:
            self.integrations = []
        data_hdus = []
        for hdu in hdul:
            if not isinstance(hdu, fits.BinTableHDU):
                continue
            extname = str(hdu.header.get('EXTNAME')).lower().strip()
            if extname == 'timestream':
                data_hdus.append(hdu)

        integration = self.get_integration_instance()
        integration.read(data_hdus)
        self.integrations.append(integration)

    def validate(self):
        """
        Validate the scan after a read.

        Returns
        -------
        None
        """
        tolerance = self.configuration.get_float(
            'chopper.tolerance', default=np.nan)
        if not np.isnan(tolerance):
            self.info.chopping.transit_tolerance = (
                tolerance * units.Unit('arcsec'))

        self.use_between_scans = self.configuration.has_option('betweenscans')
        super().validate()

        if not self.have_valid_integrations():
            return

        if self.is_nonsidereal:
            first = self.get_first_frame().object_equatorial
            last = self.get_last_frame().object_equatorial
            offset = last.get_offset_from(first)
            if offset.is_null():
                log.debug("Scan appears to be sidereal with real-time object "
                          "coordinates.")
                self.is_nonsidereal = False

    def get_nominal_pointing_offset(self, native_pointing):
        """
        Get the nominal point offset for a native pointing coordinate.

        The nominal pointing offset ignores the reference coordinate of the
        supplied `native_coordinate` and adds the offset values to the pointing
        offset stored in the configuration.

        Parameters
        ----------
        native_pointing : Offset2D
            The native pointing offset.  The reference position is ignored.

        Returns
        -------
        nominal_pointing_offset: Coordinate2D
        """
        offset = super().get_nominal_pointing_offset(native_pointing)
        offset.subtract(
            self.get_first_integration().get_mean_chopper_position())
        return offset

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
        if name == 'dfoc' or name == 'hawc.dfoc':
            return self.focus_t_offset.to('um')
        return super().get_table_entry(name)
