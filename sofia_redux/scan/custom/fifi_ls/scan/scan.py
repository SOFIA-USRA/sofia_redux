# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.coordinate_systems.coordinate_2d1 import Coordinate2D1
from sofia_redux.scan.custom.sofia.scan.scan import SofiaScan

__all__ = ['FifiLsScan']


class FifiLsScan(SofiaScan):

    def __init__(self, channels, reduction=None):
        """
        Initialize a FIFI-LS scan.

        Parameters
        ----------
        channels : sofia_redux.scan.custom.sofia.channels.camera.SofiaCamera
            The instrument channels for the scan.
        reduction : sofia_redux.scan.reduction.reduction.Reduction, optional
            The reduction to which this scan belongs.
        """
        self.prior_pipeline_step = None
        self.use_between_scans = False
        super().__init__(channels, reduction=reduction)

    @property
    def info(self):
        """
        Return the information object for the scan.

        The information object contains the reduction configuration and various
        parameters pertaining the this scan.

        Returns
        -------
        FifiLsInfo
        """
        return super().info

    def copy(self):
        """
        Return a copy of the FifiLsScan.

        Returns
        -------
        FifiLsScan
        """
        return super().copy()

    @property
    def astrometry(self):
        """
        Return the scan astrometry information.

        Returns
        -------
        info : FifiLsAstrometryInfo
        """
        return super().astrometry

    def get_id(self):
        """
        Return the scan ID.

        FIFI-LS appends a "-uncor" to the ID if using uncorrected FLUX and
        WAVELENGTH data for the reduction.

        Returns
        -------
        str
        """
        scan_id = super().get_id()
        if not self.configuration.get_bool('fifi_ls.uncorrected'):
            return scan_id
        else:
            return f'{scan_id}-uncor'

    def get_integration_instance(self):
        """
        Return an integration instance of the correct type for the scan.

        Returns
        -------
        integration : FifiLsIntegration
        """
        return super().get_integration_instance()

    def get_first_integration(self):
        """
        Return the first integration of the scan.

        Returns
        -------
        integration : FifiLsIntegration
            Will be `None` if no integrations exist.
        """
        return super().get_first_integration()

    def get_last_integration(self):
        """
        Return the last integration of the scan.

        Returns
        -------
        integration : FifiLsIntegration
            Will be `None` if no integrations exist.
        """
        return super().get_last_integration()

    def get_first_frame(self):
        """
        Return the first frame of the first integration.

        Returns
        -------
        FifiLsFrames
        """
        return super().get_first_frame()

    def get_last_frame(self):
        """
        Return the last frame of the last integration.

        Returns
        -------
        FifiLsFrames
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
        integration : FifiLsIntegration or list (FifiLsIntegration)
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

        integration = self.get_integration_instance()
        integration.read(hdul)
        self.integrations.append(integration)

    def validate(self):
        """
        Validate the scan after a read.

        Returns
        -------
        None
        """
        self.use_between_scans = self.configuration.has_option('betweenscans')
        super().validate()

    def get_point_size(self):
        """
        Return the point size of the scan.

        The point size will be the maximum of either the scan or source model
        (if available).

        Returns
        -------
        Coordinate2D1
            The point size.
        """
        point_size = self.info.instrument.get_point_size()
        source_model = self.get_source_model()
        if source_model is None:
            return point_size

        model_size = source_model.get_point_size()
        return Coordinate2D1([max(point_size.x, model_size.x),
                              max(point_size.y, model_size.y),
                              max(point_size.z, model_size.z)])
